"""Complete the six external baselines against the indexed r20 development node.

The pinned source environment, rewards, observations and task lists are shared.
MAPPO uses a decentralized shared Gaussian actor and a centralized PPO critic.
MAAC is the continuous-action adaptation of the attention SAC critic: each
actor gradient holds its teammates' sampled actions fixed. Neither uses the
proposed fusion/intervention modules. Reported endpoint evaluation uses the
original source integrator; training and training-validation can use the
checked scalar integrator below.

Algorithm references:
https://github.com/marlbenchmark/on-policy
https://github.com/shariqiqbal2810/MAAC
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from source_arboids import SOURCE_COMMIT, numerical_source, sha256

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / 'train/experiments/author-source-fusion-20261011'
NODE = EXPERIMENTS / 'local-difference-joint-development/RESULT_NODE_20261020_043146_M2_R20.json'
OUTPUT = EXPERIMENTS / 'r20-six-baseline-comparison'
SUITES = ('source-development', 'new-development', 'fresh-200', 'new4310-200')
SOURCE = numerical_source()
CPU_SUBNORMAL_RUNTIME_PREDECESSOR = 'de35d4cb57e363d8ee77c276d2424d1876c0016d34e51193f8398b15f794b662'
CPU_THREADS_RUNTIME_PREDECESSOR = '44d8bf5a318cf9b56b52ff87e4423a5e287958afa5db3f40593e6d13d8447c9b'
CHECKPOINT_RUNTIME_PREDECESSOR = '0c10c1678bd7e9eb9d6fbc0db6085f9ea588282048eaecc61f30737b6169a2ad'


def exclusive_run(operation):
    def decorate(function):
        @wraps(function)
        def run(algorithm, seed=42, *args, **kwargs):
            directory = OUTPUT / (algorithm if algorithm == 'boids' else f'{algorithm}-seed{seed}')
            directory.mkdir(parents=True, exist_ok=True)
            if os.name != 'posix':
                return function(algorithm, seed, *args, **kwargs)
            import fcntl
            with (directory / ('.' + operation + '.lock')).open('a+b') as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError(f'{operation} already has an active writer for {directory.name}') from error
                try:
                    return function(algorithm, seed, *args, **kwargs)
                finally:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return run
    return decorate


@contextmanager
def checkpoint_storage_lock():
    """Serialize large checkpoint I/O across independently trained seeds."""
    if os.name != 'posix':
        yield
        return
    import fcntl
    with (Path(tempfile.gettempdir()) / 'arboids-r20-checkpoint.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_training_checkpoint(path, device='cpu'):
    path = Path(path)
    if path.suffix != '.zst':
        return torch.load(path, map_location=device, weights_only=False)
    import zstandard
    staging = '/dev/shm' if Path('/dev/shm').is_dir() else None
    with checkpoint_storage_lock(), tempfile.TemporaryFile(dir=staging) as decoded:
        with path.open('rb') as source:
            zstandard.ZstdDecompressor().copy_stream(source, decoded)
        decoded.seek(0)
        return torch.load(decoded, map_location=device, weights_only=False)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    if path.suffix == '.zst':
        import zstandard
        with checkpoint_storage_lock():
            fraction = min(1., value.get('steps', 0) / max(1, value.get('spec', {}).get('budget', 1)))
            required_free = int(.35 * 2**30 + 1.5e9 * fraction)
            while shutil.disk_usage(path.parent).free < required_free:
                emit('baseline_disk_wait', path=str(path), required_free_bytes=required_free,
                     free_bytes=shutil.disk_usage(path.parent).free)
                time.sleep(10)
            with temporary.open('wb') as output:
                with zstandard.ZstdCompressor(level=1).stream_writer(output, closefd=False) as encoded:
                    torch.save(value, encoded)
            os.replace(temporary, path)
    else:
        torch.save(value, temporary)
        os.replace(temporary, path)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temporary, path)


def emit(phase, **values):
    print(json.dumps(dict(phase=phase, **values), ensure_ascii=False), flush=True)


def indexed_results():
    index = json.loads(NODE.read_text(encoding='utf-8'))
    model = index['artifacts']['model:F11']
    if sha256(ROOT / model['relative_path']) != model['sha256']:
        raise ValueError('The user-selected r20 model differs from the indexed model.')
    results = {}
    for suite in SUITES:
        artifact = index['artifacts']['eval:F11:' + suite]
        path = ROOT / artifact['relative_path']
        if sha256(path) != artifact['sha256']:
            raise ValueError('An indexed r20 evaluation has changed: ' + suite)
        data = torch.load(path, map_location='cpu', weights_only=False)
        if not data['completed'] or len(data['rows']) != len(data['tasks']) or any(r is None for r in data['rows']):
            raise ValueError('An r20 evaluation is incomplete.')
        if data['policy_sha256'] != model['sha256']:
            raise ValueError('An r20 evaluation belongs to a different policy.')
        results[suite] = data
    return index, results


class ScalarSourceVessel(SOURCE.WAMV):
    """Same fixed WAM-V equations without repeated 3x3 matrix allocation.

    This implementation is restricted to the source's fixed, straight
    thrusters and diagonal inertia. Random currents still come from the
    unmodified environment, in the original draw order.
    """
    def step(self, action, current_velocity=np.zeros(3)):
        if self.left_pos != 0. or self.right_pos != 0.:
            raise ValueError('The scalar training integrator requires fixed thrusters.')
        left = float(np.clip(action[0], self.min_thrust, self.max_thrust))
        right = float(np.clip(action[1], self.min_thrust, self.max_thrust))
        self.left_thrust, self.right_thrust = left, right
        x, y, theta = float(self.x), float(self.y), float(self.theta)
        vxr, vyr, wr = map(float, self.velocity_r)
        cx, cy, cw = map(float, current_velocity)
        for _ in range(self.N):
            vx, vy, w = vxr + cx, vyr + cy, wr + cw
            x += vx * self.dt
            y += vy * self.dt
            theta = self.wrap_to_2pi(theta + w * self.dt)
            c, s = np.cos(theta), np.sin(theta)
            ur, vr = c * vxr + s * vyr, -s * vxr + c * vyr
            u, v = c * vx + s * vy, -s * vx + c * vy
            bx = self.m * w * v - (100. + 150. * abs(ur)) * ur + left + right
            by = -self.m * w * u + 20. * ur * w - (100. + 150. * abs(vr)) * vr
            bw = -20. * ur * vr - (980. + 950. * abs(w)) * w + 1.25 * (left - right)
            ur += self.dt * bx / 380.
            vr += self.dt * by / 400.
            wr = w + self.dt * bw / 1430.
            vxr, vyr = c * ur - s * vr, s * ur + c * vr
        self.x, self.y, self.theta = x, y, theta
        self.pos[0], self.pos[1] = x, y
        self.velocity_r = np.array([vxr, vyr, wr])
        self.velocity = np.array([vx, vy, w])
        self.vel = self.velocity[:2]


def make_environment(*, fast=False):
    env = SOURCE.TADEnv(6, True, True)
    if fast:
        env.attacker.__class__ = ScalarSourceVessel
        for boat in env.defender_list:
            boat.__class__ = ScalarSourceVessel
    return env


def layer(n_in, n_out, gain=math.sqrt(2.)):
    result = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(result.weight, gain)
    nn.init.zeros_(result.bias)
    return result


class Actor(nn.Module):
    def __init__(self, observation_dim=24, hidden=128, algorithm='mappo'):
        super().__init__()
        self.algorithm = algorithm
        self.body = nn.Sequential(nn.LayerNorm(observation_dim), layer(observation_dim, hidden),
                                  nn.Tanh(), layer(hidden, hidden), nn.Tanh())
        self.mean = layer(hidden, 2, .01)
        if algorithm == 'mappo':
            self.log_std = nn.Parameter(torch.full((2,), -.5))
        else:
            self.log_std = layer(hidden, 2, .01)
            nn.init.constant_(self.log_std.bias, -.5)

    def distribution(self, observations):
        features = self.body(observations)
        mean = self.mean(features)
        log_std = (self.log_std.expand_as(mean) if self.algorithm == 'mappo'
                   else self.log_std(features)).clamp(-5., 2.)
        return torch.distributions.Normal(mean, log_std.exp())

    def forward(self, observations, *, deterministic=False, latent=None):
        distribution = self.distribution(observations)
        if latent is None:
            latent = distribution.mean if deterministic else distribution.rsample()
        log_prob = distribution.log_prob(latent).sum(-1)
        log_prob -= (2. * (math.log(2.) - latent - F.softplus(-2. * latent))).sum(-1)
        return latent.tanh(), log_prob, latent, distribution.entropy().sum(-1)


class CentralValue(nn.Module):
    """Joint public observations at training; PopArt keeps physical values fixed."""
    def __init__(self, hidden=128):
        super().__init__()
        self.body = nn.Sequential(nn.LayerNorm(144), layer(144, hidden), nn.Tanh(),
                                  layer(hidden, hidden), nn.Tanh())
        self.head = layer(hidden, 6, 1.)
        self.register_buffer('mean', torch.zeros(()))
        self.register_buffer('std', torch.ones(()))
        self.register_buffer('count', torch.tensor(1e-4))
        self.register_buffer('m2', torch.tensor(1e-4))

    def forward(self, observations, normalized=False):
        value = self.head(self.body(observations.flatten(-2)))
        return value if normalized else value * self.std + self.mean

    @torch.no_grad()
    def update_scale(self, returns):
        values = returns.flatten()
        old_mean, old_std = self.mean.clone(), self.std.clone()
        n = values.numel()
        delta = values.mean() - self.mean
        total = self.count + n
        self.m2.add_(values.var(unbiased=False) * n + delta.square() * self.count * n / total)
        self.mean.add_(delta * n / total)
        self.count.copy_(total)
        self.std.copy_((self.m2 / self.count).sqrt().clamp_min(.01))
        self.head.weight.mul_(old_std / self.std)
        self.head.bias.copy_((old_std * self.head.bias + old_mean - self.mean) / self.std)


class AttentionCritic(nn.Module):
    """Four attention heads over teammates' observation-action embeddings."""
    def __init__(self, hidden=128, heads=4):
        super().__init__()
        self.hidden, self.heads = hidden, heads
        self.state = nn.Sequential(nn.LayerNorm(24), layer(24, hidden), nn.LeakyReLU())
        self.state_action = nn.Sequential(nn.LayerNorm(26), layer(26, hidden), nn.LeakyReLU())
        self.query = layer(hidden, hidden, 1.)
        self.key = layer(hidden, hidden, 1.)
        self.value = nn.Sequential(layer(hidden, hidden, 1.), nn.LeakyReLU())
        self.output = nn.Sequential(layer(2 * hidden, hidden), nn.LeakyReLU(), layer(hidden, 1, 1.))

    def forward(self, observations, actions, *, actor_gradient=False):
        batch, agents, _ = observations.shape
        own = self.state_action(torch.cat((observations, actions), -1))
        peers = (self.state_action(torch.cat((observations, actions.detach()), -1))
                 if actor_gradient else own)
        dimension = self.hidden // self.heads
        q = self.query(self.state(observations)).reshape(batch, agents, self.heads, dimension).transpose(1, 2)
        k = self.key(peers).reshape(batch, agents, self.heads, dimension).transpose(1, 2)
        v = self.value(peers).reshape(batch, agents, self.heads, dimension).transpose(1, 2)
        logits = q @ k.transpose(-2, -1) / math.sqrt(dimension)
        mask = torch.eye(agents, dtype=torch.bool, device=observations.device)
        regularizer = 1e-3 * logits.masked_select(~mask).square().mean()
        weights = logits.masked_fill(mask, -torch.inf).softmax(-1)
        context = (weights @ v).transpose(1, 2).reshape(batch, agents, self.hidden)
        return self.output(torch.cat((own, context), -1)).squeeze(-1), regularizer


class TrainingWorlds:
    """Separate RNG streams for eight complete source environments."""
    def __init__(self, seed, count=8):
        self.envs = [make_environment(fast=True) for _ in range(count)]
        self.rng = []
        self.observations = []
        self.episodes = 0
        self.completed_episodes = []
        for index, env in enumerate(self.envs):
            np.random.seed((seed * 1000003 + 773000000 + index * 7919) % (2**32))
            obs, _ = env.reset(2., noisy_agility=True)
            self.observations.append(obs)
            self.rng.append(np.random.get_state())
        self.observations = np.asarray(self.observations, dtype=np.float32)

    def step(self, actions, steps, budget):
        successors, rewards, dones, reset_observations = [], [], [], []
        for index, (env, action) in enumerate(zip(self.envs, actions)):
            np.random.set_state(self.rng[index])
            obs, reward, done, _ = env.step(action, 'RL')
            successors.append(obs)
            rewards.append(reward)
            dones.append(bool(done))
            if done:
                self.episodes += 1
                self.completed_episodes.append(int(done > 2))
                self.completed_episodes = self.completed_episodes[-100:]
                agility = 2. + min(3, int(4 * steps / budget)) * .25
                obs, _ = env.reset(agility, noisy_agility=True)
            reset_observations.append(obs)
            self.rng[index] = np.random.get_state()
        self.observations = np.asarray(reset_observations, dtype=np.float32)
        return (np.asarray(successors, dtype=np.float32), np.asarray(rewards, dtype=np.float32),
                np.asarray(dones, dtype=np.float32))


def row_outcome(env, done, total, steps):
    return dict(success=int(done > 2), capture=int(done == 3), collision=int(done == 2),
                source_failure=int(done == 1), timeout=int(done == 4),
                physical_breach=int(np.linalg.norm(env.attacker.pos) < env.Target_R),
                duration=float(env.Current_T), outcome=int(done), discounted_return=total,
                physical_steps=steps, environment_protocol='source')


_ACTORS = {}


def evaluation_episode(job):
    algorithm, checkpoint, scene, future = job
    torch.set_num_threads(1)
    actor = None
    if algorithm != 'boids':
        if checkpoint not in _ACTORS:
            record = torch.load(checkpoint, map_location='cpu', weights_only=False)
            actor = Actor(algorithm=algorithm)
            actor.load_state_dict(record['actor'])
            _ACTORS[checkpoint] = actor.eval()
        actor = _ACTORS[checkpoint]
    seed, defenders, agility = scene
    if defenders != 6:
        raise ValueError('This comparison is the selected six-vessel r20 cohort.')
    np.random.seed(seed)
    env = make_environment()
    obs, _ = env.reset(agility, noisy_agility=False)
    if future is not None:
        np.random.seed(future)
    done, total, discount, steps = 0, 0., 1., 0
    while not done:
        if actor is None:
            action = np.zeros((6, 2))
        else:
            with torch.no_grad():
                action = actor(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)[0].numpy()
        obs, reward, done, _ = env.step(action, 'Boids' if actor is None else 'RL')
        total += discount * float(np.mean(reward))
        discount *= .99
        steps += 1
    row = dict(seed=seed, defenders=defenders, agility=agility, **row_outcome(env, done, total, steps))
    if future is not None:
        row['future_seed'] = future
    return row


@exclusive_run('evaluation')
def evaluate_endpoint(algorithm, seed=42, workers=2):
    _, reference = indexed_results()
    directory = OUTPUT / (algorithm if algorithm == 'boids' else f'{algorithm}-seed{seed}')
    checkpoint = None if algorithm == 'boids' else str(directory / 'best.pth')
    digest = None if checkpoint is None else sha256(checkpoint)
    if algorithm != 'boids':
        training = torch.load(directory / 'training-result.pth', map_location='cpu', weights_only=False)
        if not training['completed'] or training['steps'] != training['budget']:
            raise ValueError('Endpoint comparison requires the complete training budget.')
        if training['best_sha256'] != digest:
            raise ValueError('The selected baseline checkpoint changed after training.')
    for suite, ref in reference.items():
        path = directory / (suite + '.pth')
        old = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else None
        if old and (old['tasks'] != ref['tasks'] or old['policy_sha256'] != digest):
            raise ValueError('A resumed endpoint evaluation changed its paired tasks or policy.')
        if old and old['completed']:
            continue
        rows = old['rows'] if old else [None] * len(ref['tasks'])
        jobs = [(j, (algorithm, checkpoint, scene, future)) for j, (scene, future) in enumerate(ref['tasks'])
                if rows[j] is None]
        start = time.monotonic()
        with (ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)) as pool:
            if pool:
                pending = {pool.submit(evaluation_episode, job): j for j, job in jobs}
                stream = ((pending[future], future.result()) for future in as_completed(pending))
            else:
                stream = ((j, evaluation_episode(job)) for j, job in jobs)
            for j, row in stream:
                rows[j] = row
                finished = [r for r in rows if r is not None]
                if len(finished) % 32 == 0 or len(finished) == len(rows):
                    record = dict(algorithm=algorithm, training_seed=None if algorithm == 'boids' else seed,
                                  tasks=ref['tasks'], rows=rows, completed=len(finished) == len(rows),
                                  policy=checkpoint, policy_sha256=digest, source_commit=SOURCE_COMMIT,
                                  runtime_sha256=sha256(__file__), evaluated=len(finished),
                                  success=sum(r['success'] for r in finished),
                                  environment='pinned-source-original-integrator', development_only=True,
                                  elapsed_seconds=time.monotonic() - start)
                    atomic_save(record, path)
                    emit('baseline_endpoint', algorithm=algorithm, seed=seed, suite=suite,
                         evaluated=len(finished), total=len(rows), success=record['success'])
    return summarize()


def summarize():
    index, reference = indexed_results()
    result = dict(node_id=index['node_id'], role='paired development comparison',
                  r20_policy_sha256=index['artifacts']['model:F11']['sha256'],
                  suite_order=list(SUITES), methods={}, completed=False)
    r20_rows = [row for suite in SUITES for row in reference[suite]['rows']]
    result['methods']['r20'] = dict(success=[reference[s]['success'] for s in SUITES],
                                  total_success=sum(r['success'] for r in r20_rows), episodes=len(r20_rows))
    entries = [('boids', OUTPUT / 'boids')]
    for algorithm in ('mappo', 'maac'):
        entries += [(p.name, p) for p in sorted(OUTPUT.glob(algorithm + '-seed*')) if p.is_dir()]
    for algorithm in ('sac', 'rp', 'arboids'):
        entries += [(f'{algorithm}-seed{seed}', EXPERIMENTS / 'independent-training' /
                     f'source-six-{algorithm}-seed{seed}-20261021') for seed in range(42, 47)]
    for name, directory in entries:
        rows, counts, paths = [], [], []
        for suite in SUITES:
            path = directory / (suite + '.pth')
            if not path.exists():
                break
            record = torch.load(path, map_location='cpu', weights_only=False)
            if record['tasks'] != reference[suite]['tasks']:
                raise ValueError('A baseline task list differs from r20: ' + str(path))
            if not record['completed'] or any(row is None for row in record['rows']):
                break
            rows += record['rows']
            counts.append(sum(row['success'] for row in record['rows']))
            paths.append(dict(path=str(path.relative_to(ROOT)), sha256=sha256(path)))
        if len(counts) != len(SUITES):
            result['methods'][name] = dict(completed=False, completed_suites=len(counts), success=counts)
            continue
        rescues = sum(bool(a['success']) and not b['success'] for a, b in zip(r20_rows, rows))
        harms = sum(not a['success'] and bool(b['success']) for a, b in zip(r20_rows, rows))
        result['methods'][name] = dict(completed=True, success=counts, total_success=sum(counts),
                                      episodes=len(rows), rate_percent=100. * sum(counts) / len(rows),
                                      r20_gain_percentage_points=100. * (rescues - harms) / len(rows),
                                      r20_rescues=rescues, r20_harms=harms, evaluations=paths)
    required = ('boids', 'mappo-seed42', 'maac-seed42', 'sac-seed42', 'rp-seed42', 'arboids-seed42')
    result['completed'] = all(result['methods'].get(name, {}).get('completed', False) for name in required)
    atomic_json(result, OUTPUT / 'comparison.json')
    emit('comparison_summary', completed=result['completed'],
         results={name: {k: v for k, v in row.items() if k in ('completed', 'success', 'total_success',
                 'r20_gain_percentage_points')} for name, row in result['methods'].items()})
    return result


def resource_guard():
    """This one CPU training process stays within the shared remote RAM limit."""
    current = Path('/sys/fs/cgroup/memory.current')
    maximum = Path('/sys/fs/cgroup/memory.max')
    if not current.exists() or not maximum.exists() or maximum.read_text().strip() == 'max':
        return
    while int(current.read_text()) > int(maximum.read_text()) - 4 * 2**30:
        emit('baseline_resource_wait', used_gib=int(current.read_text()) / 2**30,
             limit_gib=int(maximum.read_text()) / 2**30)
        time.sleep(20)


@torch.no_grad()
def training_validation(actor, seed, episodes=50):
    """Fixed training-only validation layouts, disjoint from the r20 suites."""
    cpu_actor = copy.deepcopy(actor).cpu().eval()
    previous_numpy = np.random.get_state()
    wins, returns, steps = 0, [], 0
    for index in range(episodes):
        np.random.seed(769000000 + seed * 10000 + index)
        env = make_environment(fast=True)
        obs, _ = env.reset(2., noisy_agility=False)
        done = 0
        total = 0.
        while not done:
            action = cpu_actor(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)[0].numpy()
            obs, reward, done, _ = env.step(action, 'RL')
            total += float(np.mean(reward))
            steps += 1
        wins += int(done > 2)
        returns.append(total)
    np.random.set_state(previous_numpy)
    return dict(success=wins, episodes=episodes, success_rate=wins / episodes,
                mean_return=float(np.mean(returns)), physical_steps=steps)


class Replay:
    def __init__(self, capacity):
        self.capacity, self.size, self.position = capacity, 0, 0
        self.obs = np.empty((capacity, 6, 24), dtype=np.float32)
        self.next_obs = np.empty_like(self.obs)
        self.actions = np.empty((capacity, 6, 2), dtype=np.float32)
        self.rewards = np.empty((capacity, 6), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.float32)

    def append(self, obs, actions, rewards, successors, dones):
        indices = (self.position + np.arange(len(obs))) % self.capacity
        self.obs[indices], self.actions[indices], self.rewards[indices] = obs, actions, rewards
        self.next_obs[indices], self.dones[indices] = successors, dones
        self.position = int((self.position + len(obs)) % self.capacity)
        self.size = min(self.size + len(obs), self.capacity)

    def sample(self, batch_size, generator, device):
        indices = generator.integers(0, self.size, size=batch_size)
        return [torch.as_tensor(array[indices], device=device) for array in
                (self.obs, self.actions, self.rewards, self.next_obs, self.dones)]


def ppo_update(actor, critic, actor_optimizer, critic_optimizer, batch, *, epochs=10, minibatch=256):
    obs, latent, old_logp, old_value, returns, advantage = batch
    critic.update_scale(returns)
    advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)
    target = (returns - critic.mean) / critic.std
    previous = (old_value - critic.mean) / critic.std
    metrics, updates = {}, 0
    for _ in range(epochs):
        order = torch.randperm(len(obs), device=obs.device)
        stop = False
        for start in range(0, len(obs), minibatch):
            indices = order[start:start + minibatch]
            _, logp, _, entropy = actor(obs[indices], latent=latent[indices])
            log_ratio = logp - old_logp[indices]
            ratio = log_ratio.exp()
            surrogate = torch.minimum(ratio * advantage[indices],
                                      ratio.clamp(.8, 1.2) * advantage[indices])
            actor_loss = -surrogate.mean() - .01 * entropy.mean()
            value = critic(obs[indices], normalized=True)
            clipped = previous[indices] + (value - previous[indices]).clamp(-.2, .2)
            value_loss = .5 * torch.maximum(F.smooth_l1_loss(value, target[indices], reduction='none'),
                                           F.smooth_l1_loss(clipped, target[indices], reduction='none')).mean()
            if not torch.isfinite(actor_loss + value_loss):
                raise FloatingPointError('Nonfinite MAPPO update.')
            actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), .5)
            actor_optimizer.step()
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), .5)
            critic_optimizer.step()
            updates += 1
            kl = ((ratio - 1.) - log_ratio).mean().detach()
            metrics = dict(actor_loss=float(actor_loss.detach()), value_loss=float(value_loss.detach()),
                           approximate_kl=float(kl), value_std=float(critic.std))
            if kl > .03:
                stop = True
                break
        if stop:
            break
    return updates, metrics


def maac_update(actor, critic, target_actor, target_critic, actor_optimizer, critic_optimizer, batch):
    obs, actions, rewards, successors, dones = batch
    alpha, gamma, tau = .01, .99, .001
    with torch.no_grad():
        next_actions, next_logp, _, _ = target_actor(successors)
        next_q, _ = target_critic(successors, next_actions)
        target = rewards + gamma * (1. - dones[:, None]) * (next_q - alpha * next_logp)
    q, regularizer = critic(obs, actions)
    critic_loss = F.mse_loss(q, target) + regularizer
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 10.)
    critic_optimizer.step()
    for parameter in critic.parameters():
        parameter.requires_grad_(False)
    proposed, logp, _, _ = actor(obs)
    value, _ = critic(obs, proposed, actor_gradient=True)
    actor_loss = (alpha * logp - value).mean()
    if not torch.isfinite(critic_loss + actor_loss):
        raise FloatingPointError('Nonfinite continuous MAAC update.')
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), .5)
    actor_optimizer.step()
    for parameter in critic.parameters():
        parameter.requires_grad_(True)
    with torch.no_grad():
        for target_module, module in ((target_actor, actor), (target_critic, critic)):
            for destination, source in zip(target_module.parameters(), module.parameters()):
                destination.lerp_(source, tau)
    return dict(actor_loss=float(actor_loss.detach()), critic_loss=float(critic_loss.detach()),
                mean_q=float(q.detach().mean()))


@exclusive_run('training')
def train(algorithm, seed=42, budget=1000000, device='cpu', cpu_threads=1, checkpoint_root=None):
    if algorithm not in ('mappo', 'maac') or budget % 8:
        raise ValueError('Choose MAPPO or continuous MAAC and a budget divisible by eight.')
    if not isinstance(cpu_threads, int) or cpu_threads < 1:
        raise ValueError('CPU thread count must be a positive integer.')
    torch.set_num_threads(cpu_threads)
    if not torch.set_flush_denormal(True):
        raise RuntimeError('This runtime cannot enable the checked CPU denormal handling.')
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(os, 'nice'):
        os.nice(5)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    directory = OUTPUT / f'{algorithm}-seed{seed}'
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / 'training-result.pth'
    if final_path.exists():
        result = torch.load(final_path, map_location='cpu', weights_only=False)
        if result['budget'] != budget or result['algorithm'] != algorithm or result['seed'] != seed:
            raise ValueError('Existing training has a different specification.')
        if result['completed']:
            return result
    spec = dict(algorithm=algorithm, seed=seed, budget=budget, training_defenders=6,
                environment='pinned source', source_commit=SOURCE_COMMIT,
                actor='shared 128x128 Gaussian actor, local original observations, two thrusters',
                reward='unmodified per-vessel source rewards', curriculum='source: 2.0,2.25,2.5,2.75 with +/-0.25',
                gamma=.99, workers=1, environments=8, device=device,
                checkpoint_selection='highest training-only 50-episode success, then mean return, first exact tie',
                validation_interval=25000, training_sampling_backend='checked scalar source equations',
                endpoint_backend='original source equations', runtime_sha256=sha256(__file__),
                cpu_flush_denormal=True, cpu_threads=cpu_threads,
                checkpoint_codec='zstd-lossless' if algorithm == 'maac' else 'torch',
                result_role='development', fresh_initialization=True)
    if algorithm == 'mappo':
        spec.update(critic='centralized joint-observation value with PopArt',
                    learning_rate=.0005, gae_lambda=.95, rollout_team_steps=2048,
                    ppo_epochs=10, minibatch_team_steps=256, clip=.2, entropy_coefficient=.01,
                    maximum_gradient_norm=.5, kl_stop=.03)
    else:
        spec.update(critic='shared four-head 128-wide attention critic',
                    action_adaptation='continuous SAC reparameterization; teammate actions detached for actor gradients',
                    learning_rate=.001, target_tau=.001, entropy_coefficient=.01,
                    warmup_team_steps=50000, replay_capacity=min(1000000, budget),
                    batch_team_steps=1024, updates_per_100_team_steps=4)
    actor = Actor(algorithm=algorithm).to(device)
    critic = (CentralValue() if algorithm == 'mappo' else AttentionCritic()).to(device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=spec['learning_rate'], eps=1e-5)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=spec['learning_rate'], eps=1e-5,
                                        weight_decay=0. if algorithm == 'mappo' else 1e-3)
    target_actor = copy.deepcopy(actor) if algorithm == 'maac' else None
    target_critic = copy.deepcopy(critic) if algorithm == 'maac' else None
    worlds = TrainingWorlds(seed)
    generator = np.random.default_rng(seed + 90000)
    replay = Replay(min(budget, 1000000)) if algorithm == 'maac' else None
    steps, updates, validation_steps = 0, 0, 0
    history = []
    resume_events = []
    selected_checkpoint = None
    best = (-1., -math.inf)
    checkpoint_directory = (Path(checkpoint_root) / directory.name) if checkpoint_root else directory
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_directory / ('last.pth.zst' if algorithm == 'maac' else 'last.pth')
    if last_path.is_symlink():
        last_path = last_path.resolve()
    resume_path = next((p for p in (last_path, directory / 'last.pth.zst', directory / 'last.pth')
                        if p.exists()), None)
    if resume_path is not None:
        saved = load_training_checkpoint(resume_path, device)
        resume_events = list(saved.get('resume_events', []))
        if saved['spec'] != spec:
            previous = dict(saved['spec'])
            device_changed = previous['device'] != device
            if device_changed and not (algorithm == 'maac' and saved['updates'] == 0):
                raise ValueError('Device migration is restricted to MAAC before its first optimizer update.')
            previous.update(runtime_sha256=spec['runtime_sha256'], cpu_flush_denormal=True,
                            cpu_threads=cpu_threads, device=device,
                            checkpoint_codec=spec['checkpoint_codec'])
            compatible_runtimes = (CPU_SUBNORMAL_RUNTIME_PREDECESSOR,
                                   CPU_THREADS_RUNTIME_PREDECESSOR,
                                   CHECKPOINT_RUNTIME_PREDECESSOR, spec['runtime_sha256'])
            if (saved['spec']['runtime_sha256'] not in compatible_runtimes
                    or previous != spec):
                raise ValueError('Resume requires the same baseline configuration and source code.')
            resume_events.append(dict(steps=saved['steps'],
                previous_runtime_sha256=saved['spec']['runtime_sha256'],
                runtime_sha256=spec['runtime_sha256'],
                previous_cpu_flush_denormal=saved['spec'].get('cpu_flush_denormal', False),
                cpu_flush_denormal=True,
                previous_cpu_threads=saved['spec'].get('cpu_threads', 1), cpu_threads=cpu_threads,
                previous_device=saved['spec']['device'], device=device,
                checkpoint_codec=spec['checkpoint_codec'], resumed_checkpoint=str(resume_path),
                reason='Execution and lossless checkpoint storage only; algorithm settings unchanged'))
        actor.load_state_dict(saved['actor'])
        critic.load_state_dict(saved['critic'])
        actor_optimizer.load_state_dict(saved['actor_optimizer'])
        critic_optimizer.load_state_dict(saved['critic_optimizer'])
        if target_actor is not None:
            target_actor.load_state_dict(saved['target_actor'])
            target_critic.load_state_dict(saved['target_critic'])
        worlds, generator, replay = saved['worlds'], saved['generator'], saved['replay']
        steps, updates, validation_steps = saved['steps'], saved['updates'], saved['validation_steps']
        history, best = saved['history'], tuple(saved['best'])
        expected_selection = max(history, key=lambda row: (row['success_rate'], row['mean_return']))
        selected_checkpoint = saved.get('best_checkpoint')
        current_selection = torch.load(directory / 'best.pth', map_location='cpu', weights_only=False)
        if selected_checkpoint is None and current_selection['selection'] == expected_selection:
            selected_checkpoint = current_selection
        if selected_checkpoint is None and expected_selection['steps'] == steps:
            selected_checkpoint = dict(actor={k: v.detach().cpu() for k, v in saved['actor'].items()},
                                       algorithm=algorithm, seed=seed, training_steps=steps,
                                       selection=expected_selection, spec=saved['spec'])
        if selected_checkpoint is None or selected_checkpoint['selection'] != expected_selection:
            raise ValueError('The complete checkpoint cannot recover its validation-selected policy.')
        if current_selection['selection'] != expected_selection:
            atomic_save(selected_checkpoint, directory / 'best.pth')
            resume_events.append(dict(steps=steps, restored_selection_steps=expected_selection['steps'],
                                     discarded_unsaved_tail_selection_steps=current_selection['training_steps'],
                                     reason='Restore the selected policy belonging to the last complete checkpoint'))
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.startswith('cuda') and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        emit('baseline_resumed', algorithm=algorithm, seed=seed, steps=steps)
    start = time.monotonic()
    start_steps = steps
    next_validation = (steps // 25000 + 1) * 25000
    next_progress = (steps // 5000 + 1) * 5000
    next_guard = (steps // 1024 + 1) * 1024
    since_update = steps % 100
    metrics = {}
    emit('baseline_training_start', **spec)
    while steps < budget:
        if steps >= next_guard or steps == start_steps:
            resource_guard()
            next_guard = steps + 1024
        if algorithm == 'mappo':
            trajectory = []
            horizon = min(256, (budget - steps) // 8)
            for _ in range(horizon):
                obs = torch.as_tensor(worlds.observations, device=device)
                with torch.no_grad():
                    actions, logp, latent, _ = actor(obs)
                    values = critic(obs)
                _, rewards, dones = worlds.step(actions.cpu().numpy(), steps, budget)
                trajectory.append((obs, latent.detach(), logp.detach(), values.detach(),
                                   torch.as_tensor(rewards, device=device), torch.as_tensor(dones, device=device)))
                steps += 8
            with torch.no_grad():
                following = critic(torch.as_tensor(worlds.observations, device=device))
                advantage = torch.zeros_like(following)
                advantages, returns = [], []
                for _, _, _, value, reward, done in reversed(trajectory):
                    mask = 1. - done[:, None]
                    delta = reward + .99 * mask * following - value
                    advantage = delta + .99 * .95 * mask * advantage
                    advantages.append(advantage)
                    returns.append(advantage + value)
                    following = value
                batch = [torch.cat([item[k] for item in trajectory], 0) for k in range(4)]
                batch += [torch.cat(returns[::-1], 0), torch.cat(advantages[::-1], 0)]
            added, metrics = ppo_update(actor, critic, actor_optimizer, critic_optimizer, batch)
            updates += added
        else:
            obs = worlds.observations.copy()
            if steps < 50000:
                actions = generator.uniform(-1., 1., size=(8, 6, 2)).astype(np.float32)
            else:
                with torch.no_grad():
                    actions = actor(torch.as_tensor(obs, device=device))[0].cpu().numpy()
            successors, rewards, dones = worlds.step(actions, steps, budget)
            replay.append(obs, actions, rewards, successors, dones)
            steps += 8
            since_update += 8
            if since_update >= 100:
                since_update %= 100
                if steps >= 50000 and replay.size >= 1024:
                    for _ in range(4):
                        metrics = maac_update(actor, critic, target_actor, target_critic,
                                              actor_optimizer, critic_optimizer, replay.sample(1024, generator, device))
                        updates += 1
        if steps >= next_progress or steps == budget:
            emit('baseline_training_progress', algorithm=algorithm, seed=seed, steps=steps,
                 budget=budget, updates=updates, elapsed_seconds=time.monotonic() - start,
                 team_steps_per_second=(steps - start_steps) / max(1., time.monotonic() - start),
                 recent_training_success=float(np.mean(worlds.completed_episodes)) if worlds.completed_episodes else 0.,
                 **metrics)
            next_progress = (steps // 5000 + 1) * 5000
        if steps >= next_validation or steps == budget:
            validation = training_validation(actor, seed)
            validation_steps += validation['physical_steps']
            entry = dict(steps=steps, updates=updates, **validation)
            history.append(entry)
            score = (validation['success_rate'], validation['mean_return'])
            if score > best:
                best = score
                selected_checkpoint = dict(actor={k: v.detach().cpu().clone() for k, v in actor.state_dict().items()},
                                           algorithm=algorithm, seed=seed, training_steps=steps,
                                           selection=entry, spec=spec)
                atomic_save(selected_checkpoint, directory / 'best.pth')
            checkpoint = dict(spec=spec, actor=actor.state_dict(), critic=critic.state_dict(),
                              actor_optimizer=actor_optimizer.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
                              worlds=worlds, generator=generator, replay=replay, steps=steps, updates=updates,
                              validation_steps=validation_steps, history=history, best=best,
                              best_checkpoint=selected_checkpoint,
                              resume_events=resume_events,
                              torch_rng=torch.get_rng_state(),
                              cuda_rng=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None,
                              target_actor=target_actor.state_dict() if target_actor is not None else None,
                              target_critic=target_critic.state_dict() if target_critic is not None else None)
            atomic_save(checkpoint, last_path)
            emit('baseline_training_validation', algorithm=algorithm, seed=seed, **entry)
            next_validation = (steps // 25000 + 1) * 25000
    record = dict(algorithm=algorithm, seed=seed, budget=budget, steps=steps, updates=updates,
                  completed=True, fresh_initialization=True, spec=spec, validation_history=history,
                  resume_events=resume_events,
                  validation_physical_steps=validation_steps, best_sha256=sha256(directory / 'best.pth'),
                  last_sha256=sha256(last_path), last_checkpoint=str(last_path),
                  elapsed_seconds=time.monotonic() - start)
    atomic_save(record, final_path)
    emit('baseline_training_complete', algorithm=algorithm, seed=seed, steps=steps, best=best)
    return record


def check(cpu_threads=1):
    """Check changed physics arithmetic, likelihoods, PopArt and MAAC gradients."""
    torch.set_num_threads(cpu_threads)
    if not torch.set_flush_denormal(True):
        raise RuntimeError('CPU denormal handling is unavailable.')
    torch.manual_seed(9341)
    maximum_error = 0.
    checked_steps = 0
    for seed in range(8):
        np.random.seed(768000000 + seed)
        original = make_environment()
        original.reset(2. + .25 * (seed % 4), noisy_agility=False)
        accelerated = copy.deepcopy(original)
        accelerated.attacker.__class__ = ScalarSourceVessel
        for vessel in accelerated.defender_list:
            vessel.__class__ = ScalarSourceVessel
        stream = np.random.default_rng(seed + 5771)
        for step in range(400):
            action = stream.uniform(-1., 1., (6, 2))
            controller = 'Boids' if seed % 2 else 'RL'
            rng = np.random.get_state()
            obs, reward, done, _ = original.step(action, controller)
            after = np.random.get_state()
            np.random.set_state(rng)
            fast_obs, fast_reward, fast_done, _ = accelerated.step(action, controller)
            np.random.set_state(after)
            error = float(np.max(np.abs(obs - fast_obs)))
            maximum_error = max(maximum_error, error)
            checked_steps += 1
            if error > 1e-7 or not np.allclose(reward, fast_reward, atol=1e-8, rtol=1e-8) or done != fast_done:
                raise AssertionError(('Source integrator mismatch', seed, step, error, done, fast_done))
            if done:
                break
    actor = Actor()
    obs = torch.randn(32, 6, 24)
    actions, logp, latent, _ = actor(obs)
    replay_logp = actor(obs, latent=latent)[1]
    torch.testing.assert_close(logp, replay_logp, rtol=0., atol=0.)
    assert bool((actions.abs() <= 1).all())
    critic = CentralValue()
    before = critic(obs).detach()
    critic.update_scale(torch.randn(32, 6) * 30. - 120.)
    torch.testing.assert_close(before, critic(obs), atol=3e-5, rtol=3e-5)
    attention = AttentionCritic()
    action_leaf = torch.randn(32, 6, 2, requires_grad=True)
    q, _ = attention(obs, action_leaf, actor_gradient=True)
    gradient = torch.autograd.grad(q[:, 0].sum(), action_leaf)[0]
    assert float(gradient[:, 0].abs().sum()) > 0.
    assert float(gradient[:, 1:].abs().max()) == 0.
    maac_actor = Actor(algorithm='maac')
    maac_critic = AttentionCritic()
    target_actor, target_critic = copy.deepcopy(maac_actor), copy.deepcopy(maac_critic)
    actor_optimizer = torch.optim.Adam(maac_actor.parameters(), lr=.001)
    critic_optimizer = torch.optim.Adam(maac_critic.parameters(), lr=.001)
    data = [obs, actions.detach(), torch.randn(32, 6), torch.randn(32, 6, 24), torch.zeros(32)]
    metrics = maac_update(maac_actor, maac_critic, target_actor, target_critic,
                          actor_optimizer, critic_optimizer, data)
    assert all(np.isfinite(value) for value in metrics.values())
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=.0005)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=.0005)
    values = critic(obs).detach()
    count, _ = ppo_update(actor, critic, actor_optimizer, critic_optimizer,
                          [obs, latent.detach(), logp.detach(), values, values + 1., torch.randn(32, 6)],
                          epochs=1, minibatch=16)
    assert count > 0
    emit('baseline_checks_passed', source_steps=checked_steps, maximum_observation_error=maximum_error,
         likelihood_replay=True, popart_value_preservation=True, maac_own_action_gradient_only=True,
         both_learning_updates_finite=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job', choices=('check', 'boids', 'train', 'evaluate', 'summarize', 'run'))
    parser.add_argument('--algorithm', choices=('mappo', 'maac'), default='mappo')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--budget', type=int, default=1000000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--cpu-threads', type=int, default=1)
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.job == 'check':
        check(args.cpu_threads)
    elif args.job == 'boids':
        evaluate_endpoint('boids', workers=args.workers)
    elif args.job == 'train':
        train(args.algorithm, args.seed, args.budget, args.device, args.cpu_threads, args.checkpoint_root)
    elif args.job == 'evaluate':
        evaluate_endpoint(args.algorithm, args.seed, args.workers)
    elif args.job == 'summarize':
        summarize()
    else:
        check(args.cpu_threads)
        evaluate_endpoint('boids', workers=args.workers)
        for algorithm in ('mappo', 'maac'):
            train(algorithm, args.seed, args.budget, args.device, args.cpu_threads, args.checkpoint_root)
            evaluate_endpoint(algorithm, args.seed, args.workers)
        summarize()


if __name__ == '__main__':
    main()
