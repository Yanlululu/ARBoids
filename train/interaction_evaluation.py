"""Independent paired task, intervention-calibration and latency evaluations."""
import study_runtime
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from envs.TADgame import TADEnv
from envs.snapshot import seed_random, preserved_random_state
from policy.networks import ActorAdap
from policy.interaction_sac import make_agent, tensor_packet
from interaction_rollout import (DeploymentPolicy, public_packet, execute, compact_snapshot,
                                FrozenPolicy, frozen_payload, branch_return)
from train_interaction import episode, outcome_row, atomic_json


SEEDS = (42, 101, 202, 303, 404)
CELLS = [(3, a) for a in (1.5, 2., 2.5, 3.)] + [(n, 2.25) for n in range(2, 8)]
METRICS = ('success', 'capture', 'collision', 'breach', 'timeout', 'capture_time')


class OriginalPolicy:
    def __init__(self, path):
        self.actor = ActorAdap(6, 8, 3, 512).eval().requires_grad_(False)
        self.actor.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))

    @torch.no_grad()
    def __call__(self, obs):
        return self.actor(torch.as_tensor(obs, dtype=torch.float32), True, False)[0].numpy()

    def choose_action(self, packet, deterministic=True):
        return self(packet['obs']), 0.


class BoidsPolicy:
    def choose_action(self, packet, deterministic=True):
        return np.zeros((len(packet['obs']), 3), dtype=np.float32), 0.


def long_reference_episode(policy, seed, defenders, agility, trajectory=False):
    from adaptive_interval_rollout import AdaptiveIntervalController
    from jit_nominal_environment import JitNominalEnvironment
    from feedback_joint_control import observe

    class PaperNominal(JitNominalEnvironment):
        def _isTerminate(self):
            self.protocol = 'paper-parameters-v1'
            return TADEnv._isTerminate(self)

    with preserved_random_state():
        seed_random(seed)
        env = TADEnv(defenders, protocol='paper-parameters-v1')
        obs, _ = env.reset(agility)
        np.random.seed(seed + 1_000_000)
        controller = AdaptiveIntervalController(defenders, policy, blend=1., capture_margin=.5,
            failure_cost='delay', tail_steps=100, tail_policy='candidate', agility_threshold=1.75)
        controller.prediction_environment_class = PaperNominal
        done, history = 0, []
        while not done:
            thrust, info = controller.control(observe(env, obs))
            action = policy(obs)
            obs, _, done, _ = env.step(action, 'AdaRes', defender_thrust=thrust)
            if trajectory:
                history.append(dict(time=float(env.Current_T), motion=public_packet(env, obs)['motion'],
                    attacker=env.attacker.pos.copy(), gates=action[:, 2], action=action, thrust=thrust))
        return outcome_row(env, done), history


def save_trajectory(path, history):
    if history:
        keys = ('time', 'motion', 'attacker', 'gates', 'action', 'thrust')
        np.savez_compressed(path, **{k: np.stack([r[k] for r in history]) for k in keys})


def task_evaluation(checkpoint, output, training_seed, arm, episodes=200, cells=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if arm == 'boids':
        policy = BoidsPolicy()
    elif arm in ('original', 'long_reference'):
        policy = OriginalPolicy(checkpoint)
    else:
        policy = DeploymentPolicy(checkpoint)
    csv_path = output / 'episodes.csv'
    fields = ['arm', 'training_seed', 'cell', 'scene_seed', 'defenders', 'agility', *METRICS, 'outcome', 'duration']
    completed = set()
    if csv_path.exists():
        with csv_path.open(encoding='utf-8') as f:
            completed = {(r['cell'], int(r['scene_seed'])) for r in csv.DictReader(f)}
    selected = list(enumerate(CELLS)) if cells is None else [(CELLS.index(c), c) for c in cells]
    for cell_id, (n, agility) in selected:
        cell = f'n{n}-a{agility:g}'
        for i in range(episodes):
            seed = 350000000 + SEEDS.index(training_seed) * 1000000 + cell_id * 10000 + i
            if (cell, seed) in completed:
                continue
            if arm == 'long_reference':
                row, history = long_reference_episode(policy, seed, n, agility, trajectory=i == 0)
            else:
                row, history = episode(policy, seed, n, agility, safety=arm not in ('boids', 'original'), trajectory=i == 0)
            row = dict(arm=arm, training_seed=training_seed, cell=cell, scene_seed=seed, defenders=n, agility=agility, **row)
            exists = csv_path.exists()
            with csv_path.open('a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
            if history:
                save_trajectory(output / f'{cell}.npz', history)
        print(f'[IA-TEST] {arm} seed={training_seed} {cell} complete', flush=True)
    atomic_json(output / 'completed.json', dict(complete=True, arm=arm, training_seed=training_seed,
                episodes_per_cell=episodes, cells=[CELLS[j] for j, _ in selected]))


def calibration(resume, common_pretrain, output, training_seed, states=64, repetitions=4):
    saved = torch.load(resume, map_location='cpu', weights_only=False)
    agent = make_agent(saved['config'])
    if agent.legacy:
        raise ValueError('The legacy individual critic has no joint conditional-value estimate.')
    agent.load_state_dict(saved['agent'])
    policy = FrozenPolicy(frozen_payload(agent, online_critic=True))
    del saved
    source = OriginalPolicy(common_pretrain)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with preserved_random_state():
        for index in range(states):
            seed = 370000000 + SEEDS.index(training_seed) * 1000000 + index
            seed_random(seed)
            env = TADEnv(3, protocol='paper-parameters-v1')
            obs, _ = env.reset(2.25)
            packet = public_packet(env, obs)
            snapshot = compact_snapshot(env)
            for _ in range(10 + index % 40):
                snapshot = compact_snapshot(env)
                action, _ = source.choose_action(packet)
                packet, _, done, _, _, _ = execute(env, packet, action)
                if done:
                    break
            packet = public_packet(snapshot.environment)
            action, _ = policy.action(packet, torch.Generator().manual_seed(seed + 100))
            alternative = action.copy()
            boat, reference = index % 3, (0., .5, 1.)[(index // 3) % 3]
            alternative[boat, 2] = reference
            with torch.no_grad():
                p = {k: v.unsqueeze(0) for k, v in tensor_packet(packet).items()}
                a = policy.critic(p, torch.as_tensor(action).unsqueeze(0))
                b = policy.critic(p, torch.as_tensor(alternative).unsqueeze(0))
                predicted = float((torch.minimum(*a) - torch.minimum(*b)).item())
            differences = []
            count = 0
            for repeat in range(repetitions):
                future, noise = seed + repeat * 10000, seed + repeat * 10000 + 1
                ga, na, _ = branch_return(policy, snapshot, action, future, noise, 300)
                gb, nb, _ = branch_return(policy, snapshot, alternative, future, noise, 300)
                differences.append(ga - gb)
                count += na + nb
            actual = float(np.mean(differences))
            rows.append(dict(state=index, training_seed=training_seed, boat=boat, reference=reference,
                predicted_difference=predicted, environment_difference=actual,
                absolute_error=abs(predicted-actual), monte_carlo_std=float(np.std(differences)), simulated_steps=count))
    with (output / 'calibration.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(output / 'completed.json', dict(complete=True, states=states, repetitions=repetitions,
                E_delta=float(np.mean([r['absolute_error'] for r in rows]))))


def hierarchical_interval(values, rng, repeats=10000):
    """Resample training seeds, then paired scenes within each selected seed."""
    values = np.asarray(values, dtype=float)
    groups, scenes = values.shape
    estimates = []
    for _ in range((repeats + 499) // 500):
        seed_index = rng.integers(groups, size=(500, groups, 1))
        scene_index = rng.integers(scenes, size=(500, groups, scenes))
        estimates.extend(values[seed_index, scene_index].mean(axis=(1, 2)).tolist())
    return [float(x) for x in np.quantile(estimates[:repeats], [.025, .975])]


def summarize(study):
    study = Path(study)
    rows = []
    for path in sorted((study / 'evaluation').glob('seed-*/*/episodes.csv')):
        with path.open(encoding='utf-8') as f:
            rows.extend(csv.DictReader(f))
    if not rows:
        return dict(complete=False, reason='No independent task evaluations completed.')
    lookup = {(r['arm'], int(r['training_seed']), r['cell'], int(r['scene_seed'])): r for r in rows}
    summaries, pairs = [], []
    rng = np.random.default_rng(7102026)
    for arm in sorted({r['arm'] for r in rows}):
        for cell in sorted({r['cell'] for r in rows}):
            selected = [r for r in rows if r['arm'] == arm and r['cell'] == cell]
            if not selected:
                continue
            summaries.append(dict(arm=arm, cell=cell, episodes=len(selected),
                trained_seeds=len({r['training_seed'] for r in selected}),
                **{m: float(np.mean([float(r[m]) for r in selected])) for m in METRICS}))
            if arm == 'full':
                continue
            by_seed = []
            for seed in SEEDS:
                matched = [(r, lookup.get(('full', seed, cell, int(r['scene_seed'])))) for r in selected
                           if int(r['training_seed']) == seed]
                matched = [(a, b) for a, b in matched if b is not None]
                by_seed.append(matched)
            if any(len(group) != 200 for group in by_seed):
                continue
            for metric in METRICS:
                difference = np.asarray([[float(b[metric])-float(a[metric]) for a, b in group] for group in by_seed])
                pairs.append(dict(reference=arm, cell=cell, metric=metric, difference=float(difference.mean()),
                                  ci95=hierarchical_interval(difference, rng)))
    result = dict(complete=len(rows) == 5 * 8 * 10 * 200, protocol='paper-parameters-v1',
                  summaries=summaries, paired_comparisons=pairs)
    atomic_json(study / 'analysis.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['task', 'calibration', 'summarize'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--pretrain', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--arm', default='full')
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--states', type=int, default=64)
    parser.add_argument('--repetitions', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'task':
        task_evaluation(args.checkpoint, args.output, args.seed, args.arm, args.episodes)
    elif args.mode == 'calibration':
        calibration(args.checkpoint, args.pretrain, args.output, args.seed, args.states, args.repetitions)
    else:
        summarize(args.output)


if __name__ == '__main__':
    main()
