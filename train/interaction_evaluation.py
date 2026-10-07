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


def long_reference_controller(policy, defenders):
    from adaptive_interval_rollout import AdaptiveIntervalController
    from jit_nominal_environment import JitNominalEnvironment

    class PaperNominal(JitNominalEnvironment):
        def _isTerminate(self):
            self.protocol = 'paper-parameters-v1'
            return TADEnv._isTerminate(self)

    controller = AdaptiveIntervalController(defenders, policy, blend=1., capture_margin=.5,
        failure_cost='delay', tail_steps=100, tail_policy='candidate', agility_threshold=1.75)
    controller.prediction_environment_class = PaperNominal
    return controller


def long_reference_episode(policy, seed, defenders, agility, trajectory=False):
    from feedback_joint_control import observe

    with preserved_random_state():
        seed_random(seed)
        env = TADEnv(defenders, protocol='paper-parameters-v1')
        obs, _ = env.reset(agility)
        controller = long_reference_controller(policy, defenders)
        np.random.seed(seed + 1_000_000)
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
            summary = dict(arm=arm, cell=cell, episodes=len(selected),
                trained_seeds=len({r['training_seed'] for r in selected}),
                **{m: float(np.mean([float(r[m]) for r in selected])) for m in METRICS})
            groups = [[r for r in selected if int(r['training_seed']) == seed] for seed in SEEDS]
            if all(len(g) == 200 for g in groups):
                summary['ci95'] = {m: hierarchical_interval([[float(r[m]) for r in g] for g in groups],rng) for m in METRICS}
            summaries.append(summary)
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
    expected = {(arm, seed, f'n{n}-a{agility:g}', 350000000+SEEDS.index(seed)*1000000+ci*10000+i)
        for arm in ('full','same_info','arboids_cbf','model_value','short','boids','original','long_reference')
        for seed in SEEDS for ci,(n,agility) in enumerate(CELLS) for i in range(200)}
    result = dict(complete=len(rows)==len(lookup) and set(lookup)==expected, protocol='paper-parameters-v1',
                  summaries=summaries, paired_comparisons=pairs)
    atomic_json(study / 'analysis.json', result)
    return result


def runtime(checkpoint, output, episodes=16):
    policy = DeploymentPolicy(checkpoint)
    rows = []
    for n in (3, 6):
        # Warm both network and the CBF specialization outside the timing sample.
        env = TADEnv(n, protocol='paper-parameters-v1')
        env.reset(2.25)
        p = public_packet(env)
        policy.control(p['obs'], p['motion'], env.boids_actions)
        for i in range(episodes):
            seed_random(400000000 + n*10000 + i)
            obs, _ = env.reset(2.25)
            done = 0
            while not done:
                started = time.perf_counter()
                p = public_packet(env, obs)
                force = policy.control(p['obs'], p['motion'], env.boids_actions, env.Current_T)
                duration = time.perf_counter()-started
                rows.append(dict(defenders=n, scene=i, seconds=duration))
                obs, _, done, _ = env.step(policy.last_action, 'AdaRes', defender_thrust=force)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'calls.csv').open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    summaries=[]
    for n in (3,6):
        values=np.asarray([r['seconds'] for r in rows if r['defenders']==n])
        summaries.append(dict(defenders=n, calls=len(values), median=float(np.median(values)),
            p95=float(np.quantile(values,.95)), p99=float(np.quantile(values,.99)),
            maximum=float(values.max()), deadline_misses=int((values>.2).sum())))
    atomic_json(output/'completed.json', dict(complete=True, summaries=summaries,
        timing_scope='public observation adaptation, candidate exchange emulation, actor and common CBF; excludes external communication'))


def runtime_matrix(study, output, episodes=16):
    from feedback_joint_control import observe
    study, output = Path(study), Path(output)
    manifest = json.loads((study/'manifest.json').read_text(encoding='utf-8'))
    inherited = manifest['inherited_pretraining'].get('pretrain-42')
    pretrain = inherited['path'] if inherited else study/'pretrain/seed-42/actor.pth'
    rows = []
    for arm in ('full','same_info','arboids_cbf','short','model_value','boids','original','long_reference'):
        if arm=='boids': policy=BoidsPolicy()
        elif arm in ('original','long_reference'): policy=OriginalPolicy(pretrain)
        else: policy=DeploymentPolicy(study/f'training/seed-42/{arm}/policy.pth')
        for n in (3,6):
            for i in range(-1,episodes):
                seed_random(400000000+n*10000+i)
                env=TADEnv(n,protocol='paper-parameters-v1')
                obs,_=env.reset(2.25)
                planner=long_reference_controller(policy,n) if arm=='long_reference' else None
                done=0
                while not done:
                    start=time.perf_counter()
                    packet=public_packet(env,obs)
                    if planner:
                        force,_=planner.control(observe(env,obs))
                        action=policy(obs)
                    elif arm in ('boids','original'):
                        action,_=policy.choose_action(packet)
                        force=action[:,2:3]*(750.*action[:,:2]+250.)+(1-action[:,2:3])*env.boids_actions
                    else:
                        force=policy.control(obs,packet['motion'],env.boids_actions)
                        action=policy.last_action
                    elapsed=time.perf_counter()-start
                    if i>=0: rows.append(dict(arm=arm,defenders=n,scene=i,seconds=elapsed))
                    obs,_,done,_=env.step(action,'AdaRes',defender_thrust=force)
    output.mkdir(parents=True,exist_ok=True)
    with (output/'calls.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    summaries=[]
    for arm in sorted({r['arm'] for r in rows}):
        for n in (3,6):
            x=np.array([r['seconds'] for r in rows if r['arm']==arm and r['defenders']==n])
            summaries.append(dict(arm=arm,defenders=n,calls=len(x),median=float(np.median(x)),
                p95=float(np.quantile(x,.95)),p99=float(np.quantile(x,.99)),maximum=float(x.max()),
                deadline_misses=int((x>.2).sum())))
    atomic_json(output/'completed.json',dict(complete=True,summaries=summaries,
        timing_scope='CPU observation adapter and controller, including shared CBF when applicable; excludes external communication'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['task', 'calibration', 'summarize', 'runtime'])
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--pretrain', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', type=Path)
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
    elif args.mode == 'runtime':
        if args.study: runtime_matrix(args.study,args.output)
        else: runtime(args.checkpoint,args.output)
    else:
        summarize(args.output)


if __name__ == '__main__':
    main()
