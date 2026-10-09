"""Five fixed 500k-step defender/attacker phases and common-opponent tests."""
import study_runtime
import argparse
import copy
import csv
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from envs.TADgame import TADEnv
from envs.snapshot import RandomState, seed_random
from policy.SAC import SAC, ReplayBuffer
from policy.networks import ActorAtt
from utils.config import _dict_to_namespace
from interaction_rollout import DeploymentPolicy, public_packet, execute, compact_snapshot
from train_interaction import run_training, atomic_json, atomic_torch, episode
from interaction_evaluation import SEEDS, METRICS


class AttackerPolicy:
    def __init__(self, path=None, *, actor=None):
        self.actor = actor
        if actor is None:
            self.actor = ActorAtt(2, 6, 2, 512).eval().requires_grad_(False)
            self.actor.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))

    @torch.no_grad()
    def __call__(self, obs):
        device = next(self.actor.parameters()).device
        action, _ = self.actor(torch.as_tensor(obs, dtype=torch.float32, device=device)[None], True, False)
        return action[0].cpu().numpy()

    def state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.actor.state_dict().items()}


def attacker_state(agent):
    return dict(actor=agent.actor.state_dict(), critic=agent.critic.state_dict(),
                critic_target=agent.critic_target.state_dict(), actor_optimizer=agent.actor_optimizer.state_dict(),
                critic_optimizer=agent.critic_optimizer.state_dict(), alpha_optimizer=agent.alpha_optimizer.state_dict(),
                log_alpha=agent.log_alpha.detach())


def load_attacker_state(agent, state):
    for name in ('actor', 'critic', 'critic_target', 'actor_optimizer', 'critic_optimizer', 'alpha_optimizer'):
        getattr(agent, name).load_state_dict(state[name])
    with torch.no_grad():
        agent.log_alpha.copy_(state['log_alpha'].cpu())
    agent.alpha = agent.log_alpha.exp().detach().to(agent.device)


def train_attacker(config, defender_checkpoint, output, seed, device, initial=None, steps=500000):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    defender = DeploymentPolicy(defender_checkpoint)
    defender_hash = hashlib.sha256(Path(defender_checkpoint).read_bytes()).hexdigest()
    seed_random(seed)
    agent = SAC(_dict_to_namespace(config), 2, 6, 2, attacker=True, device=torch.device(device))
    replay = ReplayBuffer(8, 2)
    step, env, done, previous_elapsed = 0, None, True, 0.
    saved = None
    if (output / 'resume.pth').exists():
        saved = torch.load(output / 'resume.pth', map_location='cpu', weights_only=False)
        if (saved['seed'] != seed or saved['budget'] != steps or saved.get('config') != config
                or saved.get('defender_sha256') != defender_hash):
            raise ValueError('Attacker resume specification mismatch.')
        load_attacker_state(agent, saved['agent'])
        for k in ('size', 'count', 'max_size'):
            setattr(replay, k, saved['replay'][k])
        for k in ('s', 'a', 'r', 's_', 'dw'):
            value = saved['replay'][k]
            getattr(replay, k)[:len(value)] = value
        env = copy.deepcopy(saved['snapshot'].environment)
        step, done, previous_elapsed = saved['step'], saved['done'], saved['elapsed']
        saved['random'].restore()
        del saved
        metrics = output / 'metrics.csv'
        if metrics.exists():
            with metrics.open(encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames
                rows = [r for r in reader if int(r['step']) <= step]
            with metrics.open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
    elif initial is not None:
        prior = torch.load(initial, map_location='cpu', weights_only=False)
        load_attacker_state(agent, prior['agent'])
        # The defending opponent changed: a fresh replay is mandatory.
        del prior
    started = time.perf_counter()

    def save():
        atomic_torch(output / 'resume.pth', dict(format='interaction-attacker-resume-v1', seed=seed,
            config=config, defender_sha256=defender_hash,
            budget=steps, step=step, done=done, elapsed=previous_elapsed+time.perf_counter()-started,
            agent=attacker_state(agent), snapshot=compact_snapshot(env), random=RandomState.capture(),
            replay={**{k: getattr(replay, k)[:replay.size] for k in ('s', 'a', 'r', 's_', 'dw')},
                    **{k: getattr(replay, k) for k in ('size', 'count', 'max_size')}}))
        atomic_torch(output / 'attacker.pth', {k:v.detach().cpu() for k,v in agent.actor.state_dict().items()})
        atomic_json(output / 'progress.json', dict(step=step, total=steps, complete=step == steps,
                    elapsed=previous_elapsed+time.perf_counter()-started))

    try:
        while step < steps:
            if done:
                env = TADEnv(3, protocol='paper-parameters-v1', LearningSide='Att')
                obs, att_obs = env.reset(2.25, noisy_agility=True)
                packet, done = public_packet(env, obs), False
            else:
                packet = public_packet(env)
                _, att_obs = env._get_obs()
            action, _ = defender.choose_action(packet)
            att_action = np.random.uniform(-1., 1., 2) if step < 50000 else agent.choose_action(att_obs[None])
            _, _, outcome, _, info, following = execute(env, packet, action, attacker_action=att_action)
            done = bool(outcome)
            replay.store(att_obs, att_action, info['attacker_reward'], following, done)
            step += 1
            if step >= 50000:
                agent.learn(replay)
            if step % 5000 == 0 or step == steps:
                attack = AttackerPolicy(actor=agent.actor)
                rows = [episode(defender, 380000000 + seed*100 + i, attacker=attack)[0] for i in range(100)]
                row = dict(step=step, **{m:float(np.mean([r[m] for r in rows])) for m in METRICS})
                exists = (output / 'metrics.csv').exists()
                with (output / 'metrics.csv').open('a', newline='', encoding='utf-8') as f:
                    writer=csv.DictWriter(f, fieldnames=row.keys())
                    if not exists: writer.writeheader()
                    writer.writerow(row)
                save()
                print(f'[IA-ATTACKER] step={step}/{steps}', flush=True)
    except KeyboardInterrupt:
        if env is not None: save()
        raise


def phase(study, arm, seed, stage, device):
    study = Path(study)
    root = study / 'adversarial' / f'seed-{seed}' / arm
    output = root / f'phase-{stage}'
    if stage in (1, 3, 5):
        source = study / 'training' / f'seed-{seed}' / arm if stage == 1 else root / f'phase-{stage-2}'
        config = yaml.safe_load((source / 'config.yaml').read_text(encoding='utf-8'))
        config['training'].update(gate_steps=0, joint_steps=500000, warm_steps=5000,
            initial_state=str(source / 'resume.pth'), pretrain_checkpoint=None, resume=None, fixed_agility=2.25)
        attacker = None if stage == 1 else AttackerPolicy(root / f'phase-{stage-1}' / 'attacker.pth')
        run_training(config, device=device, output=output, attacker=attacker)
    else:
        source = root / f'phase-{stage-1}'
        config = yaml.safe_load((source / 'config.yaml').read_text(encoding='utf-8'))
        train_attacker(config, source / 'policy.pth', output, seed+stage*10000, device,
                       initial=None if stage == 2 else root / f'phase-{stage-2}' / 'resume.pth')


def common_opponents(study, arm, seed):
    study = Path(study)
    base = study / 'adversarial' / f'seed-{seed}'
    opponents = [('APF', None)] + [(f'baseline-phase-{stage}', AttackerPolicy(base / 'arboids_cbf' / f'phase-{stage}' / 'attacker.pth'))
                                  for stage in (2, 4)]
    rows = []
    for stage in (1, 3, 5):
        policy = DeploymentPolicy(base / arm / f'phase-{stage}' / 'policy.pth')
        for name, attacker in opponents:
            for i in range(100):
                row, _ = episode(policy, 390000000 + SEEDS.index(seed)*10000 + i, attacker=attacker)
                rows.append(dict(arm=arm, training_seed=seed, phase=stage, opponent=name, scene=i, **row))
    output = base / arm / 'common-opponents.csv'
    with output.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['phase', 'cross-evaluate'])
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--arm', choices=['arboids_cbf', 'same_info', 'full'], required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--stage', type=int, choices=range(1,6))
    parser.add_argument('--device', default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'phase': phase(args.study,args.arm,args.seed,args.stage,args.device)
    else: common_opponents(args.study,args.arm,args.seed)


if __name__ == '__main__':
    main()
