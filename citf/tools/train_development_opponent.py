"""Short, explicitly developmental opponent/adaptation segments.

Reuse the selected main model and the prescribed attacker and defender learners.
The formal five-stage, 500k-per-stage entry remains unchanged.
"""
import argparse
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, os.environ.get('CITF_CODE_ROOT', str(Path(__file__).resolve().parents[1]/'code')))
from unified.adversary import Attacker, AttackerLearner, observation, attacker_reward
from unified.environment import FleetEnv, TaskConfig, fusion
from unified.learner import Learner, Replay
from unified.networks import CITF, ModelConfig
from unified.protocol import ATTACKER_RECIPE
from unified.train import (seed_policy, run_generators, random_state, restore_random_state,
                           policy_schedule, write_json, save)
from unified.train_alternating import evaluate_pair


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--role', choices=['attacker', 'defender'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--eval-interval', type=int, default=10000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    source = torch.load(args.previous, map_location='cpu', weights_only=False)
    continuation = source.get('role') == args.role
    previous_pair = 'defender' in source
    defender_payload = source['defender'] if previous_pair else source
    recipe = source['defender_recipe'] if previous_pair else source['arguments']
    attacker_recipe = source['attacker_recipe'] if previous_pair else dict(ATTACKER_RECIPE)
    counts = source['side_transitions'].copy() if previous_pair else dict(defender=source['transitions'], attacker=0)
    starting_counts = counts.copy()
    seed = recipe['seed']
    phase_seed = seed+(20000 if args.role == 'attacker' else 30000)
    seed_policy(phase_seed, args.device)
    defender = CITF(ModelConfig(**defender_payload['config'])).to(args.device)
    defender_learner = Learner(defender, lr=recipe['lr'], tau=recipe['target_tau'], value_huber_delta=recipe['value_huber_delta'])
    defender_learner.restore(defender_payload)
    task = TaskConfig(**source['task'])
    attacker = Attacker(task.defenders, delta=task.thrust_delta).to(args.device)
    attacker_learner = AttackerLearner(attacker, attacker_recipe['lr'])
    if previous_pair:
        attacker_learner.restore(source['attacker'])
    if not continuation:
        defender_learner.target.load_state_dict(defender.state_dict())
        attacker_learner.target.load_state_dict(attacker.state_dict())
    train_defender = args.role == 'defender'
    active_recipe = recipe if train_defender else attacker_recipe
    defender.requires_grad_(train_defender)
    attacker.requires_grad_(not train_defender)
    env = FleetEnv(active_recipe['envs'], task, phase_seed+100000, args.device)
    shapes = None if train_defender else dict(s=(8+7*task.defenders,), a=(2,), u=(2,), r=(),
                                            next_s=(8+7*task.defenders,), done=(), event=(), seconds=())
    replay = Replay(active_recipe['capacity'], task.defenders, args.device, shapes=shapes)
    generators = run_generators(phase_seed, args.device)
    if continuation:
        replay.load_state_dict(source['training_state']['replay'])
        env.load_state_dict(source['training_state']['environment'])
        restore_random_state(source['training_state']['random'], args.device, generators)
    configuration = dict(research_stage='development', role=args.role, source=str(args.previous),
                         registered_additional_transitions=args.steps, starting_side_transitions=starting_counts,
                         same_side_continuation=continuation, task=asdict(task), training_seed=seed,
                         scene_seed=73003, policy_seed=83003, episodes=100,
                         defender_recipe=recipe, attacker_recipe=attacker_recipe)
    write_json(args.out/'configuration.json', configuration)
    apf_baseline = source.get('apf_baseline')
    if apf_baseline is None:
        apf_baseline = evaluate_pair(defender, None, task, 100, 73003, 83003, args.device)
    initial = source['evaluation'] if previous_pair else evaluate_pair(defender, attacker, task, 100, 73003, 83003, args.device)
    write_json(args.out/'initial.json', dict(opponent=initial, apf=apf_baseline, reused_previous_endpoint=previous_pair))
    del source, defender_payload
    gc.collect()
    transitions, metrics = 0, {}
    next_eval = min(args.eval_interval, args.steps)
    started = last_report = time.perf_counter()
    while transitions < args.steps:
        count = min(active_recipe['envs'], next_eval-transitions, args.steps-transitions)
        active = torch.arange(active_recipe['envs'], device=args.device) < count
        with torch.no_grad():
            x = env.observation()
            if train_defender:
                schedule = policy_schedule(recipe, counts['defender'])
                result = defender.select(x, schedule['endpoint_noise'], generators['exploration'])
                random_g = torch.rand(result['g'].shape, device=args.device, generator=generators['exploration'])
                explore = torch.rand(active_recipe['envs'], 1, device=args.device, generator=generators['exploration']) < schedule['probability']
                defense = fusion(x, result['a'], result['b'], torch.where(explore, random_g, result['g']), task.thrust_delta)
                attack = attacker.select(x, env.agility)['action']
            else:
                defense = defender.select(x)['action']
                result = attacker.select(x, env.agility, deterministic=False)
                if not continuation and transitions < active_recipe['start_learning']:
                    result['latent'] = 2*torch.rand(active_recipe['envs'], 2, device=args.device, generator=generators['exploration'])-1
                    result['action'] = attacker.command(x, result['latent'], env.agility)
                attack = result['action']
            next_x, reward, done, individual = env.step(defense, attack, active)
            if train_defender:
                replay.add(x=x[:count], u=env.executed_defender_action()[:count], r=reward[:count], ri=individual[:count],
                           next_x=next_x[:count], done=done[:count],
                           **{key: value[:count] for key, value in env.replay_metadata().items()})
            else:
                replay.add(s=result['state'][:count], a=result['latent'][:count], u=env.previous[:count, -1],
                           r=attacker_reward(x, next_x, env.outcome, task.target_radius)[:count],
                           next_s=observation(next_x, env.agility)[:count], done=done[:count],
                           event=env.outcome[:count], seconds=env.elapsed[:count])
            transitions += count
            counts[args.role] += count
            env.reset((done & active).nonzero().flatten())
        if replay.size >= active_recipe['start_learning']:
            if train_defender:
                schedule = policy_schedule(recipe, counts['defender'])
                defender_learner.set_policy_learning_rate(schedule['policy_lr'])
            for _ in range(active_recipe['updates_per_step']):
                if train_defender:
                    fraction = recipe['value_terminal_fraction']
                    batch = replay.sample_value(active_recipe['batch'], generators['replay'], fraction, recipe['value_importance_exponent'])
                    states = (replay.sample(active_recipe['batch'], generators['replay'])
                              if fraction and (defender_learner.updates+1) % defender_learner.policy_delay == 0 else None)
                    metrics = defender_learner.update(batch, schedule['temperature'], schedule['feedback_scale'], states,
                                                     robust_value=schedule['robust_value'])
                else:
                    metrics = attacker_learner.update(replay.sample(active_recipe['batch'], generators['replay']))
        if time.perf_counter()-last_report >= 30:
            status = dict(status='training', role=args.role, transitions=transitions, side_transitions=counts,
                          seconds=time.perf_counter()-started, metrics={k: float(v) for k, v in metrics.items()})
            write_json(args.out/'status.json', status)
            print(json.dumps(status), flush=True)
            last_report = time.perf_counter()
        if transitions == next_eval:
            result = evaluate_pair(defender, attacker, task, 100, 73003, 83003, args.device)
            result.update(research_stage='development', role=args.role, transitions=transitions, side_transitions=counts.copy())
            state = dict(role=args.role, research_stage='development', phase_transitions=transitions,
                         registered_transitions=args.steps, side_transitions=counts.copy(), source=str(args.previous),
                         defender=defender_learner.payload(), attacker=attacker_learner.payload(),
                         defender_recipe=recipe, attacker_recipe=attacker_recipe, task=asdict(task),
                         apf_baseline=apf_baseline, evaluation=result,
                         training_state=dict(replay=replay.state_dict(), environment=env.state_dict(),
                                             random=random_state(args.device, generators)))
            save(args.out/'last.pt', state)
            del state
            with (args.out/'evaluations.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(result)+'\n')
            print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)
            next_eval = min(next_eval+args.eval_interval, args.steps)
    if train_defender:
        retention = evaluate_pair(defender, None, task, 100, 73003, 83003, args.device)
        write_json(args.out/'apf_retention.json', retention)
    write_json(args.out/'status.json', dict(status='completed', role=args.role, transitions=transitions,
               side_transitions=counts, final={k: v for k, v in result.items() if k != 'rows'}))


if __name__ == '__main__':
    main()
