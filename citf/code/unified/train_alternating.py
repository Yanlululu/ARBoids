"""The four new 500k stages after the reusable main-training defender stage."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch

from . import IMPLEMENTATION_VERSION
from .environment import FleetEnv, TaskConfig, fusion
from .networks import CITF, ModelConfig
from .learner import Learner, Replay
from .adversary import Attacker, AttackerLearner, observation, attacker_reward
from .train import (write_json, save, append_evaluation, seed_policy, device_index, run_generators,
                    random_state, restore_random_state, resume_arguments,
                    prepare_output, validate_training_arguments, rollout_count, policy_schedule)
from .protocol import (ATTACKER_RECIPE, load_registry, validate_checkpoint, checkpoint_identity,
                       outcome_metrics, phase_identity, phase_counts, validate_phase_checkpoint,
                       phase_checkpoint_identity, chain_evaluation_identity, attach_identity, validate_result)


def evaluate_pair(defender, attacker, task, count, scene_seed, policy_seed, device):
    index = device_index(device)
    with torch.random.fork_rng(devices=[] if index is None else [index]):
        seed_policy(policy_seed, device)
        env = FleetEnv(count, task, seed=scene_seed, device=device)
        while not bool((env.outcome != 0).all()):
            with torch.no_grad():
                x = env.observation()
                command = defender.select(x)['action']
                attack = None if attacker is None else attacker.select(x, env.agility)['action']
                env.step(command, attack)
        rows = env.records()
    return dict(scene_seed=scene_seed, policy_seed=policy_seed, task=asdict(task),
                implementation_version=IMPLEMENTATION_VERSION,
                **outcome_metrics(rows, count, scene_seed, task.duration), rows=rows)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', required=True)
    parser.add_argument('--phase', type=int, choices=[2, 3, 4, 5], required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--steps', type=int, default=500000)
    parser.add_argument('--seed', type=int, default=2101)
    parser.add_argument('--envs', type=int)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--updates-per-step', type=int)
    parser.add_argument('--capacity', type=int)
    parser.add_argument('--start-learning', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--eval-interval', type=int, default=50000)
    parser.add_argument('--eval-count', type=int, default=100)
    parser.add_argument('--eval-seed', type=int, default=73003)
    parser.add_argument('--test-seed', type=int, default=74003)
    parser.add_argument('--policy-eval-seed', type=int, default=83003)
    parser.add_argument('--stage', choices=['development', 'formal'], default='development')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume')
    parser.add_argument('--registry', help='Common formal-study registry.')
    parser.set_defaults(init_from=None)
    args = parser.parse_args()
    resumed = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if resumed is not None:
        resume_arguments(args, resumed)
        if args.registry is None:
            args.registry = resumed['arguments'].get('registry')
    if args.stage == 'formal' and not args.registry:
        raise ValueError('Formal alternating training requires the common study registry.')
    registry = load_registry(args.registry) if args.registry else None
    if resumed is not None:
        validate_phase_checkpoint(resumed, registry, args.phase, completed=False)
    if args.stage == 'formal' and (args.steps != 500000 or args.seed != 2101 or args.eval_count != 100
            or args.eval_seed != 73003 or args.test_seed != 74003 or args.policy_eval_seed != 83003):
        raise ValueError('Formal phases retain seed2101, 500000 team transitions and the registered 100-scene banks.')
    source = torch.load(args.previous, map_location='cpu', weights_only=False)
    if source.get('implementation_version') != IMPLEMENTATION_VERSION:
        raise ValueError('Continue the unified controller with its matching implementation version.')
    if args.phase == 2:
        if source.get('transitions') != 500000 or 'config' not in source:
            raise ValueError('Phase 2 requires the reusable 500k main defender checkpoint.')
        root_run = validate_checkpoint(source, registry, 'full', args.seed, 500000)
        source_checkpoint = checkpoint_identity(args.previous, source)
        defender_payload, attacker_payload = source, None
        recipe = root_run['arguments']
        attacker_recipe = dict(ATTACKER_RECIPE)
        counts = dict(defender=500000, attacker=0)
    else:
        if source.get('phase') != args.phase-1 or source['transitions'] != 500000:
            raise ValueError('Continue directly from the preceding completed 500k phase.')
        source_phase = validate_phase_checkpoint(source, registry, args.phase-1)
        source_checkpoint = phase_checkpoint_identity(args.previous, source)
        root_run = source_phase['root_run']
        defender_payload, attacker_payload = source['defender'], source['attacker']
        recipe, counts = source['defender_recipe'], source['side_transitions'].copy()
        attacker_recipe = source['attacker_recipe']
    train_defender = args.phase in (3, 5)
    defaults = attacker_recipe
    training_overrides = {}
    for key, default in defaults.items():
        inherited = recipe[key] if train_defender else default
        if getattr(args, key) is None:
            setattr(args, key, inherited)
        elif getattr(args, key) != inherited:
            training_overrides[key] = dict(inherited=inherited, actual=getattr(args, key))
    if training_overrides and (train_defender or args.phase != 2):
        raise ValueError('Every later phase retains its own side\'s selected recipe.')
    if args.phase == 2:
        attacker_recipe = {key: getattr(args, key) for key in ATTACKER_RECIPE}
    validate_training_arguments(args)
    if args.stage == 'formal' and source['arguments']['stage'] != 'formal':
        raise ValueError('The registered formal chain starts from a formal main training seed.')
    if args.seed != recipe['seed']:
        raise ValueError('The chain must keep the registered main-training seed.')
    task = TaskConfig(**source['task'])
    if task.agility != 2. or task.defenders != 3:
        raise ValueError('The alternating experiment has three defenders and fixed agility 2.0.')
    identity = phase_identity(vars(args), asdict(task), root_run, source_checkpoint, attacker_recipe, counts,
                              dict(defender=defender_payload['updates'],
                                   attacker=0 if attacker_payload is None else attacker_payload['updates']), registry)
    if resumed is not None and resumed.get('phase_identity') != identity:
        raise ValueError('Exact phase resume requires the same parent checkpoint and complete recipe.')
    out = Path(args.out)
    prepare_output(out, resumed)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    phase_seed = args.seed+10000*args.phase
    seed_policy(phase_seed, args.device)
    defender = CITF(ModelConfig(**defender_payload['config'])).to(args.device)
    defender_learner = Learner(defender, lr=recipe['lr'], tau=recipe['target_tau'])
    defender_learner.restore(defender_payload)
    attacker = Attacker(task.defenders, delta=task.thrust_delta).to(args.device)
    attacker_learner = AttackerLearner(attacker, attacker_recipe['lr'])
    if attacker_payload is not None:
        attacker_learner.restore(attacker_payload)
    # Each newly changed opponent starts a new replay and synchronized targets.
    defender_learner.target.load_state_dict(defender.state_dict())
    attacker_learner.target.load_state_dict(attacker.state_dict())
    del source, defender_payload, attacker_payload
    active_side = 'defender' if train_defender else 'attacker'
    defender.requires_grad_(train_defender)
    attacker.requires_grad_(not train_defender)
    env = FleetEnv(args.envs, task, seed=phase_seed+100000, device=args.device)
    shapes = None if train_defender else dict(s=(8+7*task.defenders,), a=(2,), u=(2,), r=(),
                                               next_s=(8+7*task.defenders,), done=(), event=(), seconds=())
    replay = Replay(args.capacity, task.defenders, args.device, shapes=shapes)
    generators = run_generators(phase_seed, args.device)
    transitions, metrics, evaluation = 0, {}, None
    last_checkpoint = None
    if resumed is not None:
        defender_learner.restore(resumed['defender'])
        attacker_learner.restore(resumed['attacker'])
        saved = resumed['training_state']
        replay.load_state_dict(saved['replay'])
        env.load_state_dict(saved['environment'])
        restore_random_state(saved['random'], args.device, generators)
        transitions, counts = resumed['transitions'], resumed['side_transitions'].copy()
        evaluation = resumed['evaluation']
        last_checkpoint = phase_checkpoint_identity(args.resume, resumed)
        del saved, resumed
    write_json(out/'configuration.json', dict(arguments=vars(args), task=asdict(task),
               implementation_version=IMPLEMENTATION_VERSION, active_side=active_side,
               defender_recipe=recipe, attacker_recipe=attacker_recipe, phase_identity=identity,
               phase_seed=phase_seed, training_overrides=training_overrides,
               attacker_reward='breach +100; capture/timeout -100; defender collision 0; otherwise radial progress/15',
               stage_transition='retain both policies and optimizers; new active replay; synchronize targets'))
    if args.phase == 2 and not args.resume:
        baseline = evaluate_pair(defender, None, task, args.eval_count, args.test_seed, args.policy_eval_seed, args.device)
        baseline.update(phase=1, transitions=500000, training_seed=args.seed, opponent='APF',
                        role='fixed-stage-endpoint', research_stage=args.stage)
        if registry is not None:
            attach_identity(baseline, chain_evaluation_identity(registry, 1, source_checkpoint, asdict(task), 'fixed-stage-endpoint'))
        write_json(out/'phase1_apf.json', baseline)
    started = last_report = time.perf_counter()
    next_eval = ((transitions//args.eval_interval)+1)*args.eval_interval
    try:
        while transitions < args.steps:
            count = rollout_count(args, transitions, next_eval)
            active = torch.arange(args.envs, device=args.device) < count
            with torch.no_grad():
                x = env.observation()
                if train_defender:
                    schedule = policy_schedule(recipe, counts['defender'])
                    result = defender.select(x, schedule['endpoint_noise'], generators['exploration'])
                    random_g = torch.rand(result['g'].shape, device=args.device, generator=generators['exploration'])
                    explore = torch.rand(args.envs, 1, device=args.device, generator=generators['exploration']) < schedule['probability']
                    g = torch.where(explore, random_g, result['g'])
                    defense = fusion(x, result['a'], result['b'], g, task.thrust_delta)
                    attack = attacker.select(x, env.agility)['action']
                else:
                    defense = defender.select(x)['action']
                    result = attacker.select(x, env.agility, deterministic=False)
                    if transitions < args.start_learning:
                        result['latent'] = 2*torch.rand(args.envs, 2, device=args.device, generator=generators['exploration'])-1
                        result['action'] = attacker.command(x, result['latent'], env.agility)
                    attack = result['action']
                next_x, reward, done, individual_reward = env.step(defense, attack, active)
                if train_defender:
                    replay.add(x=x[:count], u=env.executed_defender_action()[:count], r=reward[:count],
                               ri=individual_reward[:count], next_x=next_x[:count], done=done[:count],
                               **{key: value[:count] for key, value in env.replay_metadata().items()})
                else:
                    reward = attacker_reward(x, next_x, env.outcome, task.target_radius)
                    replay.add(s=result['state'][:count], a=result['latent'][:count], u=env.previous[:count, -1],
                               r=reward[:count], next_s=observation(next_x, env.agility)[:count],
                               done=done[:count], event=env.outcome[:count], seconds=env.elapsed[:count])
                transitions += count
                counts[active_side] += count
                env.reset((done & active).nonzero().flatten())
            if replay.size >= args.start_learning:
                if train_defender:
                    schedule = policy_schedule(recipe, counts['defender'])
                    policy_lr = schedule['policy_lr']
                    defender_learner.set_policy_learning_rate(policy_lr)
                    metrics['policy_lr'] = policy_lr
                for _ in range(args.updates_per_step):
                    if train_defender:
                        fraction = recipe.get('value_terminal_fraction', 0.)
                        batch = replay.sample_value(args.batch, generators['replay'], fraction,
                                                    recipe.get('value_importance_exponent', .5))
                        policy_batch = (replay.sample(args.batch, generators['replay'])
                                        if fraction and (defender_learner.updates+1) % defender_learner.policy_delay == 0
                                        else None)
                        update = defender_learner.update(batch, schedule['temperature'], schedule['feedback_scale'], policy_batch)
                    else:
                        update = attacker_learner.update(replay.sample(args.batch, generators['replay']))
                    metrics.update(update)
            if time.perf_counter()-last_report >= 30:
                status = dict(status='training', phase=args.phase, active_side=active_side, transitions=transitions,
                              side_transitions=counts, seconds=time.perf_counter()-started,
                              metrics={key: float(value) for key, value in metrics.items()})
                write_json(out/'status.json', status)
                print(json.dumps(status), flush=True)
                last_report = time.perf_counter()
            if transitions >= next_eval or transitions >= args.steps:
                evaluation = evaluate_pair(defender, attacker, task, args.eval_count, args.eval_seed, args.policy_eval_seed, args.device)
                expected_counts, update_counts = phase_counts(identity, transitions, defender_learner.updates, attacker_learner.updates)
                if counts != expected_counts:
                    raise ValueError('Cumulative side transition counts have diverged.')
                evaluation.update(phase=args.phase, transitions=transitions, role='validation',
                                  phase_id=identity['id'], update_counts=update_counts)
                checkpoint = dict(phase=args.phase, transitions=transitions, side_transitions=counts,
                                  defender=defender_learner.payload(), attacker=attacker_learner.payload(),
                                  defender_recipe=recipe, attacker_recipe=attacker_recipe, phase_identity=identity,
                                  update_counts=update_counts, task=asdict(task), arguments=vars(args), evaluation=evaluation,
                                  implementation_version=IMPLEMENTATION_VERSION,
                                  training_state=dict(replay=replay.state_dict(), environment=env.state_dict(),
                                                      random=random_state(args.device, generators)))
                save(out/'last.pt', checkpoint)
                if transitions == args.steps:
                    last_checkpoint = phase_checkpoint_identity(out/'last.pt', checkpoint)
                del checkpoint
                append_evaluation(out, evaluation)
                print(json.dumps({key: value for key, value in evaluation.items() if key != 'rows'}), flush=True)
                next_eval = ((transitions//args.eval_interval)+1)*args.eval_interval
        endpoint = evaluate_pair(defender, attacker, task, args.eval_count, args.test_seed, args.policy_eval_seed, args.device)
        endpoint.update(phase=args.phase, transitions=transitions, training_seed=args.seed,
                        role='fixed-stage-endpoint', research_stage=args.stage)
        if registry is not None:
            attach_identity(endpoint, chain_evaluation_identity(registry, args.phase, last_checkpoint, asdict(task), 'fixed-stage-endpoint'))
            validate_result(endpoint, endpoint['identity'])
        write_json(out/'endpoint.json', endpoint)
        if args.phase == 5:
            retention = evaluate_pair(defender, None, task, args.eval_count, args.test_seed, args.policy_eval_seed, args.device)
            retention.update(phase=5, transitions=transitions, training_seed=args.seed,
                             opponent='APF', role='retention', research_stage=args.stage)
            if registry is not None:
                attach_identity(retention, chain_evaluation_identity(registry, 5, last_checkpoint, asdict(task), 'retention'))
                validate_result(retention, retention['identity'])
            write_json(out/'apf_retention.json', retention)
        write_json(out/'status.json', dict(status='completed', phase=args.phase, active_side=active_side,
                   transitions=transitions, side_transitions=counts, phase_id=identity['id'],
                   update_counts=last_checkpoint['update_counts'], seconds=time.perf_counter()-started,
                   final={key: value for key, value in endpoint.items() if key != 'rows'}))
    except BaseException as error:
        write_json(out/'status.json', dict(status='failed', phase=args.phase, transitions=transitions,
                                         error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
