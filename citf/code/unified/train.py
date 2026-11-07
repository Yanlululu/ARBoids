"""Train the unified CITF controller and evaluate full development episodes."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time
import torch

from .environment import TaskConfig, FleetEnv, boids, execute, fusion
from .networks import CITF, ModelConfig
from .learner import Learner, Replay
from . import IMPLEMENTATION_VERSION
from .protocol import (bind_run, learning_counts, outcome_metrics, validate_metrics,
                       validate_checkpoint, digest)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def save(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.pending')
    torch.save(payload, temporary)
    temporary.replace(path)


def device_index(device):
    device = torch.device(device)
    if device.type != 'cuda':
        return None
    torch.cuda.init()
    return torch.cuda.current_device() if device.index is None else device.index


def seed_policy(seed, device):
    # Seed only the devices whose RNG states we preserve; manual_seed_all would
    # also change unrelated GPUs outside evaluate()'s fork_rng context.
    torch.random.default_generator.manual_seed(seed)
    index = device_index(device)
    if index is not None:
        torch.cuda.default_generators[index].manual_seed(seed)


def random_state(device, generators):
    index = device_index(device)
    return dict(cpu=torch.get_rng_state(),
                cuda=None if index is None else torch.cuda.get_rng_state(index),
                streams={key: value.get_state() for key, value in generators.items()})


def restore_random_state(saved, device, generators):
    torch.set_rng_state(saved['cpu'].cpu())
    index = device_index(device)
    if (saved['cuda'] is None) != (index is None):
        raise ValueError('Exact resume requires the same CPU/CUDA execution kind; use --init-from for a new run.')
    if index is not None:
        torch.cuda.set_rng_state(saved['cuda'].cpu(), index)
    for key, generator in generators.items():
        generator.set_state(saved['streams'][key].cpu())


def resume_arguments(args, payload):
    if payload.get('implementation_version') != IMPLEMENTATION_VERSION or 'training_state' not in payload:
        raise ValueError('This checkpoint has no compatible complete training state. Use --init-from for weights-only initialization in a new output directory.')
    explicit = {value.split('=', 1)[0] for value in sys.argv[1:] if value.startswith('--')}
    for key, saved in payload['arguments'].items():
        if key in ('out', 'steps', 'device', 'resume', 'init_from', 'registry') or not hasattr(args, key):
            continue
        if '--'+key.replace('_', '-') in explicit and getattr(args, key) != saved:
            raise ValueError(f'Exact resume cannot change --{key.replace("_", "-")}; use --init-from for a new experiment.')
        setattr(args, key, saved)
    if args.steps < payload['transitions']:
        raise ValueError('--steps is the total run budget and is smaller than the saved transition count.')


def prepare_output(out, resume=None):
    out.mkdir(parents=True, exist_ok=True)
    occupied = any((out/name).exists() for name in ('configuration.json', 'last.pt', 'evaluations.jsonl'))
    if occupied and resume is None:
        raise FileExistsError(f'{out} already contains a run; choose a new run directory or use --resume.')
    if resume is not None and (out/'configuration.json').exists():
        configuration = json.loads((out/'configuration.json').read_text(encoding='utf-8'))
        ignored = {'out', 'steps', 'device', 'resume', 'init_from', 'registry'}
        prior = {key: value for key, value in configuration['arguments'].items() if key not in ignored}
        restored = {key: value for key, value in resume['arguments'].items() if key not in ignored}
        if prior != restored or configuration.get('implementation_version') != IMPLEMENTATION_VERSION:
            raise ValueError('The output directory belongs to a different training configuration.')
    if resume is not None and (out/'evaluations.jsonl').exists():
        nodes = []
        with (out/'evaluations.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                nodes.append(row['transitions'])
                validate_metrics(row)
                if row['transitions'] > resume['transitions']:
                    raise ValueError('Output already contains evaluations newer than this checkpoint; resume into a new directory.')
        if nodes != sorted(set(nodes)):
            raise ValueError('The learning curve contains duplicate or out-of-order evaluations.')
        # A crash can occur between the atomic checkpoint save and JSONL append.
        if resume['evaluation'] is not None and resume['transitions'] not in nodes:
            with (out/'evaluations.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(resume['evaluation'])+'\n')


def run_generators(seed, device):
    return {key: torch.Generator(device=device).manual_seed(seed+offset)
            for key, offset in (('exploration', 200000), ('replay', 300000))}


def validate_training_arguments(args):
    for key in ('steps', 'envs', 'batch', 'updates_per_step', 'capacity', 'start_learning',
                'eval_interval', 'eval_count'):
        if hasattr(args, key) and getattr(args, key) < 1:
            raise ValueError(f'--{key.replace("_", "-")} must be positive.')
    if args.lr <= 0 or args.capacity < args.start_learning:
        raise ValueError('Positive learning rate and replay capacity at least start-learning are required.')
    if not 0 <= getattr(args, 'value_terminal_fraction', 0.) < 1:
        raise ValueError('Value replay must retain a positive uniform sampling component.')
    if not 0 <= getattr(args, 'value_importance_exponent', .5) <= 1:
        raise ValueError('The value importance exponent must lie between zero and one.')
    if args.resume and args.init_from:
        raise ValueError('Choose exact resume or weights-only initialization, not both.')


def rollout_count(args, transitions, next_eval):
    # Stop exactly at evaluations, the reusable 500k defender stage and the
    # requested budget, even when those counts are not divisible by envs.
    boundary = min(args.steps, next_eval)
    if transitions < 500000:
        boundary = min(boundary, 500000)
    return min(args.envs, boundary-transitions)


def policy_learning_rate(recipe, transitions):
    end = recipe.get('policy_lr_end')
    if end is None:
        return recipe['lr']
    fraction = min(1., max(0., (transitions-recipe['policy_lr_decay_start'])/
                           recipe['policy_lr_decay_steps']))
    return recipe['lr']+(end-recipe['lr'])*fraction


def policy_schedule(recipe, transitions):
    """All policy schedules use cumulative defender transitions, never SGD calls."""
    noise_fraction = min(1., transitions/recipe['exploration_steps'])
    temperature_fraction = min(1., transitions/recipe['temperature_steps'])
    return dict(probability=max(.1, 1-.9*noise_fraction),
                endpoint_noise=recipe['endpoint_noise_start']+(recipe['endpoint_noise_end']-recipe['endpoint_noise_start'])*noise_fraction,
                temperature=recipe['temperature_start']+(recipe['temperature_end']-recipe['temperature_start'])*temperature_fraction,
                feedback_scale=(0. if transitions < recipe['feedback_start'] else
                                min(1., (transitions-recipe['feedback_start'])/recipe['feedback_ramp'])
                                if recipe['feedback_ramp'] else 1.),
                policy_lr=policy_learning_rate(recipe, transitions))


def evaluate(model, cfg, count=100, scene_seed=73001, policy_seed=83001, device='cuda'):
    if count < 1:
        raise ValueError('Evaluation must contain at least one scene.')
    model_delta = model.cfg.thrust_delta if isinstance(model, CITF) else getattr(model, 'delta', cfg.thrust_delta)
    if model_delta != cfg.thrust_delta:
        raise ValueError('Policy scoring and environment execution must use the same thrust increment constraint.')
    index = device_index(device)
    cuda_devices = [] if index is None else [index]
    started = time.perf_counter()
    with torch.random.fork_rng(devices=cuda_devices):
        seed_policy(policy_seed, device)
        env = FleetEnv(count, cfg, seed=scene_seed, device=device)
        while not bool((env.outcome != 0).all()):
            with torch.no_grad():
                x = env.observation()
                action = execute(boids(x), x[:, :-1, 6:8], cfg.thrust_delta) if model is None else model.select(x)['action']
                env.step(action)
        rows = env.records()
    return dict(scene_seed=scene_seed, policy_seed=policy_seed,
                task=asdict(cfg), implementation_version=IMPLEMENTATION_VERSION,
                **outcome_metrics(rows, count, scene_seed, cfg.duration),
                wall_seconds=time.perf_counter()-started, rows=rows)


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--seed', type=int, default=1101)
    p.add_argument('--envs', type=int, default=64)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--value-terminal-fraction', type=float, default=.5)
    p.add_argument('--value-importance-exponent', type=float, default=.5)
    p.add_argument('--updates-per-step', type=int, default=1)
    p.add_argument('--capacity', type=int, default=1000000)
    p.add_argument('--start-learning', type=int, default=5000)
    p.add_argument('--eval-interval', type=int, default=20000)
    p.add_argument('--eval-count', type=int, default=100)
    p.add_argument('--eval-seed', type=int, default=73001)
    p.add_argument('--policy-eval-seed', type=int, default=83001)
    p.add_argument('--stage', choices=['development', 'formal'], default='development')
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--neighbors', type=int, default=2)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--search-steps', type=int, default=2)
    p.add_argument('--search-rate', type=float, default=.1)
    p.add_argument('--thrust-delta', type=float, default=300.)
    p.add_argument('--collision-penalty', type=float, default=50.)
    p.add_argument('--feedback', type=float, default=1.)
    p.add_argument('--feedback-start', type=int, default=0)
    p.add_argument('--feedback-ramp', type=int, default=0)
    p.add_argument('--no-action-interaction', action='store_true')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--policy-lr-end', type=float)
    p.add_argument('--policy-lr-decay-start', type=int, default=500000)
    p.add_argument('--policy-lr-decay-steps', type=int, default=500000)
    p.add_argument('--target-tau', type=float, default=.005)
    p.add_argument('--temperature-start', type=float, default=10.)
    p.add_argument('--temperature-end', type=float, default=1.)
    p.add_argument('--endpoint-noise-start', type=float, default=0.)
    p.add_argument('--endpoint-noise-end', type=float, default=0.)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume')
    p.add_argument('--registry', help='Frozen common-study registry; required for formal training.')
    p.add_argument('--init-from')
    p.add_argument('--exploration-steps', type=int, default=100000)
    p.add_argument('--temperature-steps', type=int, default=800000)
    return p.parse_args()


def main():
    args = arguments()
    payload = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if payload is not None:
        resume_arguments(args, payload)
        validate_checkpoint(payload)
    validate_training_arguments(args)
    if min(args.exploration_steps, args.temperature_steps) <= 0:
        raise ValueError('Exploration and temperature schedules must have positive lengths.')
    if min(args.feedback_start, args.feedback_ramp) < 0:
        raise ValueError('Feedback start and ramp lengths must be nonnegative.')
    if (args.policy_lr_decay_start < 0 or args.policy_lr_decay_steps <= 0
            or args.policy_lr_end is not None and
            (not math.isfinite(args.policy_lr_end) or not 0 < args.policy_lr_end <= args.lr)):
        raise ValueError('The policy learning-rate decay has a positive duration and a positive endpoint at most --lr.')
    if (any(not math.isfinite(value) or value <= 0 for value in (args.temperature_start, args.temperature_end))
            or any(not math.isfinite(value) or value < 0 for value in (args.endpoint_noise_start, args.endpoint_noise_end))):
        raise ValueError('Temperatures must be positive and endpoint noise scales nonnegative and finite.')
    out = Path(args.out)
    prepare_output(out, payload)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    seed_policy(args.seed, args.device)
    cfg = ModelConfig(width=args.width, layers=args.layers, samples=args.samples, neighbors=args.neighbors,
                      search_steps=args.search_steps, search_rate=args.search_rate,
                      thrust_delta=args.thrust_delta, feedback=args.feedback,
                      action_interaction=not args.no_action_interaction)
    task = TaskConfig(thrust_delta=args.thrust_delta, collision_penalty=args.collision_penalty)
    run = bind_run(args, task, payload)
    model = CITF(cfg).to(args.device)
    learner = Learner(model, lr=args.lr, tau=args.target_tau)
    transitions = 0
    if payload is not None:
        learner.restore(payload)
        transitions = payload['transitions']
    elif args.init_from:
        initialized = torch.load(args.init_from, map_location='cpu', weights_only=False)
        model.load_state_dict(initialized['model'])
        learner.target.load_state_dict(model.state_dict())
        del initialized
    replay = Replay(args.capacity, task.defenders, args.device)
    env = FleetEnv(args.envs, task, seed=args.seed+100000, device=args.device, curriculum=True)
    generators = run_generators(args.seed, args.device)
    best, metrics, outcomes, episode_count = -1., {}, torch.zeros(5, device=args.device), 0
    evaluation = None
    if payload is not None:
        saved = payload['training_state']
        replay.load_state_dict(saved['replay'])
        env.load_state_dict(saved['environment'])
        restore_random_state(saved['random'], args.device, generators)
        best, episode_count = saved['best'], saved['episodes']
        outcomes.copy_(saved['outcomes'])
        evaluation = payload['evaluation']
        del saved
    del payload
    write_json(out/'configuration.json', dict(arguments=vars(args), model=asdict(cfg), task=asdict(task),
               implementation_version=IMPLEMENTATION_VERSION, research_stage=args.stage, run_identity=run,
               reward='unified event/formation structure; collision magnitude recorded in task',
               random_streams=dict(policy=args.seed, environment=args.seed+100000,
                                   exploration=args.seed+200000, replay=args.seed+300000,
                                   evaluation_policy=args.policy_eval_seed),
               dynamics=dict(M=[380,400,1430], damping_linear=[100,100,980],
                             damping_quadratic=[150,150,950], half_thruster_spacing=1.25),
               parameters={name: sum(p.numel() for p in module.parameters()) for name,module in
                           [('endpoint', model.endpoint), ('generator', model.generator), ('q1', model.q1), ('q2', model.q2)]}))
    if args.resume and (out/'boids_development.json').exists():
        reference = json.loads((out/'boids_development.json').read_text(encoding='utf-8'))
        if (reference.get('task') != asdict(task) or reference.get('scene_seed') != args.eval_seed
                or reference['tasks'] != args.eval_count
                or reference.get('implementation_version') != IMPLEMENTATION_VERSION):
            raise ValueError('Cached Boids reference belongs to a different task, scene set or implementation.')
    else:
        reference = evaluate(None, task, args.eval_count, args.eval_seed, args.policy_eval_seed, args.device)
        write_json(out/'boids_development.json', reference)
    print(json.dumps(dict(event='boids_reference', **{k:v for k,v in reference.items() if k != 'rows'})), flush=True)
    started, last_report = time.perf_counter(), time.perf_counter()
    start_transitions = transitions
    next_eval = ((transitions//args.eval_interval)+1)*args.eval_interval
    try:
        while transitions < args.steps:
            count = rollout_count(args, transitions, next_eval)
            active = torch.arange(args.envs, device=args.device) < count
            x = env.observation()
            schedule = policy_schedule(vars(args), transitions)
            probability, endpoint_noise = schedule['probability'], schedule['endpoint_noise']
            with torch.no_grad():
                if transitions < args.start_learning:
                    rule_source = boids(x)
                    b = execute(rule_source, x[:, :-1, 6:8], task.thrust_delta)
                    a = model.endpoint(x, rule_source)
                    if endpoint_noise > 0:
                        a = execute(a+endpoint_noise*torch.randn(a.shape, device=a.device,
                                    generator=generators['exploration']), x[:, :-1, 6:8], task.thrust_delta)
                    g = torch.rand(args.envs, task.defenders, device=args.device, generator=generators['exploration'])
                    u = fusion(x, a, b, g, task.thrust_delta)
                else:
                    result = model.select(x, endpoint_noise, generators['exploration'])
                    random_g = torch.rand(result['g'].shape, device=args.device, generator=generators['exploration'])
                    exploratory = torch.rand(args.envs, 1, device=args.device, generator=generators['exploration']) < probability
                    g = torch.where(exploratory, random_g, result['g'])
                    u = fusion(x, result['a'], result['b'], g, task.thrust_delta)
                next_x, reward, done, individual_reward = env.step(u, active_mask=active)
                # next_x is recorded before resetting any terminal lane.
                replay.add(x=x[:count], u=env.executed_defender_action()[:count],
                           r=reward[:count], ri=individual_reward[:count], next_x=next_x[:count], done=done[:count],
                           **{key: value[:count] for key, value in env.replay_metadata().items()})
                ended = done & active
                outcomes += torch.bincount(env.outcome[ended], minlength=5)
                episode_count += int(ended.sum())
                transitions += count
                env.reset(ended.nonzero().flatten(), transitions)
            if replay.size >= args.start_learning:
                schedule = policy_schedule(vars(args), transitions)
                policy_lr = schedule['policy_lr']
                learner.set_policy_learning_rate(policy_lr)
                metrics['policy_lr'] = policy_lr
                temperature, feedback_scale = schedule['temperature'], schedule['feedback_scale']
                for _ in range(args.updates_per_step):
                    value_batch = replay.sample_value(args.batch, generators['replay'],
                                                      args.value_terminal_fraction, args.value_importance_exponent)
                    policy_batch = (replay.sample(args.batch, generators['replay'])
                                    if args.value_terminal_fraction and
                                    (learner.updates+1) % learner.policy_delay == 0 else None)
                    metrics.update(learner.update(value_batch, temperature, feedback_scale, policy_batch))
            now = time.perf_counter()
            if now-last_report >= 30:
                record = dict(status='training', transitions=transitions, updates=learner.updates,
                              episodes=episode_count, episodes_by_outcome=outcomes.cpu().tolist(),
                              seconds=now-started, transitions_per_second=(transitions-start_transitions)/(now-started),
                              metrics={k: float(v) for k,v in metrics.items()})
                write_json(out/'status.json', record)
                print(json.dumps(record), flush=True)
                last_report = now
            if transitions >= next_eval or transitions >= args.steps or transitions == 500000:
                evaluation = evaluate(model, task, args.eval_count, args.eval_seed, args.policy_eval_seed, args.device)
                counts = learning_counts(run['method'], transitions, vars(args), learner.updates)
                evaluation.update(transitions=transitions, updates=learner.updates, update_counts=counts,
                                  run_id=run['id'], role='validation', research_stage=args.stage)
                improved = evaluation['success_rate'] > best
                best = max(best, evaluation['success_rate'])
                checkpoint = dict(**learner.payload(), transitions=transitions, task=asdict(task),
                                  arguments=vars(args), evaluation=evaluation,
                                  implementation_version=IMPLEMENTATION_VERSION, run_identity=run, update_counts=counts)
                if improved:
                    save(out/'best_development.pt', checkpoint)
                if transitions == 500000:
                    save(out/'defender_stage1.pt', checkpoint)
                checkpoint['training_state'] = dict(replay=replay.state_dict(), environment=env.state_dict(),
                                                     random=random_state(args.device, generators), best=best,
                                                     outcomes=outcomes.cpu(), episodes=episode_count)
                save(out/'last.pt', checkpoint)
                del checkpoint
                with (out/'evaluations.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(evaluation)+'\n')
                print(json.dumps(dict(event='development_evaluation', **{k:v for k,v in evaluation.items() if k != 'rows'})), flush=True)
                next_eval = ((transitions//args.eval_interval)+1)*args.eval_interval
        final = dict(status='completed', transitions=transitions, updates=learner.updates,
                     run_id=run['id'], update_counts=learning_counts(run['method'], transitions, vars(args), learner.updates),
                     best_development_success=best, seconds=time.perf_counter()-started,
                     final={k:v for k,v in evaluation.items() if k != 'rows'})
        write_json(out/'status.json', final)
        print(json.dumps(final), flush=True)
    except BaseException as error:
        write_json(out/'status.json', dict(status='failed', transitions=transitions, updates=learner.updates,
                                         error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
