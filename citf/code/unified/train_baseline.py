"""Train an author-architecture public baseline on the unified common task."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import torch

from .baselines import PublicBaseline, BaselineLearner
from .environment import FleetEnv, TaskConfig
from .learner import Replay
from . import IMPLEMENTATION_VERSION
from .protocol import baseline_updates, bind_run, learning_counts, validate_checkpoint
from .train import (evaluate, write_json, save, seed_policy, random_state, restore_random_state,
                    resume_arguments, prepare_output, run_generators, validate_training_arguments, rollout_count)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True)
    p.add_argument('--method', choices=['arboids','rp','sac'], default='arboids')
    p.add_argument('--steps', type=int, default=300000)
    p.add_argument('--seed', type=int, default=1101)
    p.add_argument('--envs', type=int, default=64)
    p.add_argument('--batch', type=int, default=4096)
    p.add_argument('--updates-per-joint-transition', type=float, default=1.)
    p.add_argument('--target-entropy', type=float, help='Author target: ARBoids -3; RP/SAC -2.')
    p.add_argument('--gamma', type=float, default=.99)
    p.add_argument('--target-tau', type=float, default=.005)
    p.add_argument('--registry', help='Frozen common-study registry; required for formal training.')
    p.add_argument('--eval-interval', type=int, default=50000)
    p.add_argument('--eval-count', type=int, default=100)
    p.add_argument('--eval-seed', type=int, default=73001)
    p.add_argument('--policy-eval-seed', type=int, default=83001)
    p.add_argument('--stage', choices=['development', 'formal'], default='development')
    p.add_argument('--thrust-delta', type=float, default=300.)
    p.add_argument('--collision-penalty', type=float, default=50.)
    p.add_argument('--capacity', type=int, default=1000000)
    p.add_argument('--start-learning', type=int, default=50000)
    p.add_argument('--resume')
    p.add_argument('--init-from')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--width', type=int, default=512)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    payload = torch.load(args.resume, map_location='cpu', weights_only=False) if args.resume else None
    if payload is not None:
        resume_arguments(args, payload)
        validate_checkpoint(payload)
    if args.target_entropy is None:
        args.target_entropy = -3. if args.method == 'arboids' else -2.
    if not 0 < args.updates_per_joint_transition < float('inf'):
        raise ValueError('The baseline update-to-joint-transition ratio must be positive and finite.')
    validate_training_arguments(args)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    seed_policy(args.seed, args.device)
    out = Path(args.out)
    prepare_output(out, payload)
    task = TaskConfig(thrust_delta=args.thrust_delta, collision_penalty=args.collision_penalty)
    run = bind_run(args, task, payload)
    model = PublicBaseline(args.method, args.width, task.thrust_delta).to(args.device)
    learner = BaselineLearner(model, args.lr, args.gamma, args.target_tau, args.target_entropy)
    if payload is not None:
        learner.restore(payload)
    elif args.init_from:
        initialized = torch.load(args.init_from, map_location='cpu', weights_only=False)
        model.load_state_dict(initialized['model'])
        learner.target.load_state_dict(model.critic.state_dict())
        del initialized
    env = FleetEnv(args.envs, task, args.seed+100000, args.device, curriculum=True)
    dimension = model.observe(env.observation())[0].shape[-1]
    replay = Replay(args.capacity, task.defenders, args.device,
                    shapes=dict(s=(dimension,), a=(model.action_dim,), u=(2,), r=(),
                                next_s=(dimension,), done=(), event=(), seconds=()))
    generators = run_generators(args.seed, args.device)
    transitions, best, metrics, evaluation = 0, -1., {}, None
    if payload is not None:
        transitions = payload['transitions']
        saved = payload['training_state']
        replay.load_state_dict(saved['replay'])
        env.load_state_dict(saved['environment'])
        restore_random_state(saved['random'], args.device, generators)
        best, evaluation = saved['best'], payload['evaluation']
        del saved
    del payload
    next_eval = ((transitions//args.eval_interval)+1)*args.eval_interval
    started = last_report = time.perf_counter()
    write_json(out/'configuration.json',dict(arguments=vars(args),task=asdict(task),research_stage=args.stage,
                implementation_version=IMPLEMENTATION_VERSION, replay_unit='individual transition', run_identity=run,
                optimizer_updates_per_team_transition=args.updates_per_joint_transition,
               architecture_source='third_party/arboids_release/train/policy/networks.py',
                adaptations=['common task and actuator constraints','own previous actual action and remaining time in observations',
                            'fixed physical-unit observation normalization'],
                target_entropy=args.target_entropy, temperature_objective='author alpha-space'))
    try:
        while transitions < args.steps:
            count = rollout_count(args, transitions, next_eval)
            active = torch.arange(args.envs, device=args.device) < count
            with torch.no_grad():
                x = env.observation()
                result = model.select(x, deterministic=False, exploration_generator=generators['exploration'],
                                      uniform=transitions < args.start_learning)
                next_x, _, done, individual = env.step(result['action'], active_mask=active)
                next_s = model.observe(next_x)[0]
                replay.add(s=result['observation'][:count].flatten(0, 1),
                           a=result['latent'][:count].flatten(0, 1),
                           u=env.executed_defender_action()[:count].flatten(0, 1),
                           r=individual[:count].flatten(), next_s=next_s[:count].flatten(0, 1),
                           **{key: value[:count, None].expand(-1, task.defenders).flatten()
                              for key, value in dict(done=done, event=env.outcome, seconds=env.elapsed).items()})
                transitions += count
                env.reset((done & active).nonzero().flatten(), transitions)
            due = baseline_updates(transitions, args.start_learning, args.updates_per_joint_transition)-learner.updates
            if due < 0:
                raise ValueError('Checkpoint learning count exceeds its registered transition budget.')
            for _ in range(due):
                batch = replay.sample(args.batch, generators['replay'])
                metrics = learner.update(*(batch[key] for key in ('s','a','r','next_s','done')))
            if time.perf_counter()-last_report > 30:
                record = dict(status='training',transitions=transitions,updates=learner.updates,
                              seconds=time.perf_counter()-started,metrics=metrics)
                write_json(out/'status.json',record)
                print(json.dumps(record),flush=True)
                last_report = time.perf_counter()
            if transitions >= next_eval or transitions >= args.steps or transitions == 500000:
                evaluation = evaluate(model,task,args.eval_count,args.eval_seed,args.policy_eval_seed,args.device)
                counts = learning_counts(args.method, transitions, vars(args), learner.updates)
                evaluation.update(transitions=transitions, updates=learner.updates, update_counts=counts,
                                  run_id=run['id'], role='validation', research_stage=args.stage)
                improved = evaluation['success_rate'] > best
                best = max(best, evaluation['success_rate'])
                checkpoint = dict(**learner.payload(), method=args.method, width=args.width,
                                  task=asdict(task), transitions=transitions, arguments=vars(args),
                                  evaluation=evaluation, implementation_version=IMPLEMENTATION_VERSION,
                                  run_identity=run, update_counts=counts)
                if improved:
                    save(out/'best_development.pt', checkpoint)
                checkpoint['training_state'] = dict(replay=replay.state_dict(), environment=env.state_dict(),
                                                     random=random_state(args.device, generators), best=best)
                save(out/'last.pt', checkpoint)
                del checkpoint
                with (out/'evaluations.jsonl').open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(evaluation)+'\n')
                print(json.dumps(dict(event='development_evaluation',**{k:v for k,v in evaluation.items() if k != 'rows'})),flush=True)
                next_eval = ((transitions//args.eval_interval)+1)*args.eval_interval
        write_json(out/'status.json',dict(status='completed',transitions=transitions,updates=learner.updates,
                   run_id=run['id'], update_counts=learning_counts(args.method, transitions, vars(args), learner.updates),
                   seconds=time.perf_counter()-started,best_development_success=best,
                   final={k:v for k,v in evaluation.items() if k != 'rows'}))
    except BaseException as error:
        write_json(out/'status.json',dict(status='failed',transitions=transitions,updates=learner.updates,error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
