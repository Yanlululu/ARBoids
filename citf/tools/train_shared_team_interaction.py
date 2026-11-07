"""Finite matched online continuation for a shared adverse team interaction.

Every training and deployment team uses the same interaction aggregation.
All ordinary task rewards, candidate generation, search and feedback remain.
Source weights, targets, Adam states, replay, environment and RNG are restored.
"""
import argparse
from dataclasses import asdict, replace
import json
import inspect
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, os.environ.get('CITF_CODE_ROOT', str(Path(__file__).resolve().parents[1]/'code')))
from unified.environment import FleetEnv, TaskConfig, fusion
from unified.learner import Learner, Replay
from unified.networks import (CITF, ModelConfig, Endpoint, JointGenerator, TeamCritic,
                              controlled_pair_effect, nearest_other)
from unified.train import (evaluate, policy_schedule, random_state, restore_random_state,
                           run_generators, save, seed_policy, write_json, validate_training_arguments)
from unified.protocol import validate_checkpoint, validate_optimizer_counts

CONDITIONS = ((3, 2., 73001), (7, 2., 73005), (3, 3., 73004))


class SharedTeamCritic(TeamCritic):
    def from_encoded(self, encoded, u, individual=False):
        z, pair, heading, rule = encoded
        multiple = u.ndim == 4
        if multiple:
            batch, count, n = u.shape[:3]
            z = z[:, None].expand(-1, count, -1, -1).reshape(batch*count, n+1, -1)
            pair = pair[:, None].expand(-1, count, -1, -1, -1).reshape(batch*count, n+1, n+1, 8)
            heading = heading[:, None].expand(-1, count, -1).reshape(batch*count, n+1)
            rule = rule[:, None].expand(-1, count, -1, -1).reshape(batch*count, n, 2)
            u = u.reshape(batch*count, n, 2)
        n = u.shape[1]
        context = self.value(z[:, :n]).squeeze(-1)
        command = torch.cat((u, torch.zeros_like(u[:, :1])), 1)
        action_state = z+self.action(command/1000)
        h = action_state
        for layer in self.layers:
            h = layer(h)
        q = context+self.readout(h[:, :n]).squeeze(-1)
        if self.action_interaction:
            edge = torch.cat((pair, controlled_pair_effect(heading, command)), -1)
            h = action_state
            for layer in self.interaction_layers:
                h = layer(h, edge)
            pair_response = self.interaction_pair(edge[:, :n, :n])
            receiver = h[:, :n, None].expand(-1, -1, n, -1)
            response = self.interaction_readout(torch.cat((receiver, pair_response), -1))
            departure = (u-rule)/1000.
            pair_value = (response*departure[:, None]).sum(-1)
            diagonal = torch.eye(n, dtype=torch.bool, device=u.device)[None]
            allowed = (nearest_other(edge[:, :n, :n], self.neighbors)
                       if n > self.neighbors+1 else ~diagonal)
            permitted = pair_value.masked_fill(~allowed, 0.)
            negative = torch.where(permitted < 0., permitted, torch.zeros_like(permitted))
            positive_local = (permitted-negative).sum(-1)/min(self.neighbors, n-1)
            # An adverse pair can end the entire team's future. Learn this
            # common influence already with three vessels, then use the same
            # operator for larger teams. There is no N>3-only correction.
            adverse_team = negative.flatten(1).amin(-1, keepdim=True)
            q = q+positive_local+adverse_team
        if individual:
            return q.reshape(batch, count, n) if multiple else q
        q = q.mean(-1)
        return q.reshape(batch, count) if multiple else q


class SharedTeamCITF(CITF):
    def __init__(self, cfg=None):
        nn.Module.__init__(self)
        self.cfg = cfg or ModelConfig()
        self.endpoint = Endpoint(self.cfg)
        self.generator = JointGenerator(self.cfg)
        self.q1, self.q2 = SharedTeamCritic(self.cfg, 1), SharedTeamCritic(self.cfg, 2)


def persist_evaluations(out, checkpoint):
    """Publish only evaluations belonging to an already saved complete state."""
    path = out/'evaluations.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []
    def key(row):
        return row['transitions'], row['task']['defenders'], row['task']['agility']
    existing = {key(row): row for row in rows}
    if (len(existing) != len(rows) or any(row['transitions'] > checkpoint['transitions']
            or row.get('continuation') != checkpoint['continuation'] for row in rows)):
        raise ValueError('Continuation output contains duplicate, newer, or unrelated evaluations.')
    for row in checkpoint.get('evaluations', [checkpoint['evaluation']]):
        if key(row) in existing and existing[key(row)] != row:
            raise ValueError('Saved evaluation differs from the complete checkpoint.')
        if key(row) not in existing:
            rows.append(row)
            existing[key(row)] = row
        target = out/f"evaluation_{row['transitions']}_team{row['task']['defenders']}_agility{row['task']['agility']:.1f}.json"
        if target.exists() and json.loads(target.read_text(encoding='utf-8')) != row:
            raise ValueError('A condition artifact belongs to a different continuation.')
        write_json(target, row)
    temporary = path.with_suffix('.pending')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in rows), encoding='utf-8')
    temporary.replace(path)


def continuation_counts(continuation, recipe, transitions, updates, starting_counts=None):
    position, added = continuation['source_transitions'], 0
    while position < transitions:
        node = (continuation['source_transitions']+
                ((position-continuation['source_transitions'])//continuation['evaluation_interval']+1)*continuation['evaluation_interval'])
        boundary = min(transitions, node, 500000 if position < 500000 else transitions)
        position += min(recipe['envs'], boundary-position)
        if position >= recipe['start_learning']:
            added += recipe['updates_per_step']
    initial = starting_counts if starting_counts is not None else continuation['starting_update_counts']
    if updates != initial['critic']+added:
        raise ValueError('Online continuation lost or added critic updates relative to its actual collection clock.')
    delayed = updates//2-initial['critic']//2
    return dict(critic=updates, **{key: initial[key]+delayed for key in ('endpoint', 'generator', 'target')})


def main(candidate_type=None, candidate_name='shared_team', candidate_definition=None, learner_type=None):
    candidate_type = candidate_type or SharedTeamCITF
    learner_type = learner_type or Learner
    release_root = os.environ.get('CITF_RELEASE_ROOT')
    if release_root and any(not Path(inspect.getfile(cls)).resolve().is_relative_to(Path(release_root).resolve())
                            for cls in (candidate_type, learner_type)):
        raise ValueError('Freeze the candidate and its paired core/tools together before launching it from the active release.')
    candidate_definition = candidate_definition or 'only action-interaction aggregation changes in shared_team'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--variant', choices=('ordinary', candidate_name), required=True)
    parser.add_argument('--steps', type=int, default=50000, help='Additional actual team transitions.')
    parser.add_argument('--eval-interval', type=int, default=25000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-at', type=int, help='Pause at a scheduled cumulative evaluation node; retain the final budget.')
    args = parser.parse_args()
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    recipe = source['arguments']
    previous = source.get('continuation')
    validate_training_arguments(SimpleNamespace(**dict(recipe, steps=args.steps, eval_interval=args.eval_interval,
        eval_count=100, resume=None, init_from=None, fork_from=None, stop_at=None)))
    if 'training_state' not in source or recipe['stage'] != 'development':
        raise ValueError('Online candidates require the complete development training state.')
    if source['training_state']['replay']['size'] < recipe['start_learning']:
        raise ValueError('This candidate comparison starts from a prefix with ready replay; use main training for a new structure from scratch.')
    if args.resume:
        if previous is None or previous['variant'] != args.variant:
            raise ValueError('Resume requires the same recorded continuation variant.')
        if source.get('controller_variant', previous['variant']) != args.variant:
            raise ValueError('The saved controller and continuation variant disagree.')
        continuation = previous
        explicit = {value.split('=', 1)[0] for value in sys.argv[1:] if value.startswith('--')}
        if ('--steps' in explicit and args.steps != continuation['final_transitions']-continuation['source_transitions']
                or '--eval-interval' in explicit and args.eval_interval != continuation['evaluation_interval']):
            raise ValueError('Resume retains the original additional budget and evaluation interval.')
    else:
        if previous is not None:
            raise ValueError('Use --resume for an existing candidate; new matched candidates start from a main-training prefix.')
        validate_checkpoint(source, current_source=False)
        continuation = dict(variant=args.variant, source=str(args.checkpoint),
                            source_implementation=source['implementation_version'],
                            source_transitions=source['transitions'],
                            final_transitions=source['transitions']+args.steps,
                            evaluation_interval=args.eval_interval,
                            starting_update_counts=source['update_counts'].copy(),
                            implementation=candidate_name+'-online-v1')
    starting_counts = source.get('starting_update_counts', continuation.get('starting_update_counts'))
    if starting_counts is None:
        ancestor = torch.load(continuation['source'], map_location='cpu', weights_only=False, mmap=True)
        if ancestor['transitions'] != continuation['source_transitions']:
            raise ValueError('The original candidate prefix was replaced; retain the current checkpoint and recover its original source.')
        starting_counts = ancestor['update_counts'].copy()
        del ancestor
    endpoint = continuation['final_transitions'] if args.stop_at is None else args.stop_at
    if (not source['transitions'] <= endpoint <= continuation['final_transitions']
            or endpoint != continuation['final_transitions'] and
            (endpoint-continuation['source_transitions']) % continuation['evaluation_interval']):
        raise ValueError('--stop-at must be a scheduled node between this checkpoint and the original endpoint.')
    if not args.resume and endpoint == source['transitions']:
        raise ValueError('The online candidate must collect a positive additional interval.')
    if args.resume and 'evaluations' not in source:
        # Older candidates published all conditions separately, but retained
        # only the main condition inside last.pt. Recover the recorded set.
        source['evaluations'] = []
        for n, agility, _ in CONDITIONS:
            path = args.checkpoint.parent/f"evaluation_{source['transitions']}_team{n}_agility{agility:.1f}.json"
            if not path.exists():
                raise ValueError(f'Legacy continuation requires its saved condition artifact: {path}')
            row = json.loads(path.read_text(encoding='utf-8'))
            if (row['transitions'] != source['transitions'] or row['continuation'] != continuation
                    or row['task']['defenders'] != n or row['task']['agility'] != agility):
                raise ValueError('A legacy condition artifact does not match the saved continuation.')
            source['evaluations'].append(row)
        if source['evaluations'][0] != source['evaluation']:
            raise ValueError('Legacy main evaluation differs from its complete checkpoint.')
    fresh_output = not (args.out/'configuration.json').exists()
    if not args.resume:
        args.out.mkdir(parents=True, exist_ok=False)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        if not fresh_output:
            configuration = json.loads((args.out/'configuration.json').read_text(encoding='utf-8'))
            if (configuration['continuation'] != continuation or configuration['training_recipe'] != recipe
                    or configuration['task'] != source['task'] or configuration['model'] != source['config']):
                raise ValueError('Resume output belongs to a different candidate, source or training recipe.')
        elif any(args.out.iterdir()):
            raise ValueError('Resume into an empty new directory or the matching existing output.')
        persist_evaluations(args.out, source)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    seed_policy(recipe['seed'], args.device)
    model_type = CITF if args.variant == 'ordinary' else candidate_type
    model = model_type(ModelConfig(**source['config'])).to(args.device)
    selected_learner = Learner if args.variant == 'ordinary' else learner_type
    learner = selected_learner(model, lr=recipe['lr'], **source['learner_recipe'])
    learner.restore(source)
    task = TaskConfig(**source['task'])
    replay = Replay(recipe['capacity'], task.defenders, args.device)
    replay.load_state_dict(source['training_state']['replay'])
    env = FleetEnv(recipe['envs'], task, seed=recipe['seed']+100000, device=args.device, curriculum=True)
    env.load_state_dict(source['training_state']['environment'])
    generators = run_generators(recipe['seed'], args.device)
    restore_random_state(source['training_state']['random'], args.device, generators)
    transitions = source['transitions']
    outcomes = source['training_state']['outcomes'].to(args.device)
    episodes = source['training_state']['episodes']
    best = source['training_state']['best'] if args.resume else -1.
    source_identity = source.get('parent_run_identity', source.get('run_identity'))
    source_evaluation = source['evaluation']
    evaluations = source.get('evaluations', [source_evaluation]) if args.resume else []
    del source
    conditions = CONDITIONS
    if not args.resume or fresh_output:
        write_json(args.out/'configuration.json', dict(
            research_stage='development', continuation=continuation, seed=recipe['seed'],
            resumed_checkpoint=str(args.checkpoint) if args.resume else None,
            model=asdict(model.cfg), task=asdict(task), training_recipe=recipe,
            migration='all parameter tensors, Adam moments, targets, replay, environment and random streams retained; '+
                      ('ordinary controller and learner retained' if args.variant == 'ordinary' else candidate_definition),
            decision='compare both seeds at 175k and 200k against ordinary continuation; seek a repeated meaningful reduction of dense-team collisions while retaining main defense and agility-3 interception; use the complete two-node pattern, not a one-scene gain',
            source_evaluation={k:v for k,v in source_evaluation.items() if k != 'rows'},
            conditions=[dict(defenders=n, agility=a, scene=s) for n,a,s in conditions],
            episodes_per_condition=100, policy_seed=83001,
            new_training_team_sizes=[3], additional_formal_experiments=[]))
    next_eval = min(continuation['final_transitions'],
                    continuation['source_transitions']+
                    ((transitions-continuation['source_transitions'])//continuation['evaluation_interval']+1)*
                    continuation['evaluation_interval'])
    started = last_report = time.perf_counter()
    started_at = transitions
    metrics = {}
    try:
        while transitions < endpoint:
            count = min(recipe['envs'], next_eval-transitions, endpoint-transitions)
            if transitions < 500000:
                count = min(count, 500000-transitions)
            active = torch.arange(recipe['envs'], device=args.device) < count
            x = env.observation()
            schedule = policy_schedule(recipe, transitions)
            with torch.no_grad():
                result = model.select(x, schedule['endpoint_noise'], generators['exploration'])
                random_g = torch.rand(result['g'].shape, device=args.device, generator=generators['exploration'])
                exploratory = torch.rand(recipe['envs'], 1, device=args.device,
                                          generator=generators['exploration']) < schedule['probability']
                g = torch.where(exploratory, random_g, result['g'])
                u = fusion(x, result['a'], result['b'], g, task.thrust_delta)
                next_x, reward, done, individual_reward = env.step(u, active_mask=active)
                replay.add(x=x[:count], u=env.executed_defender_action()[:count], r=reward[:count],
                           ri=individual_reward[:count], next_x=next_x[:count], done=done[:count],
                           **{key:value[:count] for key,value in env.replay_metadata().items()})
                ended = done & active
                outcomes += torch.bincount(env.outcome[ended], minlength=5)
                episodes += int(ended.sum())
                transitions += count
                env.reset(ended.nonzero().flatten(), transitions)
            schedule = policy_schedule(recipe, transitions)
            learner.set_policy_learning_rate(schedule['policy_lr'])
            for _ in range(recipe['updates_per_step']):
                value_batch = replay.sample_value(recipe['batch'], generators['replay'],
                                                  recipe['value_terminal_fraction'], recipe['value_importance_exponent'])
                policy_batch = (replay.sample(recipe['batch'], generators['replay'])
                                if recipe['value_terminal_fraction'] and
                                (learner.updates+1) % learner.policy_delay == 0 else None)
                metrics.update(learner.update(value_batch, schedule['temperature'], schedule['feedback_scale'],
                                               policy_batch, robust_value=schedule['robust_value']))
            now = time.perf_counter()
            if now-last_report >= 30:
                record = dict(status='training', variant=args.variant, transitions=transitions,
                              added_transitions=transitions-continuation['source_transitions'],
                              updates=learner.updates, episodes=episodes,
                              transitions_per_second=(transitions-started_at)/(now-started),
                              metrics={k:float(v) for k,v in metrics.items()})
                write_json(args.out/'status.json', record)
                print(json.dumps(record), flush=True)
                last_report = now
            if transitions == next_eval:
                evaluations = []
                for n, agility, scene in conditions:
                    result = evaluate(model, replace(task, defenders=n, agility=agility),
                                      100, scene, 83001, args.device)
                    result.update(variant=args.variant, transitions=transitions, seed=recipe['seed'],
                                  research_stage='development', continuation=continuation,
                                  implementation_version=continuation['implementation'])
                    evaluations.append(result)
                    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)
                best = max(best, evaluations[0]['success_rate'])
                counts = continuation_counts(continuation, recipe, transitions, learner.updates, starting_counts)
                checkpoint = dict(**learner.payload(), arguments=recipe, transitions=transitions,
                                  task=asdict(task), parent_run_identity=source_identity, continuation=continuation,
                                  controller_variant=args.variant, update_counts=counts,
                                  starting_update_counts=starting_counts,
                                  evaluation=evaluations[0], evaluations=evaluations,
                                  research_stage='development', implementation_version=continuation['implementation'])
                validate_optimizer_counts(checkpoint, ('critic', 'endpoint', 'generator'), counts)
                save(args.out/f'policy_{transitions}.pt', checkpoint)
                checkpoint['training_state'] = dict(replay=replay.state_dict(), environment=env.state_dict(),
                                                     random=random_state(args.device, generators), best=best,
                                                     outcomes=outcomes.cpu(), episodes=episodes)
                save(args.out/'last.pt', checkpoint)
                persist_evaluations(args.out, checkpoint)
                del checkpoint
                next_eval = min(continuation['final_transitions'], next_eval+continuation['evaluation_interval'])
        final = dict(status='completed' if transitions == continuation['final_transitions'] else 'paused',
                     variant=args.variant, transitions=transitions, updates=learner.updates,
                     continuation=continuation, seconds=time.perf_counter()-started,
                     evaluations=[{k:v for k,v in row.items() if k != 'rows'} for row in evaluations])
        write_json(args.out/'status.json', final)
        print(json.dumps(final), flush=True)
    except BaseException as error:
        write_json(args.out/'status.json', dict(status='failed', variant=args.variant, transitions=transitions,
                                               error=f'{type(error).__name__}: {error}'))
        raise


if __name__ == '__main__':
    main()
