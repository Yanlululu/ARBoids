"""Bounded recovery diagnoses and one-update probes on training scenarios."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from parallel_mappo import ParallelRollouts, collision_branch_cases, breach_branch_cases, summarize
from policy.mappo import PredictiveMAPPO


def paired_breach_probe(checkpoint, first_seed, source_seeds, repeats=16, workers=4):
    """Separate policy effects from changed water-current realizations."""
    if first_seed < 100000 or not source_seeds or any(s < 100000 for s in source_seeds):
        raise ValueError('Explicit training-only sources and future seeds are required.')
    if repeats < 2 or not 1 <= workers <= 6:
        raise ValueError('Use at least two continuations and at most six workers.')
    torch.set_num_threads(1)
    agent = PredictiveMAPPO.from_checkpoint(checkpoint)
    agent.config.setdefault('training', {}).update(branch_paired_reference=True,
        branch_gate_scale=1., branch_gate_steps=20)
    with ParallelRollouts(agent.config, workers) as pool:
        sources = pool.run(agent, source_seeds, 2., deterministic=True)
        cases = breach_branch_cases(agent, sources, len(source_seeds)*repeats, first_seed,
                                    max_cases=len(source_seeds))
        if len(cases) != len(source_seeds):
            raise ValueError('Every requested source must be a deterministic training breach.')
        for i, case in enumerate(cases):
            case['seeds'] = list(range(first_seed+i*repeats, first_seed+(i+1)*repeats))
        rollout, _ = pool.resume_cases(agent, cases)
    rows = rollout.summaries
    references = [{key: row['paired_reference_'+key] for key in
                   ('success', 'capture', 'timeout', 'breach', 'collision', 'task_return')}
                   | dict(gate_mean=0.) for row in rows]
    result = dict(actor_updated=False, gate_scale=1., episodes=len(rows),
        reference=summarize(references), sampled=summarize(rows),
        paired={event: {f'{x}{y}': sum(a[event] == x and b[event] == y
                    for a, b in zip(references, rows)) for x in (0,1) for y in (0,1)}
                for event in ('success', 'collision', 'breach')})
    estimates = []
    for case in cases:
        selected = [(episode, row) for episode, row in zip(rollout.episodes, rows)
                    if row['source_seed'] == case['source_seed']]
        first = [episode[0] for episode, _ in selected]
        tensors = [agent.tensor(np.stack([r[key] for r in first]), key=='mask')
                   for key in ('obs','proposals','boids','edges','mask')]
        with torch.no_grad():
            distribution, _ = agent.actor.gate_distribution(*tensors)
            raw = agent.tensor(np.stack([r['gate_raw'] for r in first]))
            score = ((raw-distribution.mean)/distribution.stddev.square()).flatten(1).numpy()
        actual = np.asarray([r['task_return']-agent.lagrange.value*(r['collision']+r['breach'])
                             for _, r in selected])
        reference = np.asarray([r['paired_reference_task_return']-agent.lagrange.value*(
            r['paired_reference_collision']+r['paired_reference_breach']) for _, r in selected])
        loo = (actual.sum()-actual)/(len(actual)-1)
        estimates.append(dict(source_seed=case['source_seed'], source_step=case['source_step'],
            loo_score_variance=float(((actual-loo)[:,None]*score).var(0).mean()),
            paired_score_variance=float(((actual-reference)[:,None]*score).var(0).mean())))
    result['root_signal'] = estimates
    print('[PAIRED_ENVIRONMENT] '+json.dumps(result), flush=True)
    return result


def single_update(checkpoint, first_seed, output, workers=4):
    """One evidence-gated update, preserving task reward and the existing actor."""
    if first_seed < 100000:
        raise ValueError('Use training seeds for the bounded learning probe.')
    torch.set_num_threads(1)
    agent = PredictiveMAPPO.from_checkpoint(checkpoint, 'cuda:0' if torch.cuda.is_available() else 'cpu')
    if agent.settings.minimize_collision:
        raise ValueError('Start from the preserved task-reward objective, not the unsuccessful replacement.')
    torch.manual_seed(4202)
    np.random.seed(4202)
    for key, value in dict(cost_gae_lambda=1., cost_monte_carlo_targets=True,
                           curriculum_cost_baseline=True, guard_task_surrogate=True,
                           full_batch_guard=True).items():
        setattr(agent.settings, key, value)
        agent.config['mappo'][key] = value
    agent.settings.__post_init__()
    agent.config['training'].update(branch_gate_scale=4., branch_gate_steps=20, branch_max_cases=8)
    with ParallelRollouts(agent.config, workers, 4) as pool:
        ordinary = pool.run(agent, np.arange(first_seed, first_seed + 512), 2., noisy=True)
        cases = collision_branch_cases(ordinary, 128, 20, first_seed + 10000, max_cases=8)
        branches, _ = pool.resume_cases(agent, cases, 2., True)
        print('[BATCH] ' + json.dumps(dict(ordinary=summarize(ordinary.summaries),
                                           recovery=summarize(branches.summaries))), flush=True)
        for episode, summary in zip(branches.episodes, branches.summaries):
            ordinary.add_episode(episode, summary)
        data = ordinary.tensors(agent.settings, agent.device)
        start = data['group_cost_weight'] == 1.
        critic_error = float((data['return_c'][start] - data['value_c'][start]).square().mean())
        group_error = float((data['return_c'][start] - data['group_cost_baseline'][start]).square().mean())
        print('[SIGNAL] ' + json.dumps(dict(critic_start_brier=critic_error, independent_start_brier=group_error)), flush=True)
        del data
        if not group_error < critic_error:
            print('[STOP] The independent baseline did not improve this batch; actor was not updated.', flush=True)
            return
        metrics = agent.update(ordinary)
        print('[UPDATE] ' + json.dumps(metrics), flush=True)
        if not metrics['actor_accepted_steps']:
            print('[STOP] No update preserved both task and collision surrogates; actor and Adam were restored.', flush=True)
            return
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        agent.training_state.update(stable_count=0, performance_qualified=False,
                                    best_score=[-1., -1., -1.])
        agent.save(output)
        print('[CANDIDATE] ' + str(output), flush=True)
        # A small fresh paired comparison, only after a nonzero accepted step.
        reference = PredictiveMAPPO.from_checkpoint(checkpoint, 'cpu')
        seeds = np.arange(first_seed + 20000, first_seed + 20256)
        before = pool.run(reference, seeds, 2., deterministic=True, retain=False)
        after = pool.run(agent, seeds, 2., deterministic=True, retain=False)
        a, b = summarize(before), summarize(after)
        evidence = dict(reference=a, candidate=b, episodes=len(seeds), actor_updated=True,
                        paired={name: dict(resolved=sum(x[name] and not y[name] for x, y in zip(before, after)),
                                           introduced=sum(y[name] and not x[name] for x, y in zip(before, after)))
                                for name in ('collision', 'breach')})
        evidence['local_improvement'] = bool(b['success_rate'] >= a['success_rate'] and
            b['collision_rate'] < a['collision_rate'] and b['breach_rate'] <= a['breach_rate'])
        print('[DECISION] ' + json.dumps(evidence), flush=True)


def early_breach_update(checkpoint, first_seed, output, workers=4, source_seeds=None, censored_gate=False,
                        breach_constraint=False):
    """One PPO step after demonstrated early recovery; keep rewards and duals."""
    if first_seed < 100000 or any(seed < 100000 for seed in (source_seeds or ())):
        raise ValueError('Only training scenarios may guide the update.')
    torch.set_num_threads(1)
    agent = PredictiveMAPPO.from_checkpoint(checkpoint, 'cuda:0' if torch.cuda.is_available() else 'cpu')
    if agent.settings.minimize_collision:
        raise ValueError('This probe preserves the task-return objective.')
    if censored_gate:
        agent.enable_censored_gate_likelihood()
    if breach_constraint:
        reference = json.loads((Path(__file__).resolve().parent/
            'experiments/predictive-mappo-performance-development800/summary.json').read_text(encoding='utf-8'))
        agent.enable_breach_constraint(reference['breach_rate'])
    torch.manual_seed(4203)
    np.random.seed(4203)
    for name, value in dict(gae_lambda=1., cost_gae_lambda=1., cost_monte_carlo_targets=True,
        curriculum_cost_baseline=True, curriculum_reward_baseline=True,
        full_batch_guard=True, guard_task_surrogate=True).items():
        setattr(agent.settings, name, value)
        agent.config['mappo'][name] = value
    agent.settings.__post_init__()
    agent.config['training'].update(branch_gate_scale=4., branch_gate_steps=20,
        collision_branches=64, breach_branches=64, branch_max_cases=4, breach_max_cases=4)
    with ParallelRollouts(agent.config, workers, 4) as pool:
        ordinary = pool.run(agent, np.arange(first_seed, first_seed+512), 2., noisy=True)
        print('[ORDINARY] ' + json.dumps(summarize(ordinary.summaries)), flush=True)
        collisions = collision_branch_cases(ordinary, 64, 20, first_seed+10000, max_cases=4)
        collision_rollout, _ = pool.resume_cases(agent, collisions, 2., True)
        sources = pool.run(agent, source_seeds, 2., deterministic=True) if source_seeds else ordinary
        breaches = breach_branch_cases(agent, sources, 64, first_seed+11000, max_cases=4)
        breach_rollout, _ = pool.resume_cases(agent, breaches, 2., not bool(source_seeds))
        if not breach_rollout.summaries or not any(row['success'] for row in breach_rollout.summaries):
            print('[STOP] No successful early breach continuation; no optimizer update.', flush=True)
            return
        print('[EARLY_RECOVERY] ' + json.dumps(summarize(breach_rollout.summaries)), flush=True)
        if collision_rollout.summaries:
            print('[COLLISION_RECOVERY] ' + json.dumps(summarize(collision_rollout.summaries)), flush=True)
        for batch in (collision_rollout, breach_rollout):
            for episode, summary in zip(batch.episodes, batch.summaries):
                ordinary.add_episode(episode, summary)
        data = ordinary.tensors(agent.settings, agent.device)
        roots = data['group_reward_weight'] == 1.
        critic_error = float((data['return_r'][roots]-data['value_r'][roots]).square().mean())
        group_error = float((data['return_r'][roots]-data['group_reward_baseline'][roots]).square().mean())
        if agent.settings.breach_constraint:
            old_brier = float((data['return_b'][roots]-data['value_b'][roots]).square().mean())
            independent_brier = float((data['return_b'][roots]-data['group_breach_baseline'][roots]).square().mean())
            print('[BREACH_SIGNAL] ' + json.dumps(dict(critic_brier=old_brier,
                independent_brier=independent_brier, budget=agent.settings.breach_budget,
                multiplier=agent.breach_lagrange.value)), flush=True)
            if not independent_brier < old_brier:
                print('[STOP] The independent breach baseline did not improve; no optimizer update.', flush=True)
                return
        del data
        print('[TASK_SIGNAL] ' + json.dumps(dict(critic_mse=critic_error, independent_mse=group_error)), flush=True)
        if not group_error < critic_error:
            print('[STOP] Task baseline did not improve; no optimizer update.', flush=True)
            return
        metrics = agent.update(ordinary)
        print('[UPDATE] ' + json.dumps(metrics), flush=True)
        if not metrics['actor_accepted_steps']:
            print('[STOP] Task/cost/KL guard rejected the step; actor and Adam restored.', flush=True)
            return
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        agent.training_state.update(stable_count=0, performance_qualified=False, best_score=[-1., -1., -1.])
        agent.training_state.setdefault('recovery_changes', []).append(dict(update=agent.updates,
            early_breach_recovery=True, independent_task_baseline=True, training_sources=source_seeds))
        agent.save(output)
        del ordinary, sources, collision_rollout, breach_rollout
        reference = PredictiveMAPPO.from_checkpoint(checkpoint, 'cpu')
        seeds = np.arange(first_seed+20000, first_seed+20256)
        before = pool.run(reference, seeds, 2., deterministic=True, retain=False)
        after = pool.run(agent, seeds, 2., deterministic=True, retain=False)
        a, b = summarize(before), summarize(after)
        improved = (b['success_rate'] >= a['success_rate'] and b['collision_rate'] <= a['collision_rate'] and
                    b['breach_rate'] <= a['breach_rate'] and
                    (b['collision_rate'] < a['collision_rate'] or b['breach_rate'] < a['breach_rate']))
        print('[DECISION] ' + json.dumps(dict(reference=a, candidate=b, local_improvement=bool(improved),
            paired={name: dict(resolved=sum(x[name] and not y[name] for x,y in zip(before,after)),
                introduced=sum(y[name] and not x[name] for x,y in zip(before,after)))
                for name in ('collision','breach')}, checkpoint=str(output))), flush=True)


def probe(checkpoint, first_seed, sources=256, cases_limit=12, repeats=16, workers=4,
          scales=(1., 4.), exploration_steps=0, source_seeds=None):
    if first_seed < 100000 or min(sources, cases_limit, workers, repeats) < 1:
        raise ValueError('Use fresh training seeds and positive probe sizes.')
    torch.set_num_threads(1)
    agent = PredictiveMAPPO.from_checkpoint(checkpoint, 'cpu')
    torch.manual_seed(4201)
    np.random.seed(4201)
    with ParallelRollouts(agent.config, workers, 4) as pool:
        seeds = np.arange(first_seed, first_seed + sources) if source_seeds is None else source_seeds
        if any(seed < 100000 for seed in seeds):
            raise ValueError('Recovery sources must not use reserved validation seeds.')
        source = pool.run(agent, seeds, 2.,
                          noisy=False, deterministic=True)
        print('[SOURCE] ' + json.dumps(dict(episodes=sources, **summarize(source.summaries))), flush=True)
        # These are states reached by deterministic deployment on training
        # seeds, not stochastic training accidents or reserved validation cases.
        cases = collision_branch_cases(source, cases_limit, 20, first_seed + 10000)
        if not cases:
            raise ValueError('No collision sources; do not infer recovery improvement from an empty probe.')
        if source_seeds is not None:
            by_seed = {case['source_seed']: case for case in cases}
            cases = [by_seed[seed] for seed in source_seeds]
        for index, case in enumerate(cases):
            case['seeds'] = list(range(first_seed + 20000 + repeats * index,
                                       first_seed + 20000 + repeats * (index + 1)))
        groups = {}
        for scale in scales:
            agent.config.setdefault('training', {})['branch_gate_scale'] = scale
            agent.config['training']['branch_gate_steps'] = exploration_steps
            rollout, _ = pool.resume_cases(agent, cases, 2., False)
            groups[scale] = rollout
            print('[RECOVERY] ' + json.dumps(dict(gate_scale=scale, exploration_steps=exploration_steps,
                                                 episodes=len(rollout.summaries),
                                                 **summarize(rollout.summaries))), flush=True)
        if len(scales) == 1:
            rows = groups[scales[0]].summaries
            result = dict(gate_scale=scales[0], exploration_steps=exploration_steps, actor_updated=False,
                          **summarize(rows), per_source=[dict(seed=case['source_seed'],
                          **summarize([r for r in rows if r['source_seed'] == case['source_seed']]))
                          for case in cases])
            print('[RESULT] ' + json.dumps(result), flush=True)
            return result
        left, right = groups[1.], groups[4.]
        if [(r['source_seed'], r['seed']) for r in left.summaries] != [
                (r['source_seed'], r['seed']) for r in right.summaries]:
            raise ValueError('Paired continuations do not share their start states and future seeds.')
        result = dict(checkpoint=str(checkpoint), source_seeds=[c['source_seed'] for c in cases],
                      paired_episodes=len(left.summaries))
        for event in ('collision', 'success', 'breach'):
            a = np.asarray([r[event] for r in left.summaries])
            b = np.asarray([r[event] for r in right.summaries])
            result[event] = dict(original=float(a.mean()), expanded=float(b.mean()),
                                 zero_to_one=int(((a == 0) & (b == 1)).sum()),
                                 one_to_zero=int(((a == 1) & (b == 0)).sum()))
        result['per_source'] = []
        for case in cases:
            rows = [[r for r in rollout.summaries if r['source_seed'] == case['source_seed']]
                    for rollout in (left, right)]
            result['per_source'].append(dict(seed=case['source_seed'],
                original=summarize(rows[0]), expanded=summarize(rows[1])))
        result['actor_updated'] = False
        print('[PAIRED] ' + json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--first-seed', type=int, required=True)
    parser.add_argument('--sources', type=int, default=256)
    parser.add_argument('--cases', type=int, default=12)
    parser.add_argument('--repeats', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--scales', type=float, nargs='+', default=[1., 4.])
    parser.add_argument('--exploration-steps', type=int, default=0)
    parser.add_argument('--source-seeds', type=int, nargs='+')
    parser.add_argument('--single-update', action='store_true')
    parser.add_argument('--early-breach-update', action='store_true')
    parser.add_argument('--paired-breach-probe', action='store_true')
    parser.add_argument('--censored-gate', action='store_true')
    parser.add_argument('--breach-constraint', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if (args.censored_gate or args.breach_constraint) and not args.early_breach_update:
        parser.error('The probability/constraint migrations require --early-breach-update.')
    if args.paired_breach_probe:
        if args.early_breach_update or args.single_update or args.output:
            parser.error('The paired diagnosis does not run an update or write a checkpoint.')
        paired_breach_probe(args.checkpoint, args.first_seed, args.source_seeds, args.repeats, args.workers)
    elif args.early_breach_update:
        if args.output is None or args.single_update:
            parser.error('Use --early-breach-update with --output and without --single-update.')
        early_breach_update(args.checkpoint, args.first_seed, args.output, args.workers, args.source_seeds,
                            args.censored_gate, args.breach_constraint)
    elif args.single_update:
        if args.output is None:
            parser.error('--single-update requires an explicit candidate --output path')
        single_update(args.checkpoint, args.first_seed, args.output, args.workers)
    else:
        probe(args.checkpoint, args.first_seed, args.sources, args.cases, args.repeats, args.workers,
              args.scales, args.exploration_steps, args.source_seeds)
