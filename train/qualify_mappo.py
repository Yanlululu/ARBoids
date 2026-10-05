"""Freeze one candidate; development gates precede a fresh paired audit. No training."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from parallel_mappo import ParallelRollouts, summarize
from policy.mappo import PredictiveMAPPO
from train_mappo_performance import (baseline_reference, protected_breach_reference,
                                    qualifies, validation_result, write_evaluation, meets_targets,
                                    require_fresh_audit)


ROOT = Path(__file__).resolve().parent


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    center = (p + z*z/(2*count)) / (1 + z*z/count)
    half = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / (1 + z*z/count)
    return [center-half, center+half]


def completed_development(directory, checkpoint, baseline_hash, manifest_hash, seeds):
    """Reuse only a completed, hash-bound measurement of this exact policy."""
    criteria = json.loads((directory/'criteria.json').read_text(encoding='utf-8'))
    metrics = json.loads((directory/'validation.json').read_text(encoding='utf-8'))
    if not (directory/'qualification.json').exists() or metrics.get('complete') is False:
        raise ValueError('Development reuse requires a completed frozen evaluation.')
    source_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    frozen_hash = hashlib.sha256((directory/'candidate.pth').read_bytes()).hexdigest()
    if (frozen_hash != criteria['candidate_sha256'] or
            source_hash not in (frozen_hash, criteria.get('source_checkpoint_sha256'))):
        raise ValueError('Development results belong to a different checkpoint.')
    if criteria['baseline_sha256'] != baseline_hash:
        raise ValueError('Development results use a different original baseline.')
    if metrics.get('baseline_manifest_sha256', criteria.get('development_manifest_sha256')) != manifest_hash:
        raise ValueError('Development results use a different or unrecorded evaluation manifest.')
    csv_path = directory/'validation.csv'
    with csv_path.open(newline='', encoding='utf-8') as file:
        rows = [{key: (int(value) if key in ('seed', 'steps', 'outcome_code') else float(value))
                 for key, value in row.items()} for row in csv.DictReader(file)]
    if [row['seed'] for row in rows] != list(map(int, seeds)):
        raise ValueError('Development results must cover the exact ordered scenario manifest.')
    for row in rows:
        if (not all(math.isfinite(value) for value in row.values()) or
                any(row[key] not in (0, 1) for key in ('success', 'capture', 'timeout', 'breach', 'collision'))):
            raise ValueError('Development results contain invalid physical event labels or values.')
    if any(not math.isclose(value, metrics[key], rel_tol=1e-12, abs_tol=1e-12)
           for key, value in summarize(rows).items()):
        raise ValueError('Development rows do not match their completed summary.')
    provenance = dict(directory=str(directory.resolve()), candidate_sha256=frozen_hash,
        episodes=len(rows), rows_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest())
    return rows, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--comparison-checkpoint', type=Path,
                        help='Optional previous candidate, compared on the same fresh audit scenarios')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--compatibility-prior', action='store_true')
    parser.add_argument('--peer-intent', action='store_true', help='Explicit experimental intent-weighted risk prior')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--baseline', type=Path, default=ROOT/'experiments/paper-parameters-seed42-20260921/adares1.pth')
    parser.add_argument('--baseline-validation', type=Path,
                        default=ROOT/'experiments/predictive-mappo-performance-development800/summary.json')
    parser.add_argument('--development-results', type=Path,
                        help='Reuse completed frozen development rows for this exact checkpoint and scenario manifest')
    parser.add_argument('--audit-seed', type=int, default=50000)
    parser.add_argument('--audit-episodes', type=int, default=1000)
    parser.add_argument('--report-batch-size', type=int, default=0,
                        help='Optional disjoint audit grouping for reporting; does not change acceptance gates')
    parser.add_argument('--min-success-rate', type=float, default=0.)
    parser.add_argument('--max-collision-rate', type=float, default=1.)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--development-only', action='store_true',
                      help='Compare on the frozen development suite without consuming audit scenarios')
    mode.add_argument('--report-nonqualifying', action='store_true',
                      help='Also measure a nonqualifying development candidate; strict acceptance criteria stay unchanged')
    args = parser.parse_args()
    if args.workers < 1 or args.audit_episodes < 200 or args.audit_episodes % 200:
        parser.error('Use positive workers and complete audit blocks of 200 episodes.')
    if args.report_batch_size < 0 or (args.report_batch_size and args.audit_episodes % args.report_batch_size):
        parser.error('Report batch size must be positive and divide the audit episode count, or be zero.')
    if not (0 <= args.min_success_rate <= 1 and 0 <= args.max_collision_rate <= 1):
        parser.error('Performance targets must be finite rates in [0,1].')
    if args.output_dir.exists():
        parser.error('Use a new candidate directory; never overwrite a previous frozen audit.')
    if args.development_results and (args.compatibility_prior or args.peer_intent):
        parser.error('Development reuse cannot accompany a policy modification.')
    torch.set_num_threads(1)
    agent = PredictiveMAPPO.from_checkpoint(args.checkpoint, 'cpu')
    if args.compatibility_prior:
        agent.enable_compatibility_prior(peer_intent=args.peer_intent)
    elif args.peer_intent:
        parser.error('--peer-intent requires --compatibility-prior; existing checkpoints retain their saved settings.')
    comparison = None
    comparison_hash = None
    if args.comparison_checkpoint:
        comparison = PredictiveMAPPO.from_checkpoint(args.comparison_checkpoint, 'cpu')
        if comparison.config['environment'] != agent.config['environment'] or comparison.defender_num != agent.defender_num:
            raise ValueError('The previous candidate must use the same environment and team size.')
        comparison_hash = hashlib.sha256(args.comparison_checkpoint.read_bytes()).hexdigest()
    baseline = json.loads(args.baseline_validation.read_text(encoding='utf-8'))
    protected_breach_reference(baseline, args.baseline_validation.parent/'episodes.csv')
    baseline_hash = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    if baseline_hash != baseline['checkpoint_sha256']:
        raise ValueError('The original baseline checkpoint must match its frozen development manifest.')
    if (agent.config['environment']['protocol'] != baseline['protocol'] or
            agent.config['environment']['total_time'] != baseline['duration']):
        raise ValueError('Reference and candidate protocols must match.')
    seeds = np.asarray(baseline['seeds'], dtype=np.int64)
    cached_rows, development_reuse = (None, None)
    if args.development_results:
        cached_rows, development_reuse = completed_development(
            args.development_results, args.checkpoint, baseline_hash,
            hashlib.sha256(args.baseline_validation.read_bytes()).hexdigest(), seeds)
    audit_seeds = np.arange(args.audit_seed, args.audit_seed + args.audit_episodes)
    if not args.development_only and set(seeds) & set(audit_seeds):
        raise ValueError('Development and independent audit scenarios must not overlap.')
    # A managed worktree shares input checkpoints with the original checkout.
    # Previously inspected audits remain used after moving to another worktree.
    if not args.development_only:
        require_fresh_audit(audit_seeds, (ROOT/'experiments',
            args.baseline_validation.parent.parent, args.baseline.parent.parent, args.checkpoint.parent.parent))
    args.output_dir.mkdir(parents=True)
    agent.training_state.update(performance_qualified=False, stable_count=0)
    candidate_path = args.output_dir/'candidate.pth'
    agent.save(candidate_path)
    candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    criteria = dict(baseline_sha256=baseline_hash, candidate_sha256=candidate_hash,
        development_manifest_sha256=hashlib.sha256(args.baseline_validation.read_bytes()).hexdigest(),
        source_checkpoint=str(args.checkpoint.resolve()), extra_optimizer_updates=0,
        source_checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        development_only=args.development_only, report_nonqualifying=args.report_nonqualifying,
        preserve_success=True, preserve_breach=True, strict_collision_improvement=True,
        collision_budget=agent.settings.collision_budget, development_episodes=len(seeds),
        audit_first_seed=args.audit_seed, audit_episodes=len(audit_seeds),
        report_batch_size=args.report_batch_size,
        stability_block_size=200, require_each_block=True, require_collision_upper95_within_budget=True,
        stability_block_criteria='success_not_lower_and_collision_strictly_lower_within_budget',
        compatibility_prior=agent.settings.compatibility_prior, purpose='performance_development',
        compatibility_compact_risk=agent.settings.compatibility_compact_risk,
        compatibility_task_priority=agent.settings.compatibility_task_priority,
        compatibility_peer_intent=agent.settings.compatibility_peer_intent,
        censored_gate_likelihood=agent.settings.censored_gate_likelihood,
        breach_constraint=agent.settings.breach_constraint,
        compatibility_mixture_points=agent.settings.compatibility_mixture_points,
        compatibility_joint_mixture=agent.settings.compatibility_joint_mixture,
        compatibility_task_prediction=agent.settings.compatibility_task_prediction,
        compatibility_terminal_prediction=agent.settings.compatibility_terminal_prediction,
        compatibility_context_gain=agent.settings.compatibility_context_gain,
        minimize_collision=agent.settings.minimize_collision,
        training_success_floor=agent.settings.success_floor if agent.settings.minimize_collision else None,
        min_success_rate=args.min_success_rate, max_collision_rate=args.max_collision_rate,
        target_scope='aggregate_point_estimate',
        formal_experiments_started=False)
    if comparison is not None:
        criteria.update(comparison_checkpoint=str(args.comparison_checkpoint.resolve()),
                        comparison_sha256=comparison_hash)
    if development_reuse is not None:
        criteria['development_reuse'] = development_reuse
    (args.output_dir/'criteria.json').write_text(json.dumps(criteria, indent=2), encoding='utf-8')
    print('[FROZEN] ' + json.dumps(criteria), flush=True)
    passed = False
    with ParallelRollouts(agent.config, args.workers) as pool:
        rows = (cached_rows if cached_rows is not None else
                pool.run(agent, seeds, baseline['agility'], deterministic=True, retain=False))
        metrics, development_passed, _ = validation_result(rows, baseline, agent.settings.collision_budget, True)
        development_passed = development_passed and meets_targets(
            metrics, args.min_success_rate, args.max_collision_rate)
        write_evaluation(args.output_dir, 'validation', metrics, rows,
                         performance_passed=development_passed, update=agent.updates)
        print('[DEVELOPMENT] ' + json.dumps(dict(passed=development_passed, **metrics)), flush=True)
        if not args.development_only and (development_passed or args.report_nonqualifying):
            original = baseline_reference(agent.config, torch.load(args.baseline, map_location='cpu', weights_only=True))
            before = pool.run(original, audit_seeds, baseline['agility'], deterministic=True, retain=False)
            print('[AUDIT_REFERENCE] ' + json.dumps(summarize(before)), flush=True)
            after = pool.run(agent, audit_seeds, baseline['agility'], deterministic=True, retain=False)
            a, b = summarize(before), summarize(after)
            blocks = []
            for start in range(0, len(audit_seeds), 200):
                left, right = summarize(before[start:start+200]), summarize(after[start:start+200])
                blocks.append(dict(first_seed=int(audit_seeds[start]), reference=left, candidate=right,
                    passed=qualifies(right, left['success_rate'], agent.settings.collision_budget,
                                     left['collision_rate'])))
            interval = wilson(sum(row['collision'] for row in after), len(after))
            batches = []
            if args.report_batch_size:
                for start in range(0, len(audit_seeds), args.report_batch_size):
                    left = before[start:start+args.report_batch_size]
                    right = after[start:start+args.report_batch_size]
                    batches.append(dict(first_seed=int(audit_seeds[start]), episodes=len(right),
                        reference=summarize(left), candidate=summarize(right),
                        event_counts={side: {event: int(sum(row[event] for row in group))
                            for event in ('success', 'collision', 'breach')}
                            for side, group in (('reference', left), ('candidate', right))},
                        candidate_intervals_95={event: wilson(sum(row[event] for row in right), len(right))
                            for event in ('success', 'collision', 'breach')}))
            passed = (development_passed and qualifies(b, a['success_rate'], agent.settings.collision_budget,
                               a['collision_rate'], a['breach_rate']) and
                      all(block['passed'] for block in blocks) and interval[1] <= agent.settings.collision_budget and
                      meets_targets(b, args.min_success_rate, args.max_collision_rate))
            paired = {event: {f'{x}{y}': sum(row[event] == x and other[event] == y
                       for row, other in zip(before, after)) for x in (0, 1) for y in (0, 1)}
                      for event in ('success', 'collision', 'breach')}
            comparison_result = {}
            if comparison is not None:
                previous_rows = pool.run(comparison, audit_seeds, baseline['agility'], deterministic=True, retain=False)
                previous_metrics = summarize(previous_rows)
                previous_paired = {event: {f'{x}{y}': sum(row[event] == x and other[event] == y
                    for row, other in zip(previous_rows, after)) for x in (0, 1) for y in (0, 1)}
                    for event in ('success', 'collision', 'breach')}
                write_evaluation(args.output_dir, 'audit-previous', previous_metrics, previous_rows,
                    checkpoint_sha256=comparison_hash, first_seed=args.audit_seed, episodes=len(previous_rows))
                comparison_result = dict(previous_candidate=previous_metrics, paired_previous_candidate=previous_paired)
                print('[AUDIT_PREVIOUS] ' + json.dumps(comparison_result), flush=True)
            write_evaluation(args.output_dir, 'audit-baseline', a, before,
                checkpoint_sha256=baseline_hash, first_seed=args.audit_seed, episodes=len(before),
                confidence_intervals_95={event: wilson(sum(row[event] for row in before), len(before))
                    for event in ('success', 'collision', 'breach')})
            write_evaluation(args.output_dir, 'audit-candidate', b, after,
                performance_passed=bool(passed), candidate_checkpoint_sha256=candidate_hash,
                first_seed=args.audit_seed, episodes=len(after), confidence_intervals_95=dict(
                    collision=interval, success=wilson(sum(row['success'] for row in after), len(after)),
                    breach=wilson(sum(row['breach'] for row in after), len(after))),
                paired_counts=paired, stability_blocks=blocks, report_batches=batches, **comparison_result)
            if passed:
                agent.training_state.update(performance_qualified=True, qualified_scene_blocks=len(blocks),
                                            stability_protocol='fixed_model_scene_blocks')
                agent.save(args.output_dir/'qualified.pth')
            print('[AUDIT] ' + json.dumps(dict(passed=bool(passed), reference=a, candidate=b,
                collision_interval95=interval, blocks_passed=sum(block['passed'] for block in blocks),
                blocks=len(blocks), paired=paired)), flush=True)
    (args.output_dir/'qualification.json').write_text(json.dumps(dict(
        performance_passed=bool(passed), development_passed=bool(development_passed),
        criteria=criteria, formal_experiments_started=False), indent=2), encoding='utf-8')
    print(f'[COMPLETE] performance_passed={bool(passed)} extra_optimizer_updates=0 formal_experiments_started=False', flush=True)


if __name__ == '__main__':
    main()
