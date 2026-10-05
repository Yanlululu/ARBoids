"""Bounded PPO output-head steps, gated by independent training episodes.

Features and proposals stay fixed during these optional steps. The direction
comes from executed-action log probabilities and real episode outcomes.
Nominal risk is an input, never a replacement for a collision label.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Normal
from torch.nn import functional as F

from parallel_mappo import ParallelRollouts, summarize
from policy.mappo import PredictiveMAPPO


def constrained_direction(fisher, gradient, constraints):
    """Small active-set projection in the regularized policy Fisher metric."""
    metric = fisher.double()
    metric = metric + torch.eye(len(metric), device=metric.device)*metric.diag().mean().clamp_min(1e-8)*1e-3
    gradient, constraints = gradient.double(), constraints.double()
    inverse_gradient = torch.linalg.solve(metric, gradient)
    inverse_constraints = torch.linalg.solve(metric, constraints.T)
    energy = (constraints*inverse_constraints.T).sum(-1)
    valid = energy > 1e-16
    constraints, inverse_constraints = constraints[valid], inverse_constraints[:, valid]
    norms = energy[valid].sqrt()
    constraints = constraints/norms[:, None]
    inverse_constraints = inverse_constraints/norms
    # A small inward margin avoids accepting a first-order boundary step whose
    # finite PPO surrogate immediately violates one of the constraints.
    limits = -.01*(gradient*inverse_gradient).sum().clamp_min(0.).sqrt()*torch.ones_like(norms)
    best, best_value = None, -float('inf')
    for size in range(len(constraints)+1):
        for active in itertools.combinations(range(len(constraints)), size):
            direction = inverse_gradient.clone()
            if active:
                index = list(active)
                rows, columns = constraints[index], inverse_constraints[:, index]
                multipliers = torch.linalg.pinv(rows@columns, hermitian=True, rtol=1e-10)@(rows@inverse_gradient-limits[index])
                if (multipliers < -1e-9).any():
                    continue
                direction -= columns@multipliers
            if (constraints@direction-limits > 1e-8).any():
                continue
            value = float(gradient@direction-.5*direction@metric@direction)
            if value > best_value:
                best, best_value = direction, value
    return best


def propose_gain_step(agent, rollout, gate_head=False):
    """Propose a gain/head step without mutating the policy or its optimizers."""
    cfg = agent.settings
    if (not cfg.compatibility_prior or not cfg.theta_space_gate or not cfg.joint_ratio
            or not cfg.breach_constraint or cfg.target_kl <= 0 or cfg.cost_gamma != 1.
            or not cfg.cost_monte_carlo_targets or cfg.compatibility_peer_intent
            or cfg.compatibility_context_gain):
        raise ValueError('Use a joint theta policy with collision/breach constraints and a KL budget.')
    if len(rollout.episodes) < 4 or any(s.get('curriculum', False) or s.get('seed', 0) < 100000
                                      for s in rollout.summaries):
        raise ValueError('Use at least four ordinary on-policy training episodes.')
    if len({s['seed'] for s in rollout.summaries}) != len(rollout.summaries):
        raise ValueError('Fit and holdout episodes must have distinct seeds.')
    data = rollout.tensors(cfg, agent.device)
    if any(not torch.isfinite(v).all() for v in data.values()):
        raise ValueError('All rollout data must be finite.')
    if 'gate_scale' in data and not (data['gate_scale'] == 1.).all():
        raise ValueError('Use native exploration, with no recovery overrides.')
    means, deviations, signals, head_features = [], [], [], []
    with torch.no_grad():
        for start in range(0, rollout.steps, cfg.minibatch_size):
            batch = {k: v[start:start+cfg.minibatch_size] for k, v in data.items()}
            lp, lg, _ = agent.evaluate_batch(batch)
            if max(float((lp-batch['old_logp_l']).abs().max()),
                   float((lg-batch['old_logp_g']).abs().max())) > 1e-3:
                raise ValueError('The rollout is stale; collect from the unchanged policy.')
            captured = []
            handle = agent.actor.gate_mean.register_forward_pre_hook(lambda module, inputs: captured.append(inputs[0]))
            try:
                dist, _, anchor = agent.actor.gate_distribution(batch['obs'], batch['proposals'],
                    batch['boids'], batch['edges'], batch['mask'], return_anchor=True)
            finally:
                handle.remove()
            means.append(dist.mean)
            deviations.append(dist.stddev)
            signals.append(agent.actor.compatibility_signal(batch['edges'], batch['mask'], batch['obs'], anchor))
            if gate_head:
                head_features.append(captured[0])
    mean, std, signal = (torch.cat(values) for values in (means, deviations, signals))
    old = agent.actor.gate_log_probability(Normal(mean, std), data['gate_raw']).detach()
    fit = torch.tensor(np.concatenate([np.full(len(e), i % 2 == 0) for i, e in enumerate(rollout.episodes)]),
                       device=agent.device)
    subsets = {'fit': fit, 'holdout': ~fit}
    # Exact undiscounted event returns remove dependence on inaccurate temporal
    # bootstraps in this direction check. State-only baselines stay detached.
    cost = data['return_c'] - data['value_c']
    reward, breach = data['advantage_r'], data['advantage_b']
    combined = reward - agent.lagrange.value * cost - agent.breach_lagrange.value * breach
    old_combined = reward - agent.lagrange.value * data['advantage_c'] - agent.breach_lagrange.value * breach
    scale = combined.std(unbiased=False).clamp_min(1e-8)
    advantage = ((combined-combined.mean())/scale).detach()
    old_advantage = ((old_combined-old_combined.mean())/old_combined.std(unbiased=False).clamp_min(1e-8)).detach()
    mean_variable = mean.detach().requires_grad_(True)
    mean_score = torch.autograd.grad(agent.actor.gate_log_probability(
        Normal(mean_variable, std), data['gate_raw']).sum(), mean_variable)[0].detach()
    features = (torch.cat((torch.cat(head_features), torch.ones_like(signal), signal), dim=-1)
                if gate_head else signal)
    scores = (mean_score*features).sum(1)
    score = scores[:, -1]
    gain = float(F.softplus(agent.actor.compatibility_gain_raw.detach()))
    unit_kl = .5*(signal/std).square().flatten(1).sum(-1)
    diagnostics = {name: dict(combined_gradient=float((score[keep]*advantage[keep]).mean()),
        gae_combined_gradient=float((score[keep]*old_advantage[keep]).mean()),
        cost_gradient=float((score[keep]*cost[keep]).mean()),
        breach_gradient=float((score[keep]*breach[keep]).mean()),
        unit_gain_kl=float(unit_kl[keep].mean())) for name, keep in subsets.items()}
    result = dict(accepted=False, gain_before=gain, gain_after=gain, gradients=diagnostics,
                  gate_head=gate_head,
                  steps=rollout.steps, episodes=len(rollout.episodes), target_kl=cfg.target_kl)
    if gate_head:
        fishers = {}
        for name, keep in subsets.items():
            matrix = torch.zeros(features.shape[-1], features.shape[-1], device=agent.device)
            for start in range(0, len(features), cfg.minibatch_size):
                local = keep[start:start+cfg.minibatch_size]
                standardized = (features[start:start+cfg.minibatch_size][local]/std[start:start+cfg.minibatch_size][local]).flatten(0, 1)
                matrix += standardized.T@standardized
            fishers[name] = matrix/int(keep.sum())
        gradient = (scores[fit]*advantage[fit, None]).mean(0)
        constraints = torch.stack([(scores[fit]*value[fit, None]).mean(0) for value in (cost, breach, -reward)])
        projected = constrained_direction(fishers['fit'], gradient, constraints)
        if projected is None or not torch.isfinite(projected).all():
            result['reason'] = 'No joint head direction preserved the task, collision and breach surrogates.'
            return result
        direction = projected.float()
        curvature = max(float(.5*direction@matrix@direction) for matrix in fishers.values())
        result['head_gradient_norm'] = float(gradient.norm())
    else:
        direction = signal.new_tensor([np.sign(diagnostics['fit']['combined_gradient'])])
        curvature = max(v['unit_gain_kl'] for v in diagnostics.values())
    if not np.isfinite(curvature) or curvature <= 0 or not direction.norm():
        return result
    maximum = np.sqrt(cfg.target_kl/curvature)
    if direction[-1] > 0:
        maximum = min(maximum, float((99.-gain)/direction[-1]))
    elif direction[-1] < 0:
        maximum = min(maximum, float(gain*(1.-1e-6)/(-direction[-1])))
    if maximum <= 0:
        result['reason'] = 'The proposed direction is outside the gain bounds.'
        return result
    # A fixed backtracking sequence fits only the fitting episodes; the held-out
    # episodes are inspected once after choosing that step, with no retuning.
    selected = None
    for fraction in (1., .5, .25, .125, .0625, .03125):
        delta = direction*maximum*fraction
        with torch.no_grad():
            shift = (features@delta)[..., None]
            current = agent.actor.gate_log_probability(Normal(mean+shift, std), data['gate_raw'])
            ratio = (current-old).sum(-1).exp()
            clipped = ratio.clamp(1.-cfg.clip_ratio, 1.+cfg.clip_ratio)
            gains = {}
            for name, keep in subsets.items():
                gains[name] = dict(combined=float((torch.minimum(ratio*advantage, clipped*advantage)-advantage)[keep].mean()),
                    cost_reduction=float((torch.minimum(-ratio*cost, -clipped*cost)+cost)[keep].mean()),
                    breach_reduction=float((torch.minimum(-ratio*breach, -clipped*breach)+breach)[keep].mean()),
                    reward_improvement=float((torch.minimum(ratio*reward, clipped*reward)-reward)[keep].mean()),
                    kl=float((.5*(shift[keep]/std[keep]).square()).sum(-1).sum(-1).mean()))
        if (gains['fit']['combined'] > 1e-8 and gains['fit']['cost_reduction'] > 0 and
                gains['fit']['breach_reduction'] >= -1e-8 and
                (not gate_head or gains['fit']['reward_improvement'] >= -1e-8)):
            selected = delta, gains
            break
    if selected is None:
        result['reason'] = 'No fitting step improved task/cost objective while preserving the breach surrogate.'
        return result
    delta, gains = selected
    result['surrogate_changes'] = gains
    held = gains['holdout']
    if (held['combined'] <= 0 or held['cost_reduction'] <= 0 or
            (not gate_head and held['breach_reduction'] < -1e-8)):
        result['reason'] = 'The selected direction did not transfer to the held-out training episodes.'
        return result
    result.update(accepted=True, gain_after=gain+float(delta[-1]))
    if gate_head:
        result['head_weight_delta'] = delta[:-2].cpu().tolist()
        result['head_bias_delta'] = float(delta[-2])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first-seed', type=int, required=True)
    parser.add_argument('--episodes', type=int, default=512)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--gate-head', action='store_true', help='Jointly precondition the existing final gate head and physical gain')
    args = parser.parse_args()
    if args.first_seed < 100000 or args.episodes < 4 or not 1 <= args.workers <= 6 or args.output.exists():
        parser.error('Use fresh training seeds, at least four episodes, at most six workers, and a new output path.')
    torch.set_num_threads(1)
    # Keep likelihood checks on the rollout backend. Cross-device GEMM rounding
    # can exceed the strict per-sample tolerance for low-variance proposals.
    agent = PredictiveMAPPO.from_checkpoint(args.checkpoint, 'cpu')
    agility = float(agent.config.get('training', {}).get('agility', 2.))
    with ParallelRollouts(agent.config, args.workers, 4) as pool:
        rollout = pool.run(agent, np.arange(args.first_seed, args.first_seed+args.episodes), agility)
        print('[BATCH] '+json.dumps(summarize(rollout.summaries)), flush=True)
        result = propose_gain_step(agent, rollout, args.gate_head)
        print('[DIRECTION] '+json.dumps({k: v for k, v in result.items() if k != 'head_weight_delta'}), flush=True)
        if not result['accepted']:
            return
        collision_rate, breach_rate = rollout.collision_rate, rollout.breach_rate
        del rollout
        reference = PredictiveMAPPO.from_checkpoint(args.checkpoint)
        with torch.no_grad():
            agent.actor.compatibility_gain_raw.fill_(float(np.log(np.expm1(result['gain_after']))))
            if args.gate_head:
                agent.actor.gate_mean.weight.add_(agent.tensor(result['head_weight_delta'])[None])
                agent.actor.gate_mean.bias.add_(result['head_bias_delta'])
                for parameter in agent.actor.gate_mean.parameters():
                    agent.actor_optimizer.state.pop(parameter, None)
        agent.actor_optimizer.state.pop(agent.actor.compatibility_gain_raw, None)
        comparison_start = args.first_seed+max(10000, args.episodes)
        seeds = np.arange(comparison_start, comparison_start+args.episodes)
        before = pool.run(reference, seeds, agility, deterministic=True, retain=False)
        after = pool.run(agent, seeds, agility, deterministic=True, retain=False)
    a, b = summarize(before), summarize(after)
    passed = (b['success_rate'] >= a['success_rate'] and b['collision_rate'] < a['collision_rate']
              and b['breach_rate'] <= a['breach_rate'])
    evidence = dict(before=a, after=b, local_improvement=passed, performance_qualified=False,
                    training_first_seed=args.first_seed, comparison_first_seed=comparison_start, step=result)
    print('[DECISION] '+json.dumps({**evidence, 'step': {k: v for k, v in result.items() if k != 'head_weight_delta'}}), flush=True)
    if passed:
        agent.updates += 1
        agent.total_steps += result['steps']
        agent.lagrange.update(collision_rate)
        agent.breach_lagrange.update(breach_rate)
        agent.training_state.update(stable_count=0, performance_qualified=False, best_score=[-1., -1., -1.])
        agent.training_state.setdefault('optimization_changes', []).append(
            {('gate_head_ppo' if args.gate_head else 'scalar_prior_ppo'): evidence})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        agent.save(args.output)
        print('[CANDIDATE] '+str(args.output), flush=True)


if __name__ == '__main__':
    main()
