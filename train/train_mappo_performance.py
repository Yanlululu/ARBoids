"""Bounded performance development with exact actor transfer and validation gates.

This entry point never launches formal experiments. Validation seeds may guide
development; audit seeds are evaluated only after consecutive qualifying checks.
"""
import argparse
import copy
import csv
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
import yaml

from parallel_mappo import ParallelRollouts, summarize, collision_branch_cases, breach_branch_cases
from policy.mappo import PredictiveMAPPO, LagrangeMultiplier
from train_mappo import evaluation_rng
from utils.manager import ExperimentManager, set_seed


ROOT = Path(__file__).resolve().parent


def qualifies(metrics, baseline_success, collision_budget, baseline_collision, baseline_breach=None):
    return bool(metrics['success_rate'] + 1e-12 >= baseline_success and
                metrics['collision_rate'] <= collision_budget + 1e-12 and
                metrics['collision_rate'] < baseline_collision - 1e-12 and
                (baseline_breach is None or metrics['breach_rate'] <= baseline_breach + 1e-12))


def meets_targets(metrics, min_success_rate=0., max_collision_rate=1.):
    """Point-estimate targets; uncertainty is reported separately."""
    return (metrics['success_rate'] + 1e-12 >= min_success_rate and
            metrics['collision_rate'] <= max_collision_rate + 1e-12)


def require_fresh_audit(seeds, roots):
    """Reject any overlap with completed audits, including another worktree."""
    requested = set(map(int, seeds))
    roots = {Path(root).resolve() for root in roots}
    previous_audits = {path for root in roots for path in root.glob('*/audit-candidate.json')}
    for previous in sorted(previous_audits):
        if previous.with_suffix('.csv').exists():
            with previous.with_suffix('.csv').open(newline='', encoding='utf-8') as file:
                used = {int(row['seed']) for row in csv.DictReader(file)}
        else:
            prior = json.loads(previous.read_text(encoding='utf-8'))
            used = set(range(prior['first_seed'], prior['first_seed'] + prior['episodes']))
        if requested & used:
            raise ValueError(f'Audit seeds have already been used: {previous}')


def validation_result(rows, baseline, collision_budget, preserve_breach=False):
    """An expanded development suite cannot hide regression on the original set."""
    metrics = summarize(rows)
    passed = qualifies(metrics, baseline['success_rate'], collision_budget, baseline['collision_rate'],
                       baseline['breach_rate'] if preserve_breach else None)
    success_gap = metrics['success_rate'] - baseline['success_rate']
    breach_gap = baseline['breach_rate'] - metrics['breach_rate'] if preserve_breach else 0.
    protected = baseline.get('protected_reference')
    if protected:
        seeds = set(protected['seeds'])
        subset = [row for row in rows if row['seed'] in seeds]
        if len(subset) != len(seeds):
            raise ValueError('The original development scenarios must remain in the expanded suite.')
        primary = summarize(subset)
        passed = passed and qualifies(primary, protected['success_rate'], collision_budget,
                                      protected['collision_rate'],
                                      protected['breach_rate'] if preserve_breach else None)
        success_gap = min(success_gap, primary['success_rate'] - protected['success_rate'])
        if preserve_breach:
            breach_gap = min(breach_gap, protected['breach_rate'] - primary['breach_rate'])
        metrics.update({'protected_' + key: value for key, value in primary.items()})
    if protected or preserve_breach:
        success_met = min(success_gap, breach_gap) >= -1e-12
        # Among policies preserving success, prefer the lowest actual collision rate.
        score = [int(passed), int(success_met), -metrics['collision_rate'] if success_met else min(success_gap, breach_gap),
                 metrics['success_rate'], -metrics['collision_rate']]
    else:
        score = [int(passed), metrics['success_rate'], -metrics['collision_rate']]
    return metrics, bool(passed), score


def protected_breach_reference(baseline, episodes_path):
    """Recover the frozen subset's actual breach labels, without new evaluation."""
    protected = baseline.get('protected_reference')
    if not protected or 'breach_rate' in protected:
        return
    seeds = set(protected['seeds'])
    with Path(episodes_path).open(newline='', encoding='utf-8') as file:
        rows = [row for row in csv.DictReader(file) if int(row['seed']) in seeds]
    if len(rows) != len(seeds) or {int(row['seed']) for row in rows} != seeds:
        raise ValueError('Frozen baseline episode data must cover every protected seed exactly once.')
    labels = [row['breach'].strip().lower() for row in rows]
    if any(value not in ('true', 'false', '1', '0', '1.0', '0.0') for value in labels):
        raise ValueError('Expected binary baseline breach labels.')
    protected['breach_rate'] = float(np.mean([value in ('true', '1', '1.0') for value in labels]))


def initialize_collision_minimization(agent, success_floor, breach_budget):
    """Migrate only training objectives/values; preserve the complete acting policy."""
    if agent.settings.minimize_collision:
        raise ValueError('Collision minimization is already initialized; resume without this flag.')
    if not all(np.isfinite(value) and 0 <= value <= 1 for value in (success_floor, breach_budget)):
        raise ValueError('Frozen baseline event rates must be probabilities.')
    previous = agent.critic_optimizer
    changes = dict(minimize_collision=True, breach_constraint=False,
                   success_floor=float(success_floor), breach_budget=float(breach_budget),
                   cost_gae_lambda=1., cost_monte_carlo_targets=True, lagrange_init=0., lagrange_lr=10.,
                   success_lagrange_init=1., breach_lagrange_init=5., event_lagrange_lr=10.)
    for name, value in changes.items():
        setattr(agent.settings, name, value)
    agent.settings.__post_init__()
    agent.config['mappo'].update(changes)
    agent.critic.add_event_heads(success_floor, breach_budget, preserve_breach=True)
    agent.critic_optimizer = torch.optim.Adam(agent.critic.parameters(), **previous.defaults)
    for parameter, state in previous.state.items():
        agent.critic_optimizer.state[parameter] = state
    agent.lagrange = LagrangeMultiplier(0., 10., agent.settings.collision_budget)
    agent.success_lagrange = LagrangeMultiplier(1., 10., 1. - success_floor)
    agent.breach_lagrange = LagrangeMultiplier(5., 10., breach_budget)
    agent.training_state.update(stable_count=0, best_score=[-1., -1., -1.], performance_qualified=False)
    agent.training_state.setdefault('objective_changes', []).append(dict(
        update=agent.updates, objective='minimize_actual_collision', success_floor=success_floor,
        breach_budget=breach_budget, actor_preserved=True, event_advantages='undiscounted_monte_carlo'))


def reset_gate_exploration(agent, standard_deviation):
    """Reset on-policy exploration without changing the deterministic policy."""
    if not np.isfinite(standard_deviation) or not np.exp(-5.) <= standard_deviation <= np.exp(1.):
        raise ValueError('Gate standard deviation must fit the policy log-std bounds.')
    layer = agent.actor.gate_log_std
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.fill_(float(np.log(standard_deviation)))
    for parameter in layer.parameters():
        agent.actor_optimizer.state.pop(parameter, None)
    agent.training_state.setdefault('exploration_changes', []).append(
        dict(update=agent.updates, gate_standard_deviation=standard_deviation))
    agent.training_state['stable_count'] = 0


def restore_proposal_exploration(agent, baseline_state):
    """Reuse the baseline's learned conditional proposal variance, never its Q."""
    layer = agent.actor.proposal_log_std
    layer.load_state_dict({key: baseline_state['log_std_layer.' + key] for key in ('weight', 'bias')})
    for parameter in layer.parameters():
        agent.actor_optimizer.state.pop(parameter, None)
    agent.training_state.setdefault('exploration_changes', []).append(
        dict(update=agent.updates, proposal_variance='baseline_actor'))
    agent.training_state['stable_count'] = 0


def configure_batch(agent, episodes=None, minibatch=None):
    """Change sample counts without changing policy weights or acceptance seeds."""
    for value in (episodes, minibatch):
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
            raise ValueError('Batch sizes must be positive integers.')
    if episodes is None and minibatch is None:
        return
    if episodes is not None:
        agent.config['training']['episodes_per_update'] = episodes
    if minibatch is not None:
        agent.settings.minibatch_size = minibatch
        agent.config['mappo']['minibatch_size'] = minibatch
    agent.training_state.setdefault('batch_changes', []).append(dict(
        update=agent.updates, episodes_per_update=agent.config['training']['episodes_per_update'],
        minibatch_size=agent.settings.minibatch_size))
    agent.training_state['stable_count'] = 0


def initialize_bounded_gate(agent, bound):
    """Limit inherited certainty; the prediction-conditioned mean stays unrestricted."""
    if agent.actor.theta_space_gate:
        raise ValueError('Logit reinitialization requires a sigmoid-normal gate checkpoint.')
    if not np.isfinite(bound) or bound <= 0 or agent.actor.base_gate is None:
        raise ValueError('A positive logit bound and an inherited gate are required.')
    agent.settings.baseline_gate_bound = agent.actor.baseline_gate_bound = float(bound)
    agent.config['mappo']['baseline_gate_bound'] = float(bound)
    with torch.no_grad():
        agent.actor.gate_mean.weight.zero_()
        agent.actor.gate_mean.bias.zero_()
    for parameter in agent.actor.gate_mean.parameters():
        agent.actor_optimizer.state.pop(parameter, None)
    agent.training_state.setdefault('gate_initialization_changes', []).append(
        dict(update=agent.updates, inherited_logit_bound=float(bound), prediction_mean_reset=True))
    agent.training_state['stable_count'] = 0


def initialize_new_gate(agent):
    """Keep the learned proposals, but learn the gate without the SAC prior."""
    if agent.actor.theta_space_gate:
        raise ValueError('Logit reinitialization requires a sigmoid-normal gate checkpoint.')
    if agent.actor.base_gate is None:
        raise ValueError('This initialization requires a transferred actor.')
    agent.settings.baseline_gate_bound = agent.actor.baseline_gate_bound = 0.
    agent.config['mappo']['baseline_gate_bound'] = 0.
    for layer in (agent.actor.base_gate, agent.actor.gate_mean):
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.zero_()
        for parameter in layer.parameters():
            agent.actor_optimizer.state.pop(parameter, None)
    agent.training_state.setdefault('gate_initialization_changes', []).append(
        dict(update=agent.updates, initialization='neutral_gate', proposals_preserved=True))
    agent.training_state['stable_count'] = 0


def initialize_theta_space_gate(agent, standard_deviation=.1):
    """Keep the existing deterministic policy and learn a direct theta residual.

    A latent Gaussian sample is clipped only for execution. PPO continues to
    evaluate that original sample, not the clipped action's density.
    """
    if agent.actor.theta_space_gate:
        raise ValueError('The checkpoint already uses theta-space gates; resume without reinitializing.')
    if not np.isfinite(standard_deviation) or not np.exp(-5.) <= standard_deviation <= np.exp(1.):
        raise ValueError('Gate standard deviation must fit the policy log-std bounds.')
    previous = agent.actor_optimizer
    agent.actor.prior_gate_mean = copy.deepcopy(agent.actor.gate_mean).requires_grad_(False)
    agent.settings.theta_space_gate = agent.actor.theta_space_gate = True
    agent.config['mappo']['theta_space_gate'] = True
    with torch.no_grad():
        agent.actor.gate_mean.weight.zero_()
        agent.actor.gate_mean.bias.zero_()
    for parameter in agent.actor.gate_mean.parameters():
        previous.state.pop(parameter, None)
    reset_gate_exploration(agent, standard_deviation)
    agent.actor_optimizer = agent._actor_optimizer()
    for parameter, state in previous.state.items():
        agent.actor_optimizer.state[parameter] = state
    agent.training_state.setdefault('gate_initialization_changes', []).append(
        dict(update=agent.updates, initialization='theta_space_residual',
             deterministic_policy_preserved=True, latent_std=standard_deviation))


def baseline_reference(config, state):
    """Candidate-only transfer settings must never modify the reference policy."""
    reference_config = copy.deepcopy(config)
    reference_config['mappo']['baseline_gate_bound'] = 0.
    reference_config['mappo']['theta_space_gate'] = False
    reference_config['mappo']['censored_gate_likelihood'] = False
    reference_config['mappo']['breach_constraint'] = False
    reference_config['mappo']['compatibility_prior'] = False
    reference_config['mappo']['compatibility_context_gain'] = False
    reference_config['mappo']['context_gain_only'] = False
    reference_config['mappo']['compatibility_peer_intent'] = False
    reference_config['mappo']['compatibility_mixture_points'] = 0
    reference_config['mappo']['compatibility_joint_mixture'] = False
    reference_config['mappo']['compatibility_task_prediction'] = False
    reference_config['mappo']['compatibility_terminal_prediction'] = False
    reference_config.setdefault('prediction', {})['mixture_points'] = 0
    reference_config['prediction']['task_prediction'] = False
    reference_config['prediction']['terminal_prediction'] = False
    with evaluation_rng():
        reference = PredictiveMAPPO(reference_config, 'cpu')
        reference.actor.initialize_from_baseline(state)
    return reference


def write_evaluation(directory, name, metrics, rows, **metadata):
    directory = Path(directory)
    with (directory / (name + '.csv')).open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / (name + '.json')).write_text(
        json.dumps(dict(**metadata, **metrics), indent=2), encoding='utf-8')


def fit_initial_values(agent, rollout, epochs=20):
    """Fit actual return/event heads without modifying the actor."""
    states, rewards, costs, successes, breaches = [], [], [], [], []
    for episode, summary in zip(rollout.episodes, rollout.summaries):
        states.extend(t['state'] for t in episode)
        rewards.extend(np.cumsum([t['reward'] for t in episode][::-1])[::-1])
        costs.extend(np.cumsum([t['cost'] for t in episode][::-1])[::-1])
        if agent.settings.minimize_collision:
            successes.extend([float(summary['success'])] * len(episode))
        if agent.settings.minimize_collision or agent.settings.breach_constraint:
            breaches.extend([float(summary['breach'])] * len(episode))
    states = agent.tensor(np.asarray(states))
    rewards = agent.tensor(np.asarray(rewards)) / agent.settings.reward_value_scale
    costs = agent.tensor(np.asarray(costs))
    successes, breaches = agent.tensor(np.asarray(successes)), agent.tensor(np.asarray(breaches))
    for _ in range(epochs):
        for index in torch.randperm(len(states), device=agent.device).split(agent.settings.minibatch_size):
            values = agent.critic(states[index], with_events=agent.settings.minimize_collision,
                                 with_breach=agent.settings.breach_constraint)
            r, c = values[:2]
            loss = F.mse_loss(r, rewards[index]) + F.mse_loss(c, costs[index])
            if agent.settings.minimize_collision:
                loss = loss + F.mse_loss(values[2], successes[index]) + F.mse_loss(values[3], breaches[index])
            elif agent.settings.breach_constraint:
                loss = loss + F.mse_loss(values[2], breaches[index])
            agent.critic_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), agent.settings.max_grad_norm,
                                            error_if_nonfinite=True)
            agent.critic_optimizer.step()
    return float(loss.detach())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--breach-constraint', action='store_true', help='Preserve native task reward and add the frozen baseline breach budget')
    parser.add_argument('--breach-budget', type=float,
                        help='Explicitly migrate the constraint to an expanded frozen baseline rate')
    parser.add_argument('--recovery-cost-baseline', action='store_true',
                        help='Use other independent recovery futures as a cost and breach baseline near each restart')
    parser.add_argument('--contextual-prior', action='store_true',
                        help='Learn a scene-dependent prediction gain initialized to preserve the current actor')
    parser.add_argument('--context-only', action='store_true',
                        help='With --contextual-prior, fix existing actor weights and train only the new gain head')
    parser.add_argument('--backtracking-steps', type=int, default=None,
                        help='Bounded full-rollout actor fallback when every minibatch update is rejected')
    parser.add_argument('--collision-budget', type=float, default=None,
                        help='Explicitly change the training event budget in a new run')
    parser.add_argument('--censored-gate', action='store_true', help='Use executed clipped-theta probabilities without changing acting behavior')
    parser.add_argument('--config', type=Path, default=ROOT/'configs/mappo-performance.yaml')
    parser.add_argument('--baseline', type=Path, default=ROOT/'experiments/paper-parameters-seed42-20260921/adares1.pth')
    parser.add_argument('--baseline-validation', type=Path,
                        default=ROOT/'experiments/predictive-mappo-performance-baseline/summary.json')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--train-full-actor', action='store_true',
                        help='Release a context-only checkpoint while preserving all policy weights and Adam moments')
    parser.add_argument('--run-id', default='predictive-mappo-performance-warmstart42')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--environments-per-worker', type=int, default=None,
                        help='Batch independent training worlds inside each worker; deterministic checks stay serial')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-updates', type=int, default=50)
    parser.add_argument('--actor-lr', type=float, default=None,
                        help='Explicit, recorded learning-rate override for a development continuation')
    parser.add_argument('--prediction-lr', type=float, default=None,
                        help='Separate rate and gradient clipping for the new prediction and gate modules')
    parser.add_argument('--prediction-eps', type=float, default=None,
                        help='Adam epsilon for the separate prediction group; preserves all optimizer moments')
    parser.add_argument('--gate-std-lr', type=float, default=None,
                        help='Separate PPO learning rate for the conditional gate variance head')
    parser.add_argument('--gate-std', type=float, default=None,
                        help='Explicit reset of training exploration; deterministic actions stay unchanged')
    parser.add_argument('--baseline-proposal-std', action='store_true',
                        help='Restore the baseline actor learned conditional proposal variance')
    parser.add_argument('--cost-gae-lambda', type=float, default=None,
                        help='Use 1 for exact undiscounted episode-cost targets on complete rollouts')
    parser.add_argument('--cost-monte-carlo-targets', action='store_true',
                        help='Fit cost values to actual undiscounted events independently of actor GAE')
    parser.add_argument('--lagrange-lr', type=float, default=None,
                        help='Explicit dual-step override; the collision budget stays fixed')
    parser.add_argument('--episodes-per-update', type=int, default=None,
                        help='Complete on-policy episodes collected before each update')
    parser.add_argument('--minibatch-size', type=int, default=None,
                        help='Team transitions per optimizer minibatch')
    parser.add_argument('--collision-branches', type=int, default=None,
                        help='Additional current-policy continuations from fresh training collision prefixes')
    parser.add_argument('--breach-branches', type=int, default=None,
                        help='Additional current-policy continuations before early intervention in training breaches')
    parser.add_argument('--breach-cases', type=int,
                        help='Maximum distinct breach prefixes in one recovery batch')
    parser.add_argument('--branch-lookback', type=int, default=None,
                        help='Control steps before a training collision to resume; default 20 (4 seconds)')
    parser.add_argument('--branch-gate-scale', type=float, default=None,
                        help='Known gate standard-deviation multiplier for fresh curriculum continuations only')
    parser.add_argument('--ppo-epochs', type=int, default=None,
                        help='Maximum passes over each on-policy batch; target KL can stop actor updates sooner')
    parser.add_argument('--joint-ratio', action='store_true',
                        help='Clip the complete team proposal/gate probability ratio and monitor joint KL')
    parser.add_argument('--full-batch-guard', action='store_true',
                        help='Retain the best PPO epoch within full-rollout KL; restore actor and Adam on rollback')
    parser.add_argument('--minimize-collision', action='store_true',
                        help='Initialize minimum actual collision objective constrained by frozen baseline success/breach')
    parser.add_argument('--success-floor', type=float,
                        help='With --minimize-collision, explicitly tighten the training success constraint')
    parser.add_argument('--new-gate', action='store_true',
                        help='Keep proposal weights and relearn the gate from a neutral mean')
    parser.add_argument('--theta-space-gate', action='store_true',
                        help='Transfer the current mean exactly and learn a clipped Gaussian theta residual')
    parser.add_argument('--eval-episodes', type=int, default=None,
                        help='Explicit development suite size; must match the supplied baseline manifest')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='Updates between full validation checks, counted from this explicit cadence change')
    parser.add_argument('--audit-episodes', type=int, default=None,
                        help='Fresh paired audit size, independent of development suite size')
    parser.add_argument('--total-steps', type=int, default=None,
                        help='Explicit cumulative training-step limit for this continuation')
    parser.add_argument('--baseline-gate-bound', type=float, default=None,
                        help='Reinitialize the gate with bounded inherited logits and refit both critics')
    parser.add_argument('--stable-checks', type=int, default=3)
    parser.add_argument('--min-success-rate', type=float, default=None)
    parser.add_argument('--max-collision-rate', type=float, default=None)
    parser.add_argument('--audit-seed', type=int, default=30000)
    args = parser.parse_args()
    if args.context_only and not args.contextual_prior:
        parser.error('--context-only requires --contextual-prior; resumed checkpoints retain the setting')
    if args.train_full_actor and args.context_only:
        parser.error('Choose either context-only refinement or full Actor training')
    if args.success_floor is not None and not args.minimize_collision:
        parser.error('--success-floor requires --minimize-collision; resumed checkpoints retain the setting')
    for rate in (args.min_success_rate, args.max_collision_rate, args.collision_budget, args.breach_budget, args.success_floor):
        if rate is not None and not 0 <= rate <= 1:
            parser.error('Performance targets must be finite rates in [0,1].')
    target_success = 0. if args.min_success_rate is None else args.min_success_rate
    target_collision = 1. if args.max_collision_rate is None else args.max_collision_rate
    explicit_targets = args.min_success_rate is not None or args.max_collision_rate is not None
    if min(args.workers, args.max_updates, args.stable_checks) < 1:
        parser.error('workers, max-updates and stable-checks must be positive')
    if args.ppo_epochs is not None and args.ppo_epochs <= 0:
        parser.error('ppo-epochs must be positive')
    if args.environments_per_worker is not None and args.environments_per_worker < 1:
        parser.error('environments-per-worker must be positive')
    if args.collision_branches is not None and args.collision_branches < 0:
        parser.error('collision-branches must be nonnegative')
    if args.breach_branches is not None and args.breach_branches < 0:
        parser.error('breach-branches must be nonnegative')
    if args.breach_cases is not None and args.breach_cases < 1:
        parser.error('breach-cases must be positive')
    if args.branch_lookback is not None and args.branch_lookback < 1:
        parser.error('branch-lookback must be positive')
    if args.branch_gate_scale is not None and (not np.isfinite(args.branch_gate_scale) or args.branch_gate_scale <= 0):
        parser.error('branch-gate-scale must be finite and positive')
    if any(value is not None and value <= 0 for value in
           (args.eval_episodes, args.eval_interval, args.audit_episodes, args.total_steps)):
        parser.error('evaluation sizes and total-steps must be positive')
    if sum((args.new_gate, args.theta_space_gate, args.baseline_gate_bound is not None)) > 1:
        parser.error('Choose one gate initialization per run.')
    if any(value is not None and (not np.isfinite(value) or value <= 0)
           for value in (args.actor_lr, args.prediction_lr, args.gate_std_lr, args.prediction_eps)):
        parser.error('learning-rate overrides must be finite and positive')
    torch.set_num_threads(1)
    set_seed(args.seed)
    baseline = json.loads(args.baseline_validation.read_text(encoding='utf-8'))
    baseline_hash = hashlib.sha256(args.baseline.read_bytes()).hexdigest()
    if baseline['checkpoint_sha256'] != baseline_hash:
        raise ValueError('Baseline checkpoint differs from the validated baseline.')
    if args.resume:
        agent = PredictiveMAPPO.from_checkpoint(args.resume, args.device, restore_rng=True)
        config = copy.deepcopy(agent.config)
    else:
        config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
        agent = PredictiveMAPPO(config, args.device)
        agent.actor.initialize_from_baseline(torch.load(args.baseline, map_location='cpu', weights_only=True))
    if args.train_full_actor:
        agent.config = config
        agent.resume_full_actor_training()
    if any(value is not None for value in (args.actor_lr, args.prediction_lr, args.gate_std_lr, args.prediction_eps)):
        agent.config = config
        agent.set_actor_learning_rates(args.actor_lr, args.prediction_lr, args.gate_std_lr, args.prediction_eps)
        agent.training_state['stable_count'] = 0
        if args.prediction_eps is not None:
            agent.training_state.setdefault('optimization_changes', []).append(
                dict(update=agent.updates, prediction_eps=args.prediction_eps))
    if args.gate_std is not None and not args.theta_space_gate:
        reset_gate_exploration(agent, args.gate_std)
    if args.baseline_proposal_std:
        restore_proposal_exploration(agent, torch.load(args.baseline, map_location='cpu', weights_only=True))
    if args.cost_gae_lambda is not None:
        if not np.isfinite(args.cost_gae_lambda) or not 0 <= args.cost_gae_lambda <= 1:
            parser.error('cost-gae-lambda must be in [0,1]')
        agent.settings.cost_gae_lambda = args.cost_gae_lambda
        config['mappo']['cost_gae_lambda'] = args.cost_gae_lambda
        agent.training_state['stable_count'] = 0
    if args.cost_monte_carlo_targets:
        agent.settings.cost_monte_carlo_targets = True
        config['mappo']['cost_monte_carlo_targets'] = True
        agent.training_state['stable_count'] = 0
    if args.lagrange_lr is not None:
        if not np.isfinite(args.lagrange_lr) or args.lagrange_lr <= 0:
            parser.error('lagrange-lr must be positive')
        agent.settings.lagrange_lr = agent.lagrange.learning_rate = args.lagrange_lr
        config['mappo']['lagrange_lr'] = args.lagrange_lr
        agent.training_state['stable_count'] = 0
    agent.config = config
    if args.joint_ratio:
        agent.settings.joint_ratio = config['mappo']['joint_ratio'] = True
        agent.training_state.setdefault('probability_rule_changes', []).append(
            dict(update=agent.updates, clipping='joint_team_two_stage'))
        agent.training_state['stable_count'] = 0
    if args.full_batch_guard:
        if agent.settings.target_kl <= 0:
            parser.error('full-batch-guard requires a positive configured target_kl')
        agent.settings.full_batch_guard = config['mappo']['full_batch_guard'] = True
        agent.training_state.setdefault('optimization_changes', []).append(
            dict(update=agent.updates, full_batch_guard=True))
        agent.training_state['stable_count'] = 0
    configure_batch(agent, args.episodes_per_update, args.minibatch_size)
    if args.environments_per_worker is not None:
        config['training']['environments_per_worker'] = args.environments_per_worker
    if any(value is not None for value in (args.collision_branches, args.breach_branches, args.branch_lookback, args.branch_gate_scale)):
        for name, value in (('collision_branches', args.collision_branches),
                            ('breach_branches', args.breach_branches),
                            ('branch_lookback', args.branch_lookback), ('branch_gate_scale', args.branch_gate_scale)):
            if value is not None:
                config['training'][name] = value
        agent.training_state['stable_count'] = 0
    if args.breach_cases is not None:
        config['training']['breach_max_cases'] = args.breach_cases
        agent.training_state['stable_count'] = 0
    if args.ppo_epochs is not None:
        agent.settings.ppo_epochs = config['mappo']['ppo_epochs'] = args.ppo_epochs
        agent.training_state.setdefault('optimization_changes', []).append(
            dict(update=agent.updates, ppo_epochs=args.ppo_epochs))
        agent.training_state['stable_count'] = 0
    if args.baseline_gate_bound is not None:
        initialize_bounded_gate(agent, args.baseline_gate_bound)
    if args.new_gate:
        initialize_new_gate(agent)
    if args.theta_space_gate:
        initialize_theta_space_gate(agent, .1 if args.gate_std is None else args.gate_std)
    if args.censored_gate:
        agent.enable_censored_gate_likelihood()
    if args.minimize_collision:
        success_floor = baseline['success_rate'] if args.success_floor is None else args.success_floor
        if success_floor < baseline['success_rate']:
            parser.error('The training success constraint cannot weaken the frozen baseline')
        initialize_collision_minimization(agent, success_floor, baseline['breach_rate'])
    if args.breach_constraint:
        agent.enable_breach_constraint(baseline['breach_rate'])
    if args.breach_budget is not None:
        if not agent.settings.breach_constraint or args.breach_budget != baseline['breach_rate']:
            parser.error('An explicit breach budget must match the frozen baseline and an active breach constraint')
        previous_budget = agent.settings.breach_budget
        agent.settings.breach_budget = config['mappo']['breach_budget'] = args.breach_budget
        agent.breach_lagrange.budget = args.breach_budget
        agent.training_state.update(stable_count=0, performance_qualified=False)
        agent.training_state.setdefault('objective_changes', []).append(dict(
            update=agent.updates, old_breach_budget=previous_budget, breach_budget=args.breach_budget,
            acting_policy_preserved=True))
    if args.recovery_cost_baseline:
        agent.settings.curriculum_cost_baseline = config['mappo']['curriculum_cost_baseline'] = True
        agent.settings.__post_init__()
        agent.training_state.update(stable_count=0, performance_qualified=False)
        agent.training_state.setdefault('optimization_changes', []).append(dict(
            update=agent.updates, curriculum_cost_baseline=True))
    if args.contextual_prior:
        agent.enable_contextual_compatibility(args.context_only)
    if args.backtracking_steps is not None:
        agent.settings.full_batch_backtracking_steps = config['mappo']['full_batch_backtracking_steps'] = args.backtracking_steps
        agent.settings.__post_init__()
        agent.training_state.update(stable_count=0, performance_qualified=False)
        agent.training_state.setdefault('optimization_changes', []).append(dict(
            update=agent.updates, full_batch_backtracking_steps=args.backtracking_steps))
    if args.collision_budget is not None:
        previous_budget = agent.settings.collision_budget
        agent.settings.collision_budget = config['mappo']['collision_budget'] = args.collision_budget
        agent.lagrange.budget = args.collision_budget
        agent.settings.__post_init__()
        if previous_budget != args.collision_budget:
            agent.training_state.update(stable_count=0, performance_qualified=False)
            agent.training_state.setdefault('objective_changes', []).append(dict(
                update=agent.updates, old_collision_budget=previous_budget,
                collision_budget=args.collision_budget, acting_policy_preserved=True))
    # Apply the same breach guard as frozen qualification when a physical
    # safety prior or breach-recovery curriculum changes interception behavior.
    preserve_breach = bool(agent.settings.minimize_collision or agent.settings.breach_constraint or agent.settings.compatibility_prior or
                           config['training'].get('breach_branches', 0))
    if preserve_breach:
        protected_breach_reference(baseline, args.baseline_validation.parent/'episodes.csv')
    if agent.settings.breach_constraint and agent.settings.breach_budget != baseline['breach_rate']:
        raise ValueError('The breach constraint must match the frozen baseline probability budget.')
    if agent.settings.minimize_collision:
        agent.settings.__post_init__()
        if (agent.settings.success_floor < baseline['success_rate'] or
                agent.settings.breach_budget != baseline['breach_rate']):
            raise ValueError('Resumed event constraints must match the frozen baseline.')
    if args.eval_episodes is not None:
        config['training']['eval_episodes'] = args.eval_episodes
    if args.eval_interval is not None:
        config['training']['eval_interval'] = args.eval_interval
        agent.training_state['last_validation_update'] = agent.updates
        agent.training_state['stable_count'] = 0
    if args.total_steps is not None:
        if args.total_steps <= agent.total_steps:
            parser.error('total-steps must exceed the resumed checkpoint step count')
        config['training']['total_steps'] = args.total_steps
        agent.training_state['stable_count'] = 0
    if baseline['protocol'] != config['environment']['protocol'] or baseline['duration'] != config['environment']['total_time']:
        raise ValueError('Baseline and candidate must use the same environment protocol and duration.')
    training = config['training']
    if baseline['agility'] != training['agility'] or baseline['episodes'] != training['eval_episodes']:
        raise ValueError('Baseline and candidate must share agility and validation episode count.')
    validation_seeds = np.asarray(baseline.get('seeds', list(range(
        baseline['first_seed'], baseline['first_seed'] + baseline['episodes']))), dtype=np.int64)
    if len(validation_seeds) != baseline['episodes'] or len(set(validation_seeds)) != len(validation_seeds):
        raise ValueError('The development manifest must contain the declared number of unique seeds.')
    audit_seeds = np.arange(args.audit_seed, args.audit_seed + (args.audit_episodes or baseline['episodes']))
    if set(validation_seeds) & set(audit_seeds):
        raise ValueError('Development validation and audit seeds must not overlap.')
    audit_roots = [ROOT/'experiments', args.baseline_validation.parent.parent, args.baseline.parent.parent]
    if args.resume:
        audit_roots.append(args.resume.parent.parent)
    require_fresh_audit(audit_seeds, audit_roots)
    exp = ExperimentManager(config, str(ROOT/'experiments'), args.run_id)
    directory = Path(exp.exp_dir)
    if args.resume and not (directory/'best-validation.pth').exists():
        # A different run's best score need not belong to the resumed weights.
        agent.training_state['best_score'] = [-1., -1., -1.]
    criteria = dict(baseline_sha256=baseline_hash, baseline_success_rate=baseline['success_rate'],
                    baseline_collision_rate=baseline['collision_rate'], strict_collision_improvement=True,
                    collision_budget=agent.settings.collision_budget, stable_checks=args.stable_checks,
                    validation_first_seed=int(validation_seeds[0]), audit_first_seed=args.audit_seed,
                    episodes=len(validation_seeds), protocol=baseline['protocol'], agility=baseline['agility'],
                    duration=baseline['duration'], purpose='performance_development')
    if 'seeds' in baseline:
        criteria['validation_seeds'] = validation_seeds.tolist()
        criteria['protected_reference'] = baseline.get('protected_reference')
    if len(audit_seeds) != len(validation_seeds):
        criteria['audit_episodes'] = len(audit_seeds)
    if agent.settings.minimize_collision:
        criteria['objective'] = 'minimize_actual_collision'
    elif agent.settings.breach_constraint:
        criteria['objective'] = 'native_task_with_collision_and_breach_constraints'
    if preserve_breach:
        criteria.update(preserve_breach=True, baseline_breach_rate=baseline['breach_rate'])
    criteria_path = directory/'criteria.json'
    if explicit_targets:
        criteria.update(min_success_rate=target_success, max_collision_rate=target_collision,
                        target_scope='aggregate_point_estimate')
    if criteria_path.exists():
        previous = json.loads(criteria_path.read_text())
        # The user explicitly tightened acceptance: a tie in collisions cannot pass.
        previous.setdefault('baseline_collision_rate', baseline['collision_rate'])
        previous.setdefault('strict_collision_improvement', True)
        if previous != criteria:
            raise ValueError('Do not change acceptance criteria silently within an existing run.')
    criteria_path.write_text(json.dumps(criteria, indent=2), encoding='utf-8')
    started = time.monotonic()
    print('[PERFORMANCE] ' + json.dumps(criteria), flush=True)
    qualified = False
    with ParallelRollouts(config, args.workers, training.get('environments_per_worker', 1)) as pool:
        if (not args.resume or args.baseline_gate_bound is not None or args.new_gate or
                args.theta_space_gate or args.minimize_collision or args.breach_constraint):
            seeds = np.random.randint(100000, 2**31-1, training['episodes_per_update'])
            warmup = pool.run(agent, seeds, training['agility'], training['noisy_agility'])
            loss = fit_initial_values(agent, warmup)
            agent.training_state.update(dict(baseline_sha256=baseline_hash, stable_count=0,
                                        value_warmup_steps=warmup.steps + agent.training_state.get('value_warmup_steps', 0), value_warmup_loss=loss,
                                        best_score=[-1., -1., -1.]))
            exp.save_model(agent, training['model_name'])
            print(f'[WARMSTART] episodes={len(warmup.summaries)} steps={warmup.steps} '
                  f'success={summarize(warmup.summaries)["success_rate"]:.3f} '
                  f'collision={warmup.collision_rate:.3f} value_loss={loss:.4f}', flush=True)
            del warmup
            gc.collect()
        while agent.updates < args.max_updates and agent.total_steps < training['total_steps']:
            tick = time.monotonic()
            seeds = np.random.randint(100000, 2**31-1, training['episodes_per_update'])
            rollout = pool.run(agent, seeds, training['agility'], training['noisy_agility'])
            ordinary_rows, ordinary_steps = list(rollout.summaries), rollout.steps
            curriculum_metrics = {}
            if training.get('collision_branches', 0) or training.get('breach_branches', 0):
                cases = collision_branch_cases(rollout, training.get('collision_branches', 0),
                    training.get('branch_lookback', 20), args.seed + 9973 * agent.updates,
                    max_cases=training.get('branch_max_cases'))
                cases += breach_branch_cases(agent, rollout, training.get('breach_branches', 0),
                    args.seed + 9973 * agent.updates + 1, max_cases=training.get('breach_max_cases', 4))
                branches, restored = pool.resume_cases(agent, cases, training['agility'], training['noisy_agility'])
                if branches.summaries:
                    curriculum_metrics = {'curriculum_'+k: v for k, v in summarize(branches.summaries).items()}
                    for kind in ('collision', 'breach'):
                        subset = [s for s in branches.summaries if s['recovery_kind']==kind]
                        if subset:
                            curriculum_metrics.update({f'{kind}_recovery_'+k: v for k, v in summarize(subset).items()})
                    for episode, summary in zip(branches.episodes, branches.summaries):
                        rollout.add_episode(episode, summary)
                curriculum_metrics.update(curriculum_episodes=len(branches.summaries),
                                          curriculum_steps=branches.steps, curriculum_restore_steps=restored)
                agent.training_state['curriculum_steps'] = agent.training_state.get('curriculum_steps', 0) + branches.steps
                agent.training_state['curriculum_restore_steps'] = agent.training_state.get('curriculum_restore_steps', 0) + restored
            rollout_seconds = time.monotonic()-tick
            metrics = agent.update(rollout)
            metrics.update({'train_'+k: v for k, v in summarize(ordinary_rows).items()})
            metrics.update(curriculum_metrics)
            metrics['ordinary_steps'] = ordinary_steps
            metrics['rollout_seconds'] = rollout_seconds
            metrics['elapsed_seconds'] = time.monotonic()-started
            metrics['actor_lr'] = agent.actor_optimizer.param_groups[0]['lr']
            metrics['prediction_lr'] = agent.settings.prediction_lr
            metrics['prediction_eps'] = agent.settings.prediction_eps
            metrics['gate_std_lr'] = agent.settings.gate_std_lr
            metrics['ppo_epochs'] = agent.settings.ppo_epochs
            last_validation = agent.training_state.get('last_validation_update',
                                                       agent.updates - agent.updates % training['eval_interval'])
            check = (agent.updates - last_validation >= training['eval_interval'] or
                     ('last_validation_update' not in agent.training_state and
                      agent.updates % training['eval_interval'] == 0) or
                     agent.updates >= args.max_updates or agent.total_steps >= training['total_steps'])
            if check:
                rows = pool.run(agent, validation_seeds, training['agility'], deterministic=True, retain=False)
                validation, passed, score = validation_result(rows, baseline, agent.settings.collision_budget,
                                                             preserve_breach)
                if explicit_targets:
                    target_passed = meets_targets(validation, target_success, target_collision)
                    passed = passed and target_passed
                    score = [int(passed), *score]
                    validation['target_passed'] = bool(target_passed)
                count = agent.training_state.get('stable_count', 0) + 1 if passed else 0
                agent.training_state['stable_count'] = count
                agent.training_state['last_validation_update'] = agent.updates
                if score > agent.training_state['best_score']:
                    agent.training_state['best_score'] = score
                    exp.save_model(agent, 'best-validation.pth')
                metrics.update({'eval_'+k: v for k, v in validation.items()})
                metrics['validation_passed'] = int(passed)
                metrics['stable_count'] = count
                write_evaluation(directory, 'validation', validation, rows,
                                 performance_passed=passed, stable_count=count, update=agent.updates)
                print(f'[VALIDATION] update={agent.updates} success={validation["success_rate"]:.3f} '
                      f'collision={validation["collision_rate"]:.3f} stable={count}/{args.stable_checks}', flush=True)
            exp.record_metrics(**metrics)
            exp.save_model(agent, training['model_name'])
            print(f'[PERFORMANCE] update={agent.updates} steps={agent.total_steps} '
                  f'train_success={metrics["train_success_rate"]:.3f} collision={rollout.collision_rate:.3f} '
                  f'lambda={agent.lagrange.value:.2f} rollout_s={rollout_seconds:.1f} '
                  f'KL={(metrics["kl_joint"] if agent.settings.joint_ratio else max(metrics["kl_gate"], metrics["kl_proposal"])):.5f}', flush=True)
            if curriculum_metrics:
                print(f'[CURRICULUM] episodes={curriculum_metrics["curriculum_episodes"]} '
                      f'collision={curriculum_metrics.get("curriculum_collision_rate", 0.):.3f} '
                      f'capture={curriculum_metrics.get("curriculum_capture_rate", 0.):.3f} '
                      f'learning_steps={curriculum_metrics["curriculum_steps"]} '
                      f'prefix_replay_steps={curriculum_metrics["curriculum_restore_steps"]}', flush=True)
            # Do not retain the preceding large rollout while workers collect
            # its successor (or while independent validation is running).
            del rollout, ordinary_rows
            if training.get('collision_branches', 0) or training.get('breach_branches', 0):
                del branches, cases
                if curriculum_metrics['curriculum_episodes']:
                    del episode, summary
            gc.collect()
            if check and agent.training_state['stable_count'] >= args.stable_checks:
                # Fresh paired audit. This baseline actor has identical deterministic actions.
                require_fresh_audit(audit_seeds, audit_roots)
                base = baseline_reference(config, torch.load(args.baseline, map_location='cpu', weights_only=True))
                base_rows = pool.run(base, audit_seeds, training['agility'], deterministic=True, retain=False)
                candidate_rows = pool.run(agent, audit_seeds, training['agility'], deterministic=True, retain=False)
                base_metrics, candidate_metrics = summarize(base_rows), summarize(candidate_rows)
                qualified = qualifies(candidate_metrics, base_metrics['success_rate'], agent.settings.collision_budget,
                                      base_metrics['collision_rate'],
                                      base_metrics['breach_rate'] if preserve_breach else None)
                qualified = qualified and meets_targets(candidate_metrics, target_success, target_collision)
                write_evaluation(directory, 'audit-baseline', base_metrics, base_rows,
                                 checkpoint_sha256=baseline_hash, first_seed=args.audit_seed, episodes=len(audit_seeds))
                write_evaluation(directory, 'audit-candidate', candidate_metrics, candidate_rows,
                                 performance_passed=qualified, first_seed=args.audit_seed,
                                 episodes=len(audit_seeds), update=agent.updates,
                                 baseline_success_rate=base_metrics['success_rate'],
                                 baseline_collision_rate=base_metrics['collision_rate'])
                if qualified:
                    agent.training_state['performance_qualified'] = True
                    exp.save_model(agent, 'qualified.pth')
                print('[AUDIT] ' + json.dumps(dict(performance_passed=qualified,
                      baseline=base_metrics, candidate=candidate_metrics)), flush=True)
                break
    (directory/'qualification.json').write_text(json.dumps(dict(
        performance_passed=qualified, updates=agent.updates, steps=agent.total_steps,
        stable_count=agent.training_state['stable_count'], criteria=criteria,
        formal_experiments_started=False), indent=2), encoding='utf-8')
    print(f'[COMPLETE] performance_passed={qualified} formal_experiments_started=False', flush=True)


if __name__ == '__main__':
    main()
