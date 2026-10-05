"""Two-stage MAPPO with explicit clipping rules and a collision multiplier."""
from dataclasses import dataclass, asdict
from pathlib import Path
import copy
import random
import numpy as np
import torch
from torch.nn import functional as F

from policy.interaction_prediction import PredictionConfig, InteractionPredictor, exchange_candidates
from policy.mappo_networks import PredictiveActor, CentralizedCritic


@dataclass
class MAPPOConfig:
    hidden_dim: int = 512
    relation_dim: int = 64
    coordination_dim: int = 16
    actor_lr: float = 1e-4
    critic_lr: float = 1e-4
    ppo_epochs: int = 5
    minibatch_size: int = 128
    clip_ratio: float = .2
    entropy_coef: float = .001
    max_grad_norm: float = .5
    reward_gamma: float = 1.
    cost_gamma: float = 1.
    gae_lambda: float = .95
    cost_gae_lambda: float = .95
    cost_monte_carlo_targets: bool = False
    collision_budget: float = .05
    lagrange_init: float = 1.
    lagrange_lr: float = 1.
    baseline_initialization: bool = False
    baseline_gate_bound: float = 0.
    theta_space_gate: bool = False
    censored_gate_likelihood: bool = False
    compatibility_prior: bool = False
    compatibility_gain: float = 1.
    compatibility_margin: float = 7.
    compatibility_temperature: float = 1.
    compatibility_neighbor_temperature: float = 2.
    compatibility_compact_risk: bool = False
    compatibility_task_priority: bool = False
    compatibility_priority_temperature: float = 5.
    compatibility_peer_intent: bool = False
    compatibility_mixture_points: int = 0
    compatibility_joint_mixture: bool = False
    compatibility_task_prediction: bool = False
    compatibility_terminal_prediction: bool = False
    compatibility_task_penalty: float = 1.
    compatibility_context_gain: bool = False
    fixed_compatibility_gain: bool = False
    prediction_disabled: bool = False
    context_gain_only: bool = False
    reward_value_scale: float = 1.
    target_kl: float = 0.
    prediction_lr: float = 0.
    prediction_eps: float = 1e-5
    gate_std_lr: float = 0.
    joint_ratio: bool = False
    full_batch_guard: bool = False
    full_batch_backtracking_steps: int = 0
    guard_task_surrogate: bool = False
    curriculum_cost_baseline: bool = False
    curriculum_reward_baseline: bool = False
    curriculum_baseline_steps: int = 20
    minimize_collision: bool = False
    breach_constraint: bool = False
    success_floor: float = 0.
    breach_budget: float = 1.
    success_lagrange_init: float = 1.
    breach_lagrange_init: float = 5.
    event_lagrange_lr: float = 10.

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError('All MAPPO settings must be finite.')
        for name in ('hidden_dim', 'relation_dim', 'coordination_dim', 'ppo_epochs', 'minibatch_size'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer.')
        if min(self.actor_lr, self.critic_lr, self.max_grad_norm, self.lagrange_lr) <= 0:
            raise ValueError('Learning rates and gradient bound must be positive.')
        if not 0 < self.clip_ratio < 1 or not 0 < self.reward_gamma <= 1:
            raise ValueError('Invalid PPO clip ratio or reward discount.')
        if self.cost_gamma != 1.:
            raise ValueError('Binary episode collision probability requires cost_gamma=1.')
        if not 0 <= self.gae_lambda <= 1 or not 0 <= self.cost_gae_lambda <= 1 or not 0 <= self.collision_budget <= 1:
            raise ValueError('GAE lambda and collision budget must be in [0,1].')
        if self.lagrange_init < 0 or self.entropy_coef < 0:
            raise ValueError('Multiplier and entropy coefficient cannot be negative.')
        if self.reward_value_scale <= 0 or min(self.target_kl, self.prediction_lr, self.gate_std_lr) < 0:
            raise ValueError('Value scale must be positive; target KL and prediction LR nonnegative.')
        if self.prediction_eps <= 0:
            raise ValueError('Prediction optimizer epsilon must be positive.')
        if self.full_batch_guard and self.target_kl <= 0:
            raise ValueError('A full-batch policy guard requires a positive target KL.')
        if (not isinstance(self.full_batch_backtracking_steps, int) or
                not 0 <= self.full_batch_backtracking_steps <= 12 or
                (self.full_batch_backtracking_steps and not self.full_batch_guard)):
            raise ValueError('Use at most twelve backtracking trials with the full-batch guard.')
        if self.guard_task_surrogate and not self.full_batch_guard:
            raise ValueError('Task preservation requires the full-batch policy guard.')
        if not isinstance(self.curriculum_baseline_steps, int) or self.curriculum_baseline_steps < 1:
            raise ValueError('The curriculum baseline window must be a positive integer.')
        if self.curriculum_cost_baseline and (self.cost_gae_lambda != 1. or not self.cost_monte_carlo_targets):
            raise ValueError('Independent recovery baselines require undiscounted Monte Carlo cost advantages.')
        if self.curriculum_reward_baseline and (self.reward_gamma != 1. or self.gae_lambda != 1.):
            raise ValueError('Independent task baselines require undiscounted Monte Carlo task advantages.')
        if not 0 <= self.success_floor <= 1 or not 0 <= self.breach_budget <= 1:
            raise ValueError('Event probability bounds must be in [0,1].')
        if min(self.success_lagrange_init, self.breach_lagrange_init) < 0 or self.event_lagrange_lr <= 0:
            raise ValueError('Event multipliers must be nonnegative, with a positive update rate.')
        if self.minimize_collision and (self.cost_gae_lambda != 1. or not self.cost_monte_carlo_targets):
            raise ValueError('Collision minimization requires actual undiscounted Monte Carlo cost advantages.')
        if self.breach_constraint and self.minimize_collision:
            raise ValueError('Native task reward with a breach constraint is a separate objective from collision minimization.')
        if self.baseline_gate_bound < 0 or (self.baseline_gate_bound and not self.baseline_initialization):
            raise ValueError('A bounded inherited gate requires baseline initialization and a nonnegative bound.')
        if self.compatibility_prior and not self.theta_space_gate:
            raise ValueError('The compatibility prior is expressed in physical theta units.')
        if self.compatibility_context_gain and not self.compatibility_prior:
            raise ValueError('Contextual compatibility requires the physical prior.')
        if self.fixed_compatibility_gain and (not self.compatibility_prior or self.compatibility_context_gain):
            raise ValueError('A fixed gain requires the prior and excludes a learned context gain.')
        if self.prediction_disabled and (self.compatibility_prior or self.compatibility_peer_intent):
            raise ValueError('The no-prediction ablation cannot use a predictive prior or peer intent.')
        if self.context_gain_only and not self.compatibility_context_gain:
            raise ValueError('Context-only optimization requires the contextual gain head.')
        if self.censored_gate_likelihood and not self.theta_space_gate:
            raise ValueError('Censored gate probabilities require theta-space clipping.')
        if self.compatibility_mixture_points not in (0, 3, 5, 9):
            raise ValueError('Mixed compatibility uses a three, five or nine point grid.')
        if self.compatibility_mixture_points and not (self.compatibility_prior and self.compatibility_peer_intent):
            raise ValueError('Mixed compatibility requires the prior and exchanged peer intent context.')
        if self.compatibility_joint_mixture and not self.compatibility_mixture_points:
            raise ValueError('Joint mixed-action planning requires a physical prediction grid.')
        if (self.compatibility_task_prediction or self.compatibility_terminal_prediction) and not self.compatibility_joint_mixture:
            raise ValueError('Predictive task preservation requires joint mixed-action planning.')
        if self.compatibility_task_penalty <= 0:
            raise ValueError('The nominal task preservation penalty must be positive.')
        if (not 0 < self.compatibility_gain < 100 or
                min(self.compatibility_margin, self.compatibility_temperature, self.compatibility_neighbor_temperature,
                    self.compatibility_priority_temperature) <= 0):
            raise ValueError('Compatibility prior scales must be positive, with gain in (0,100).')


class LagrangeMultiplier:
    def __init__(self, value, learning_rate, budget):
        self.value, self.learning_rate, self.budget = float(value), float(learning_rate), float(budget)

    def update(self, episode_collision_rate):
        if not np.isfinite(episode_collision_rate) or not 0 <= episode_collision_rate <= 1:
            raise ValueError('Expected the binary collision rate of complete episodes.')
        self.value = max(0., self.value + self.learning_rate * (episode_collision_rate - self.budget))
        return self.value


def clipped_policy_loss(delta_l, delta_g, advantage, clip_ratio, joint=False):
    """Score-function PPO loss on the saved joint proposal/gate samples.

    Divide the joint surrogate by boat count to match the existing factor
    objective's gradient scale at the behavior policy. No action is resampled.
    """
    adv = advantage[:, None]
    def surrogate(ratio):
        return torch.minimum(ratio * adv, ratio.clamp(1-clip_ratio, 1+clip_ratio) * adv)
    if joint:
        ratio = (delta_l + delta_g).sum(-1, keepdim=True).exp()
        return -surrogate(ratio).mean() / delta_l.shape[-1]
    return -(surrogate(delta_l.exp()) + surrogate(delta_g.exp())).mean()


class PredictiveMAPPO:
    algorithm = 'predictive-mappo'
    format_version = 1

    def __init__(self, config, device='cpu'):
        self.config = config
        self.settings = MAPPOConfig(**config.get('mappo', {}))
        self.prediction_config = PredictionConfig(**config.get('prediction', {}))
        if self.prediction_config.mixture_points != self.settings.compatibility_mixture_points:
            raise ValueError('Actor and physical predictor must use the same mixture grid.')
        if self.prediction_config.task_prediction != self.settings.compatibility_task_prediction:
            raise ValueError('Actor and predictor must agree on task prediction features.')
        if self.prediction_config.terminal_prediction != self.settings.compatibility_terminal_prediction:
            raise ValueError('Actor and predictor must agree on terminal prediction features.')
        self.device = torch.device(device)
        self.defender_num = int(config['agent']['defender_num'])
        if self.defender_num < 2:
            raise ValueError('At least two defenders are required.')
        s = self.settings
        self.actor = PredictiveActor(s.hidden_dim, s.relation_dim, s.coordination_dim,
                                     baseline_initialization=s.baseline_initialization,
                                     baseline_gate_bound=s.baseline_gate_bound,
                                     theta_space_gate=s.theta_space_gate, compatibility_prior=s.compatibility_prior,
                                     compatibility_gain=s.compatibility_gain, compatibility_margin=s.compatibility_margin,
                                     compatibility_temperature=s.compatibility_temperature,
                                     compatibility_neighbor_temperature=s.compatibility_neighbor_temperature,
                                     prediction_distance_scale=self.prediction_config.distance_scale,
                                     compatibility_compact_risk=s.compatibility_compact_risk,
                                     compatibility_task_priority=s.compatibility_task_priority,
                                     compatibility_priority_temperature=s.compatibility_priority_temperature,
                                     compatibility_peer_intent=s.compatibility_peer_intent,
                                     compatibility_mixture_points=s.compatibility_mixture_points,
                                     compatibility_joint_mixture=s.compatibility_joint_mixture,
                                     compatibility_task_prediction=s.compatibility_task_prediction,
                                     compatibility_task_penalty=s.compatibility_task_penalty,
                                     compatibility_context_gain=s.compatibility_context_gain,
                                     compatibility_terminal_prediction=s.compatibility_terminal_prediction,
                                     prediction_horizon=self.prediction_config.horizon,
                                     prediction_terminal_buffer=self.prediction_config.terminal_buffer,
                                     censored_gate_likelihood=s.censored_gate_likelihood).to(self.device)
        self.critic = CentralizedCritic(7 * (self.defender_num + 1) + 2, s.hidden_dim,
            s.minimize_collision, s.success_floor, s.breach_budget, s.breach_constraint).to(self.device)
        if s.fixed_compatibility_gain:
            self.actor.compatibility_gain_raw.requires_grad_(False)
        self.actor_optimizer = self._actor_optimizer()
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=s.critic_lr, eps=1e-5)
        self.predictor = InteractionPredictor(self.prediction_config)
        self.lagrange = LagrangeMultiplier(s.lagrange_init, s.lagrange_lr, s.collision_budget)
        self.success_lagrange = LagrangeMultiplier(s.success_lagrange_init, s.event_lagrange_lr, 1. - s.success_floor)
        self.breach_lagrange = LagrangeMultiplier(s.breach_lagrange_init, s.event_lagrange_lr, s.breach_budget)
        self.total_steps = 0
        self.updates = 0
        self.training_state = {}

    def _actor_optimizer(self):
        cfg = self.settings
        if cfg.context_gain_only:
            context = list(self.actor.compatibility_context_head.parameters())
            identifiers = {id(parameter) for parameter in context}
            fixed = [parameter for parameter in self.actor.parameters()
                     if parameter.requires_grad and id(parameter) not in identifiers]
            # Keep old parameters and their Adam state in the checkpoint; a
            # zero learning rate fixes their weights during this refinement.
            return torch.optim.Adam([dict(params=fixed, lr=0., name='fixed'),
                dict(params=context, lr=cfg.prediction_lr or cfg.actor_lr, name='context',
                     eps=cfg.prediction_eps)], eps=1e-5)
        if not cfg.prediction_lr and not cfg.gate_std_lr:
            return torch.optim.Adam((p for p in self.actor.parameters() if p.requires_grad),
                                    lr=cfg.actor_lr, eps=1e-5)
        prediction_names = ('relation_encoder.', 'relation_query.', 'priority_head.',
                            'coordination_encoder.', 'adapter.', 'gate_mean.', 'compatibility_gain_raw',
                            'compatibility_context_head.')
        base, prediction, variance = [], [], []
        for name, parameter in self.actor.named_parameters():
            if not parameter.requires_grad:
                continue
            if cfg.gate_std_lr and name.startswith('gate_log_std.'):
                variance.append(parameter)
            else:
                (prediction if cfg.prediction_lr and name.startswith(prediction_names) else base).append(parameter)
        groups = [dict(params=base, lr=cfg.actor_lr, name='base')]
        if prediction:
            groups.append(dict(params=prediction, lr=cfg.prediction_lr, eps=cfg.prediction_eps, name='prediction'))
        if variance:
            groups.append(dict(params=variance, lr=cfg.gate_std_lr, name='gate_variance'))
        return torch.optim.Adam(groups, eps=1e-5)

    def enable_compatibility_prior(self, gain=1., compact_risk=True, task_priority=True, peer_intent=False):
        """Add the predictive policy feature without resetting any learned weights."""
        previous = self.actor_optimizer
        self.actor.enable_compatibility_prior(gain, compact_risk, task_priority, peer_intent)
        self.settings.compatibility_prior, self.settings.compatibility_gain = True, float(gain)
        self.settings.compatibility_compact_risk = bool(compact_risk)
        self.settings.compatibility_task_priority = bool(task_priority)
        self.settings.compatibility_peer_intent = bool(peer_intent)
        self.settings.__post_init__()
        self.config['mappo'].update(compatibility_prior=True, compatibility_gain=float(gain),
            compatibility_margin=self.settings.compatibility_margin,
            compatibility_temperature=self.settings.compatibility_temperature,
            compatibility_neighbor_temperature=self.settings.compatibility_neighbor_temperature,
            compatibility_compact_risk=bool(compact_risk), compatibility_task_priority=bool(task_priority),
            compatibility_priority_temperature=self.settings.compatibility_priority_temperature,
            compatibility_peer_intent=bool(peer_intent))
        self.actor_optimizer = self._actor_optimizer()
        for parameter, state in previous.state.items():
            self.actor_optimizer.state[parameter] = state
        self.training_state.update(stable_count=0, performance_qualified=False)
        self.training_state.setdefault('policy_structure_changes', []).append(dict(
            update=self.updates, feature='candidate_compatibility_prior', initial_gain=float(gain),
            compact_risk=bool(compact_risk), task_priority=bool(task_priority),
            peer_intent=bool(peer_intent),
            priority_temperature_metres=self.settings.compatibility_priority_temperature,
            margin_metres=self.settings.compatibility_margin, temperature_metres=self.settings.compatibility_temperature,
            neighbor_temperature_metres=self.settings.compatibility_neighbor_temperature))

    def enable_contextual_compatibility(self, context_only=False):
        """Learn scene-dependent intervention strength, preserving all prior Adam state."""
        previous = self.actor_optimizer
        self.actor.enable_contextual_compatibility()
        self.settings.compatibility_context_gain = True
        self.config['mappo']['compatibility_context_gain'] = True
        self.settings.context_gain_only = self.config['mappo']['context_gain_only'] = bool(context_only)
        self.settings.__post_init__()
        self.actor_optimizer = self._actor_optimizer()
        for parameter, state in previous.state.items():
            self.actor_optimizer.state[parameter] = state
        self.training_state.update(stable_count=0, performance_qualified=False)
        self.training_state.setdefault('policy_structure_changes', []).append(dict(
            update=self.updates, feature='contextual_compatibility_gain',
            initial_factor=1., context_gain_only=bool(context_only), acting_policy_preserved=True))

    def resume_full_actor_training(self):
        """Release context-only refinement without resetting the acting policy or Adam."""
        if not self.settings.context_gain_only:
            raise ValueError('The checkpoint is not restricted to context-only training.')
        previous = self.actor_optimizer
        self.settings.context_gain_only = self.config['mappo']['context_gain_only'] = False
        self.actor_optimizer = self._actor_optimizer()
        for parameter, state in previous.state.items():
            self.actor_optimizer.state[parameter] = state
        self.training_state.update(stable_count=0, performance_qualified=False)
        self.training_state.setdefault('optimization_changes', []).append(dict(
            update=self.updates, context_gain_only=False, acting_policy_preserved=True))

    def enable_mixture_compatibility(self, points=5, joint=False, task_prediction=False, terminal_prediction=False):
        """Refine the existing physical prior; require fresh on-policy samples."""
        if points not in (3, 5, 9) or not self.settings.compatibility_prior or ((task_prediction or terminal_prediction) and not joint):
            raise ValueError('Enable the compatibility prior before selecting a mixed-action grid.')
        self.settings.compatibility_mixture_points = self.actor.compatibility_mixture_points = points
        self.settings.compatibility_joint_mixture = self.actor.compatibility_joint_mixture = bool(joint)
        self.settings.compatibility_task_prediction = self.actor.compatibility_task_prediction = bool(task_prediction)
        self.settings.compatibility_terminal_prediction = self.actor.compatibility_terminal_prediction = bool(terminal_prediction)
        self.settings.compatibility_peer_intent = self.actor.compatibility_peer_intent = True
        self.config['mappo'].update(compatibility_mixture_points=points, compatibility_peer_intent=True,
                                    compatibility_joint_mixture=bool(joint),
                                    compatibility_task_prediction=bool(task_prediction),
                                    compatibility_terminal_prediction=bool(terminal_prediction))
        self.config.setdefault('prediction', {})['mixture_points'] = points
        self.config['prediction']['task_prediction'] = bool(task_prediction)
        self.config['prediction']['terminal_prediction'] = bool(terminal_prediction)
        self.prediction_config = PredictionConfig(**self.config['prediction'])
        self.predictor = InteractionPredictor(self.prediction_config)
        self.actor.prediction_horizon = self.prediction_config.horizon
        self.actor.prediction_terminal_buffer = self.prediction_config.terminal_buffer
        self.settings.__post_init__()
        self.training_state.update(stable_count=0, performance_qualified=False)
        self.training_state.setdefault('policy_structure_changes', []).append(dict(
            update=self.updates, feature='mixed_action_compatibility', points=points,
            peer_intent=True, joint=bool(joint), task_prediction=bool(task_prediction),
            terminal_prediction=bool(terminal_prediction), extra_optimizer_updates=0))

    def enable_breach_constraint(self, budget, initial_multiplier=None):
        """Add a real-event breach constraint without replacing native task reward."""
        if self.settings.minimize_collision or self.settings.breach_constraint:
            raise ValueError('Initialize the native-task breach constraint only once.')
        value = self.lagrange.value if initial_multiplier is None else float(initial_multiplier)
        if not np.isfinite(budget) or not 0 <= budget <= 1 or not np.isfinite(value) or value < 0:
            raise ValueError('Use a probability budget and a finite nonnegative multiplier.')
        previous = self.critic_optimizer
        self.critic.add_breach_head(budget)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), **previous.defaults)
        for parameter, state in previous.state.items():
            self.critic_optimizer.state[parameter] = state
        changes = dict(breach_constraint=True, breach_budget=float(budget), breach_lagrange_init=value,
                       event_lagrange_lr=self.lagrange.learning_rate)
        for name, setting in changes.items():
            setattr(self.settings, name, setting)
        self.config['mappo'].update(changes)
        self.breach_lagrange = LagrangeMultiplier(value, self.settings.event_lagrange_lr, budget)
        self.training_state.update(stable_count=0, performance_qualified=False)
        self.training_state.setdefault('objective_changes', []).append(dict(update=self.updates,
            objective='native_task_with_collision_and_breach_constraints', breach_budget=float(budget),
            initial_breach_multiplier=value, acting_policy_preserved=True))

    @torch.no_grad()
    def calibrate_compatibility_prior(self, rollout, target_kl=None):
        """Initialize the physical residual within the existing Gaussian KL budget.

        Proposal and gate standard deviations do not change. Conditional gate
        KL is exactly half the squared, standardized mean displacement; summing
        boats gives team KL. The latent KL also bounds KL after theta clipping.
        Calibration samples are not reused for a subsequent PPO update.
        """
        if self.settings.compatibility_prior or not self.settings.theta_space_gate:
            raise ValueError('Calibrate once from a theta-space policy without the prior.')
        budget = self.settings.target_kl if target_kl is None else float(target_kl)
        if not np.isfinite(budget) or budget <= 0:
            raise ValueError('Use a finite, positive policy KL budget.')
        if any(s.get('curriculum', False) or s.get('seed', 0) < 100000 for s in rollout.summaries):
            raise ValueError('Calibration requires ordinary training scenarios, never evaluation or recovery cases.')
        data = rollout.tensors(self.settings, self.device)
        if not all(torch.isfinite(value).all() for value in data.values()):
            raise ValueError('Calibration samples must be finite.')
        if 'gate_scale' in data and not (data['gate_scale'] == 1.).all():
            raise ValueError('Calibrate at the native exploration scale.')
        # Verify the input behavior before changing the policy structure.
        for start in range(0, rollout.steps, self.settings.minibatch_size):
            batch = {key: value[start:start+self.settings.minibatch_size] for key, value in data.items()}
            lp, lg, _ = self.evaluate_batch(batch)
            if not torch.isfinite(lp).all() or not torch.isfinite(lg).all() or max(float((lp-batch['old_logp_l']).abs().max()),
                   float((lg-batch['old_logp_g']).abs().max())) > 1e-3:
                raise ValueError('Collect fresh calibration samples from the unchanged source policy.')
        self.enable_compatibility_prior()
        total = 0.
        for start in range(0, rollout.steps, self.settings.minibatch_size):
            batch = {key: value[start:start+self.settings.minibatch_size] for key, value in data.items()}
            distribution, _ = self.actor.gate_distribution(
                batch['obs'], batch['proposals'], batch['boids'], batch['edges'], batch['mask'])
            signal = self.actor.compatibility_signal(batch['edges'], batch['mask'], batch['obs'])
            total += float((.5*(signal/distribution.stddev).square()).sum(dtype=torch.float64))
        unit_kl = total/rollout.steps
        gain = min(1., np.sqrt(budget/unit_kl)) if unit_kl > 0. else 1.
        self.actor.compatibility_gain_raw.fill_(float(np.log(np.expm1(gain))))
        self.settings.compatibility_gain = self.config['mappo']['compatibility_gain'] = float(gain)
        result = dict(gain=float(gain), unit_gain_joint_kl=unit_kl, initialized_joint_kl=unit_kl*gain**2,
                      target_joint_kl=budget, calibration_steps=rollout.steps,
                      calibration_episodes=len(rollout.summaries))
        self.training_state['policy_structure_changes'][-1].update(initial_gain=float(gain), **result)
        return result

    def enable_censored_gate_likelihood(self):
        """Preserve acting behavior; optimize the probability of actual clipped theta."""
        if not self.settings.theta_space_gate or self.settings.censored_gate_likelihood:
            raise ValueError('Enable censored likelihoods once on a theta-space policy.')
        self.settings.censored_gate_likelihood = self.actor.censored_gate_likelihood = True
        self.config['mappo']['censored_gate_likelihood'] = True
        self.training_state.update(stable_count=0, performance_qualified=False)
        self.training_state.setdefault('probability_rule_changes', []).append(dict(
            update=self.updates, gate_probability='clipped_normal_mixed_measure', acting_policy_preserved=True))

    def set_actor_learning_rates(self, actor_lr=None, prediction_lr=None, gate_std_lr=None, prediction_eps=None):
        """Change development optimizer settings, preserving each parameter's Adam state."""
        if prediction_eps is not None and not (self.settings.prediction_lr if prediction_lr is None else prediction_lr):
            raise ValueError('Prediction epsilon requires a separate prediction parameter group.')
        for name, value in (('actor_lr', actor_lr), ('prediction_lr', prediction_lr),
                            ('gate_std_lr', gate_std_lr), ('prediction_eps', prediction_eps)):
            if value is not None:
                if not np.isfinite(value) or value <= 0:
                    raise ValueError('Learning-rate overrides must be finite and positive.')
                setattr(self.settings, name, value)
                self.config['mappo'][name] = value
        previous = self.actor_optimizer
        self.actor_optimizer = self._actor_optimizer()
        for parameter, state in previous.state.items():
            self.actor_optimizer.state[parameter] = state

    def tensor(self, value, boolean=False):
        return torch.as_tensor(value, dtype=torch.bool if boolean else torch.float32, device=self.device)

    @torch.no_grad()
    def act(self, observation, states, boids, timestamp, deterministic=False, gate_scale=1.):
        """The only deployment path: propose -> exchange -> predict -> gate."""
        observation = np.asarray(observation, dtype=np.float32)
        n = self.defender_num
        if observation.shape != (n, 14 + 2 * (n - 1)) or not np.isfinite(observation).all():
            raise ValueError('Wrong observation shape or non-finite observation.')
        obs = self.tensor(observation).unsqueeze(0)
        h = self.actor.encode(obs)
        proposal, raw, logp_l = self.actor.sample_proposals(obs, deterministic, encoded=h)
        proposals = proposal[0].cpu().numpy()
        messages = exchange_candidates(states, boids, proposals, timestamp,
            actor_observations=observation if self.settings.compatibility_peer_intent else None)
        if len(messages.states) != n:
            raise ValueError('Message count must match the checkpoint defender count.')
        if self.settings.prediction_disabled:
            edges, mask = np.zeros((n, n, 18), dtype=np.float32), np.zeros((n, n), dtype=bool)
        else:
            edges, mask = self.predictor.features(messages)
        gate_obs = self.tensor(np.array(messages.actor_observations)).unsqueeze(0) if messages.actor_observations is not None else obs
        gates, gate_raw, logp_g, _ = self.actor.sample_gates(
            gate_obs, proposal, self.tensor(np.array(messages.boids)).unsqueeze(0),
            self.tensor(edges).unsqueeze(0), self.tensor(mask, True).unsqueeze(0), deterministic,
            encoded=h, gate_scale=gate_scale)
        action = np.concatenate((proposals, gates[0].cpu().numpy()), axis=-1)
        record = dict(obs=observation.copy(), proposals=proposals.copy(), boids=np.array(messages.boids),
                      edges=edges, mask=mask, proposal_raw=raw[0].cpu().numpy(),
                      gate_raw=gate_raw[0].cpu().numpy(), old_logp_l=logp_l[0].cpu().numpy(),
                      old_logp_g=logp_g[0].cpu().numpy(), message_states=np.array(messages.states),
                      timestamp=float(messages.timestamps[0]), gate_scale=float(gate_scale),
                      gate_likelihood_censored=self.settings.censored_gate_likelihood)
        if messages.actor_observations is not None:
            record['message_observations'] = np.array(messages.actor_observations)
        if not np.isfinite(action).all():
            raise FloatingPointError('Non-finite action.')
        return action, record

    @torch.no_grad()
    def act_batch(self, observations, states, boids, timestamps, generators, deterministic=False):
        """Collect independent worlds in one forward pass, with one RNG per world."""
        observations = np.asarray(observations, dtype=np.float32)
        n = self.defender_num
        if (observations.ndim != 3 or observations.shape[1:] != (n, 14 + 2 * (n - 1)) or
                not np.isfinite(observations).all()):
            raise ValueError('Invalid batch of observations.')
        count = len(observations)
        if not count or any(len(value) != count for value in (states, boids, timestamps, generators)):
            raise ValueError('Every world needs matching states, candidates, time and RNG.')
        obs = self.tensor(observations)
        h = self.actor.encode(obs)
        proposal, raw, logp_l = self.actor.sample_proposals(obs, deterministic, encoded=h, generators=generators)
        proposals = proposal.cpu().numpy()
        messages = [exchange_candidates(state, boid, proposed, timestamp,
                    actor_observations=observations[i] if self.settings.compatibility_peer_intent else None)
                    for i, (state, boid, proposed, timestamp) in enumerate(zip(states, boids, proposals, timestamps))]
        if any(len(message.states) != n for message in messages):
            raise ValueError('Message counts must match the checkpoint.')
        if self.settings.prediction_disabled:
            edges = np.zeros((count, n, n, 18), dtype=np.float32)
            mask = np.zeros((count, n, n), dtype=bool)
        else:
            edges, mask = self.predictor.features_batch(messages)
        boid_batch = np.stack([message.boids for message in messages])
        gate_obs = self.tensor(np.stack([message.actor_observations for message in messages])) if self.settings.compatibility_peer_intent else obs
        gates, gate_raw, logp_g, _ = self.actor.sample_gates(
            gate_obs, proposal, self.tensor(boid_batch), self.tensor(edges), self.tensor(mask, True),
            deterministic, encoded=h, generators=generators)
        actions = np.concatenate((proposals, gates.cpu().numpy()), axis=-1)
        if not np.isfinite(actions).all():
            raise FloatingPointError('Non-finite batched action.')
        raw, gate_raw, logp_l, logp_g = [value.cpu().numpy() for value in (raw, gate_raw, logp_l, logp_g)]
        records = [dict(obs=observations[i].copy(), proposals=proposals[i].copy(), boids=boid_batch[i].copy(),
                        edges=edges[i].copy(), mask=mask[i].copy(), proposal_raw=raw[i].copy(),
                        gate_raw=gate_raw[i].copy(), old_logp_l=logp_l[i].copy(), old_logp_g=logp_g[i].copy(),
                        message_states=np.array(message.states), timestamp=float(message.timestamps[0]), gate_scale=1.,
                        gate_likelihood_censored=self.settings.censored_gate_likelihood)
                   for i, message in enumerate(messages)]
        for record, message in zip(records, messages):
            if message.actor_observations is not None:
                record['message_observations'] = np.array(message.actor_observations)
        return actions, records

    @torch.no_grad()
    def values(self, state):
        r, c = self.critic(self.tensor(state).unsqueeze(0))
        return float(r.item()) * self.settings.reward_value_scale, float(c.item())

    @torch.no_grad()
    def values_batch(self, states):
        r, c = self.critic(self.tensor(np.asarray(states)))
        return (r.cpu().numpy().astype(np.float64) * self.settings.reward_value_scale,
                c.cpu().numpy().astype(np.float64))

    @torch.no_grad()
    def rollout_values(self, states):
        """State-only baselines, evaluated before either sampled action stage."""
        values = self.critic(self.tensor(np.asarray(states)), with_events=self.settings.minimize_collision,
                             with_breach=self.settings.breach_constraint)
        fields = ('value_r', 'value_c', 'value_b') if self.settings.breach_constraint else ('value_r', 'value_c', 'value_s', 'value_b')
        result = {name: value.cpu().numpy().astype(np.float64) for name, value in zip(fields, values)}
        result['value_r'] *= self.settings.reward_value_scale
        return result

    def evaluate_batch(self, batch):
        return self.actor.evaluate_actions(batch['obs'], batch['proposals'], batch['boids'],
                                           batch['edges'], batch['mask'], batch['proposal_raw'], batch['gate_raw'],
                                           batch.get('gate_scale', 1.))

    @torch.no_grad()
    def _policy_statistics(self, data, advantage):
        """Evaluate the saved behavior samples after an optimizer epoch.

        These are empirical training-batch quantities, not environment returns
        or a guarantee on unseen states. No new actions or messages are sampled.
        """
        totals = np.zeros(4, dtype=np.float64)
        count = len(advantage)
        constraint_totals, constraint_count = np.zeros(3 if self.settings.breach_constraint else 2), 0
        if self.settings.guard_task_surrogate:
            ordinary = ~data['is_curriculum']
            if not ordinary.any():
                raise ValueError('Task preservation needs ordinary on-policy episodes.')
            constraint_advantages = []
            factors = [(1., 'advantage_r'), (-1., 'advantage_c')]
            if self.settings.breach_constraint:
                factors.append((-1., 'advantage_b'))
            for sign, key in factors:
                raw = data[key]
                constraint_advantages.append(sign * (raw - raw[ordinary].mean()) /
                    raw[ordinary].std(unbiased=False).clamp_min(1e-8))
        for start in range(0, count, self.settings.minibatch_size):
            end = min(start + self.settings.minibatch_size, count)
            batch = {key: value[start:end] for key, value in data.items()}
            logp_l, logp_g, _ = self.evaluate_batch(batch)
            dl, dg = logp_l - batch['old_logp_l'], logp_g - batch['old_logp_g']
            joint = (dl + dg).sum(-1)
            surrogate = -clipped_policy_loss(dl, dg, advantage[start:end],
                                             self.settings.clip_ratio, self.settings.joint_ratio)
            totals += (end - start) * np.asarray([
                float(surrogate), float((joint.expm1() - joint).mean()),
                float((dl.expm1() - dl).mean()), float((dg.expm1() - dg).mean())])
            if self.settings.guard_task_surrogate:
                keep = ordinary[start:end]
                n = int(keep.sum())
                if n:
                    constraint_totals += n * np.asarray([float(-clipped_policy_loss(
                        dl[keep], dg[keep], values[start:end][keep], self.settings.clip_ratio,
                        self.settings.joint_ratio)) for values in constraint_advantages])
                    constraint_count += n
        surrogate, kl_joint, kl_l, kl_g = totals / count
        result = dict(surrogate=float(surrogate), kl_joint=float(kl_joint),
                      kl_limit=float(kl_joint if self.settings.joint_ratio else max(kl_l, kl_g)))
        if self.settings.guard_task_surrogate:
            names = ('task_surrogate', 'cost_surrogate', 'breach_surrogate')
            result.update(zip(names, constraint_totals / constraint_count))
        return result

    def _backtrack_full_policy(self, data, advantage, reference):
        """Retry one complete-rollout Adam direction without collecting new data.

        Gradients are computed once at the restored policy. Each trial restores
        its parameters AND pre-step Adam state, then scales only the step size.
        Thus accepted moments belong to exactly one actual gradient step.
        """
        cfg = self.settings
        saved_actor = copy.deepcopy(self.actor.state_dict())
        saved_optimizer = copy.deepcopy(self.actor_optimizer.state_dict())
        learning_rates = [group['lr'] for group in self.actor_optimizer.param_groups]
        self.actor_optimizer.zero_grad(set_to_none=True)
        count = len(advantage)
        for start in range(0, count, cfg.minibatch_size):
            end = min(start+cfg.minibatch_size, count)
            batch = {key:value[start:end] for key,value in data.items()}
            logp_l, logp_g, entropy = self.evaluate_batch(batch)
            loss = clipped_policy_loss(logp_l-batch['old_logp_l'], logp_g-batch['old_logp_g'],
                advantage[start:end], cfg.clip_ratio, cfg.joint_ratio) - cfg.entropy_coef*entropy.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite full-rollout policy loss.')
            (loss*((end-start)/count)).backward()
        norms = [torch.nn.utils.clip_grad_norm_(group['params'], cfg.max_grad_norm, error_if_nonfinite=True)
                 for group in self.actor_optimizer.param_groups]
        gradients = [(parameter, parameter.grad.detach().clone()) for parameter in self.actor.parameters()
                     if parameter.grad is not None]
        accepted, candidate, trials, scale = False, reference, 0, 0.
        for attempt in range(cfg.full_batch_backtracking_steps):
            trials = attempt+1
            self.actor.load_state_dict(saved_actor)
            self.actor_optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
            for parameter, gradient in gradients:
                parameter.grad = gradient.clone()
            for group, learning_rate in zip(self.actor_optimizer.param_groups, learning_rates):
                group['lr'] = learning_rate*(.5**attempt)
            self.actor_optimizer.step()
            # Record the configured LR, with effective trial scale separately.
            for group, learning_rate in zip(self.actor_optimizer.param_groups, learning_rates):
                group['lr'] = learning_rate
            candidate = self._policy_statistics(data, advantage)
            feasible = (all(np.isfinite(value) for value in candidate.values()) and
                        candidate['kl_limit'] <= cfg.target_kl and
                        candidate['surrogate'] > reference['surrogate']+1e-12)
            if cfg.guard_task_surrogate:
                names = ('task_surrogate', 'cost_surrogate', 'breach_surrogate') if cfg.breach_constraint else ('task_surrogate', 'cost_surrogate')
                feasible = feasible and all(candidate[key] >= reference[key]-1e-8 for key in names)
            if feasible:
                accepted, scale = True, .5**attempt
                break
        if not accepted:
            self.actor.load_state_dict(saved_actor)
            self.actor_optimizer.load_state_dict(copy.deepcopy(saved_optimizer))
            candidate = reference
        self.actor_optimizer.zero_grad(set_to_none=True)
        return accepted, candidate, dict(full_gradient_backtracking_trials=trials,
            full_gradient_step_scale=scale, full_gradient_norm=float(torch.linalg.vector_norm(torch.stack(norms))))

    def update(self, rollout):
        cfg = self.settings
        data = rollout.tensors(cfg, self.device)
        if any(not torch.isfinite(value).all() for value in data.values()):
            raise FloatingPointError('Non-finite rollout data.')
        fixed_lambda = self.lagrange.value
        if cfg.minimize_collision:
            fixed_success, fixed_breach = self.success_lagrange.value, self.breach_lagrange.value
            advantage = (fixed_success * data['advantage_s'] - fixed_breach * data['advantage_b'] -
                         (1. + fixed_lambda) * data['advantage_c'])
        else:
            advantage = data['advantage_r'] - fixed_lambda * data['advantage_c']
            if cfg.breach_constraint:
                fixed_breach = self.breach_lagrange.value
                advantage = advantage - fixed_breach * data['advantage_b']
        diagnostics = dict(reward_advantage_std=float(data['advantage_r'].std(unbiased=False)),
                           weighted_cost_advantage_std=float(
                               ((fixed_lambda + float(cfg.minimize_collision)) * data['advantage_c']).std(unbiased=False)))
        if cfg.curriculum_cost_baseline:
            selected = data['group_cost_weight'] > 0
            if selected.any():
                diagnostics.update(
                    recovery_critic_brier=float((data['return_c'][selected] - data['value_c'][selected]).square().mean()),
                    recovery_group_brier=float((data['return_c'][selected] - data['group_cost_baseline'][selected]).square().mean()))
                if cfg.breach_constraint or cfg.minimize_collision:
                    diagnostics.update(
                        recovery_breach_critic_brier=float((data['return_b'][selected] - data['value_b'][selected]).square().mean()),
                        recovery_breach_group_brier=float((data['return_b'][selected] - data['group_breach_baseline'][selected]).square().mean()))
        if cfg.curriculum_reward_baseline:
            selected = data['group_reward_weight'] > 0
            if selected.any():
                diagnostics.update(
                    recovery_reward_critic_mse=float((data['return_r'][selected] - data['value_r'][selected]).square().mean()),
                    recovery_reward_group_mse=float((data['return_r'][selected] - data['group_reward_baseline'][selected]).square().mean()))
        value_suffixes = ('r', 'c', 's', 'b') if cfg.minimize_collision else (('r', 'c', 'b') if cfg.breach_constraint else ('r', 'c'))
        for suffix in value_suffixes:
            variance = data['return_' + suffix].var(unbiased=False)
            diagnostics['explained_variance_' + suffix] = (
                float(1. - (data['return_' + suffix] - data['value_' + suffix]).var(unbiased=False) / variance)
                if variance > 1e-8 else None)
        advantage = ((advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)).detach()
        count = len(advantage)
        totals, weight = {}, 0
        self.actor.train()
        self.critic.train()
        actor_stopped = False
        actor_steps = 0
        guard_metrics = {}
        if cfg.full_batch_guard:
            initial_policy = self._policy_statistics(data, advantage)
            if (not all(np.isfinite(value) for value in initial_policy.values()) or
                    initial_policy['kl_limit'] > cfg.target_kl):
                raise ValueError('The rollout does not match the current behavior policy within target KL.')
            best_policy = initial_policy
            best_actor = copy.deepcopy(self.actor.state_dict())
            best_optimizer = copy.deepcopy(self.actor_optimizer.state_dict())
            best_epoch = best_steps = 0
        for epoch in range(cfg.ppo_epochs):
            steps_before_epoch = actor_steps
            indices = torch.randperm(count, device=self.device)
            for start in range(0, count, cfg.minibatch_size):
                index = indices[start:start + cfg.minibatch_size]
                batch = {key: value[index] for key, value in data.items()}
                logp_l, logp_g, entropy = self.evaluate_batch(batch)
                delta_l, delta_g = logp_l - batch['old_logp_l'], logp_g - batch['old_logp_g']
                ratio_l, ratio_g = delta_l.exp(), delta_g.exp()
                delta_joint = (delta_l + delta_g).sum(-1)
                kl_joint = (delta_joint.expm1() - delta_joint).mean()
                kl_limit = float(kl_joint.detach()) if cfg.joint_ratio else max(
                    float((ratio_l - 1 - delta_l).mean().detach()),
                    float((ratio_g - 1 - delta_g).mean().detach()))
                if cfg.target_kl and kl_limit > cfg.target_kl:
                    actor_stopped = True
                actor_loss = clipped_policy_loss(delta_l, delta_g, advantage[index], cfg.clip_ratio,
                                                 cfg.joint_ratio) - cfg.entropy_coef * entropy.mean()
                values = self.critic(batch['state'], with_events=cfg.minimize_collision, with_breach=cfg.breach_constraint)
                value_r, value_c = values[:2]
                reward_loss = F.mse_loss(value_r, batch['return_r'] / cfg.reward_value_scale)
                cost_loss = F.mse_loss(value_c, batch['return_c'])
                critic_loss = reward_loss + cost_loss
                event_metrics = {}
                if cfg.minimize_collision:
                    success_loss = F.mse_loss(values[2], batch['return_s'])
                    breach_loss = F.mse_loss(values[3], batch['return_b'])
                    critic_loss = critic_loss + success_loss + breach_loss
                    event_metrics = dict(success_value_loss=float(success_loss.detach()),
                                         breach_value_loss=float(breach_loss.detach()))
                elif cfg.breach_constraint:
                    breach_loss = F.mse_loss(values[2], batch['return_b'])
                    critic_loss = critic_loss + breach_loss
                    event_metrics = dict(breach_value_loss=float(breach_loss.detach()))
                if not torch.isfinite(actor_loss + critic_loss):
                    raise FloatingPointError('Non-finite PPO loss.')
                self.actor_optimizer.zero_grad()
                actor_norm = torch.tensor(0.)
                if not actor_stopped:
                    actor_loss.backward()
                    norms = [torch.nn.utils.clip_grad_norm_(group['params'], cfg.max_grad_norm,
                              error_if_nonfinite=True) for group in self.actor_optimizer.param_groups]
                    actor_norm = torch.linalg.vector_norm(torch.stack(norms))
                gradient_metrics = {}
                for name in ('proposal_mean', 'relation_encoder', 'priority_head', 'coordination_encoder',
                             'adapter', 'gate_mean', 'gate_log_std'):
                    parameters = getattr(self.actor, name).parameters()
                    gradient_metrics['grad_' + name] = sum(float(p.grad.norm()) for p in parameters if p.grad is not None)
                if self.actor.compatibility_gain_raw is not None:
                    gradient = self.actor.compatibility_gain_raw.grad
                    gradient_metrics['grad_compatibility_prior'] = float(gradient.abs()) if gradient is not None else 0.
                if self.actor.compatibility_context_head is not None:
                    gradient_metrics['grad_compatibility_context'] = sum(float(p.grad.norm())
                        for p in self.actor.compatibility_context_head.parameters() if p.grad is not None)
                if not actor_stopped:
                    self.actor_optimizer.step()
                    actor_steps += 1
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm,
                                                             error_if_nonfinite=True)
                self.critic_optimizer.step()
                metrics = dict(actor_loss=float(actor_loss.detach()), reward_value_loss=float(reward_loss.detach()),
                               cost_value_loss=float(cost_loss.detach()), latent_entropy=float(entropy.mean().detach()),
                               actor_grad_norm=float(actor_norm), critic_grad_norm=float(critic_norm),
                               kl_proposal=float((ratio_l - 1 - delta_l).mean().detach()),
                               kl_gate=float((ratio_g - 1 - delta_g).mean().detach()),
                               kl_joint=float(kl_joint.detach()),
                               clip_fraction=float(((ratio_l - 1).abs() > cfg.clip_ratio).float().mean().detach()),
                               gate_clip_fraction=float(((ratio_g - 1).abs() > cfg.clip_ratio).float().mean().detach()),
                               joint_clip_fraction=float((delta_joint.expm1().abs() > cfg.clip_ratio).float().mean().detach()),
                               **gradient_metrics, **event_metrics)
                for key, value in metrics.items():
                    totals[key] = totals.get(key, 0.) + value * len(index)
                weight += len(index)
            if cfg.full_batch_guard and actor_steps > steps_before_epoch:
                candidate = self._policy_statistics(data, advantage)
                feasible = (all(np.isfinite(value) for value in candidate.values()) and
                            candidate['kl_limit'] <= cfg.target_kl)
                if cfg.guard_task_surrogate:
                    guarded = ('task_surrogate', 'cost_surrogate', 'breach_surrogate') if cfg.breach_constraint else ('task_surrogate', 'cost_surrogate')
                    feasible = feasible and all(candidate[key] >= initial_policy[key] - 1e-8
                        for key in guarded)
                if feasible and candidate['surrogate'] > best_policy['surrogate']:
                    best_policy = candidate
                    best_actor = copy.deepcopy(self.actor.state_dict())
                    best_optimizer = copy.deepcopy(self.actor_optimizer.state_dict())
                    best_epoch, best_steps = epoch + 1, actor_steps
                if not feasible:
                    # A minibatch KL check cannot catch its final optimizer
                    # step's overshoot. Restore weights AND Adam moments.
                    self.actor.load_state_dict(best_actor)
                    self.actor_optimizer.load_state_dict(best_optimizer)
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    actor_stopped = True
        if cfg.full_batch_guard:
            self.actor.load_state_dict(best_actor)
            self.actor_optimizer.load_state_dict(best_optimizer)
            self.actor_optimizer.zero_grad(set_to_none=True)
            backtracking = {}
            if not best_steps and cfg.full_batch_backtracking_steps:
                accepted, candidate, backtracking = self._backtrack_full_policy(data, advantage, best_policy)
                actor_steps += backtracking['full_gradient_backtracking_trials']
                if accepted:
                    best_policy, best_epoch, best_steps = candidate, -1, 1
            guard_metrics = dict(policy_kl_after=best_policy['kl_limit'],
                                 policy_joint_kl_after=best_policy['kl_joint'],
                                 policy_surrogate_gain=best_policy['surrogate'] - initial_policy['surrogate'],
                                 policy_epoch_selected=best_epoch, actor_accepted_steps=best_steps,
                                 actor_rollback=float(best_steps != actor_steps), **backtracking)
            if cfg.guard_task_surrogate:
                guard_metrics.update(task_surrogate_gain=best_policy['task_surrogate'] - initial_policy['task_surrogate'],
                                     cost_surrogate_gain=best_policy['cost_surrogate'] - initial_policy['cost_surrogate'])
                if cfg.breach_constraint:
                    guard_metrics['breach_surrogate_gain'] = best_policy['breach_surrogate'] - initial_policy['breach_surrogate']
        self.total_steps += rollout.steps
        self.updates += 1
        self.lagrange.update(rollout.collision_rate)  # Exactly once, after all epochs.
        if cfg.minimize_collision:
            self.success_lagrange.update(1. - rollout.success_rate)
            self.breach_lagrange.update(rollout.breach_rate)
            diagnostics.update(success_multiplier_used=fixed_success,
                               success_multiplier_next=self.success_lagrange.value,
                               breach_multiplier_used=fixed_breach,
                               breach_multiplier_next=self.breach_lagrange.value,
                               collision_objective_weight=1. + fixed_lambda)
        elif cfg.breach_constraint:
            self.breach_lagrange.update(rollout.breach_rate)
            diagnostics.update(breach_multiplier_used=fixed_breach,
                               breach_multiplier_next=self.breach_lagrange.value, breach_rate=rollout.breach_rate)
        return dict({key: value / weight for key, value in totals.items()},
                    **diagnostics, **guard_metrics, actor_optimizer_steps=actor_steps,
                    lambda_used=fixed_lambda, lambda_next=self.lagrange.value,
                    collision_rate=rollout.collision_rate, steps=self.total_steps, update=self.updates,
                    actor_early_stopped=float(actor_stopped))

    def save(self, path):
        state = np.random.get_state()
        rng = dict(numpy_kind=state[0], numpy_keys=torch.tensor(state[1].astype(np.int64)),
                   numpy_pos=state[2], numpy_has_gauss=state[3], numpy_cached=state[4],
                   python=random.getstate(), torch=torch.get_rng_state())
        if torch.cuda.is_available():
            rng['cuda'] = torch.cuda.get_rng_state_all()
        torch.save(dict(algorithm=self.algorithm, format_version=self.format_version, config=self.config,
                        actor=self.actor.state_dict(), critic=self.critic.state_dict(),
                        actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
                        lagrange=self.lagrange.value, success_lagrange=self.success_lagrange.value,
                        breach_lagrange=self.breach_lagrange.value, total_steps=self.total_steps, updates=self.updates,
                        training_state=self.training_state, rng=rng), Path(path))

    @classmethod
    def from_checkpoint(cls, path, device='cpu', restore_rng=False):
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved.get('algorithm') != cls.algorithm or saved.get('format_version') != cls.format_version:
            raise ValueError('Not a supported predictive MAPPO checkpoint.')
        agent = cls(saved['config'], device)
        agent.actor.load_state_dict(saved['actor'])
        agent.critic.load_state_dict(saved['critic'])
        agent.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        agent.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        agent.lagrange.value = float(saved['lagrange'])
        agent.success_lagrange.value = float(saved.get('success_lagrange', agent.settings.success_lagrange_init))
        agent.breach_lagrange.value = float(saved.get('breach_lagrange', agent.settings.breach_lagrange_init))
        agent.total_steps, agent.updates = int(saved['total_steps']), int(saved['updates'])
        agent.training_state = saved.get('training_state', {})
        if not np.isfinite(agent.lagrange.value) or agent.lagrange.value < 0:
            raise ValueError('Invalid checkpoint multiplier.')
        if any(not np.isfinite(value) or value < 0 for value in
               (agent.success_lagrange.value, agent.breach_lagrange.value)):
            raise ValueError('Invalid checkpoint event multiplier.')
        if any(not torch.isfinite(p).all() for net in (agent.actor, agent.critic) for p in net.parameters()):
            raise ValueError('Checkpoint has non-finite parameters.')
        if restore_rng:
            rng = saved['rng']
            np.random.set_state((rng['numpy_kind'], rng['numpy_keys'].numpy().astype(np.uint32),
                                 rng['numpy_pos'], rng['numpy_has_gauss'], rng['numpy_cached']))
            random.setstate(rng['python'])
            torch.set_rng_state(rng['torch'])
            if torch.cuda.is_available() and 'cuda' in rng:
                torch.cuda.set_rng_state_all(rng['cuda'])
        agent.actor.eval()
        agent.critic.eval()
        return agent
