"""On-policy hierarchical MAPPO with optional verified team guidance."""
import copy
import random
import numpy as np
import torch
from torch.nn import functional as F
from .channel_networks import ChannelActor, TeamCritic
from .calibration import TerminalCalibration
from .control import THRUST_SCALE
from .deployment import CHECKPOINT_FORMAT
from .guidance import TeamGuidance
from .observations import FEATURE_VERSION, collate_frames, select_frame
from .rollout import team_mean


class MAPPO:
    def __init__(self, config, device="cpu"):
        self.config = config
        self.device = torch.device(device)
        options = config.get("mappo", {})
        self.options = options
        self.gamma = float(options.get("gamma", .99))
        self.actor = ChannelActor(**config.get("policy", {})).to(self.device)
        hidden = int(options.get("critic_hidden", self.actor.config["hidden"]))
        normalize_value = bool(options.get("value_normalization", False))
        layer_norm = bool(options.get("critic_layer_norm", False))
        self.value = TeamCritic(hidden, normalize_value=normalize_value, layer_norm=layer_norm).to(self.device)
        self.q = TeamCritic(hidden, action_conditioned=True, normalize_value=normalize_value, layer_norm=layer_norm).to(self.device)
        self.value_target = copy.deepcopy(self.value).requires_grad_(False)
        lr = options.get("learning_rate", 1e-4)
        critic_lr = options.get("critic_learning_rate", lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=critic_lr, eps=1e-5)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=critic_lr, eps=1e-5)
        self.guidance = TeamGuidance(config.get("guidance", {}))
        self.training_steps = 0
        self.trainer_state = {}
        self.training_rng = None
        self.terminal_calibration = TerminalCalibration(int(options.get("terminal_q_capacity", 128)))
        self.terminal_q_ready = False
        self.cost_coefficient = float(options.get("collision_cost_coefficient", 0.))
        if self.cost_coefficient < 0.:
            raise ValueError("Collision cost coefficient must be nonnegative")
        self.cost_value = None
        self.cost_optimizer = None
        if self.cost_coefficient:
            self.cost_value = TeamCritic(hidden, layer_norm=layer_norm).to(self.device)
            # Estimate episode collision probability in [0, 1], with no
            # discount of a late collision. This critic is never deployed.
            with torch.no_grad():
                self.cost_value.readout[-1].weight.zero_()
                self.cost_value.readout[-1].bias.fill_(-2.9444389791664403)  # prior 5%
            self.cost_optimizer = torch.optim.Adam(self.cost_value.parameters(), lr=critic_lr, eps=1e-5)

    def collision_probability(self, frame):
        if self.cost_value is None:
            raise RuntimeError("Collision cost critic is disabled")
        return self.cost_value(frame).sigmoid()

    def _optimize(self, loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.options.get("max_grad_norm", .5), error_if_nonfinite=True)
        optimizer.step()
        return float(norm)

    def _actor_step(self, loss, frame, decisions, old_logp):
        """Accept only updates inside the rollout KL limit, including guidance.

        A narrow pretrained distribution can cross the limit in a single Adam
        step. Restore both parameters and Adam moments before retrying the same
        clipped gradient at a smaller learning rate. Rejected steps leave no
        optimizer-state changes. The configured learning rate is not decayed.
        """
        parameters = list(self.actor.parameters())
        before = [parameter.detach().clone() for parameter in parameters]
        optimizer = self.actor_optimizer
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        rates = [group["lr"] for group in optimizer.param_groups]
        limit = float(self.options.get("target_kl", .02))
        retries = int(self.options.get("max_kl_backtracks", 8))
        if limit <= 0 or retries < 0:
            raise ValueError("target_kl must be positive and max_kl_backtracks nonnegative")

        @torch.no_grad()
        def restore():
            for parameter, original in zip(parameters, before):
                parameter.copy_(original)
            # load_state_dict may share state tensors on the same device.
            optimizer.load_state_dict(copy.deepcopy(optimizer_before))

        @torch.no_grad()
        def divergence():
            updated, _ = self.actor.evaluate_actions(frame, decisions, entropy=False)
            delta = updated - old_logp
            return float(team_mean(delta.clamp(-20., 20.).exp() - 1 - delta, frame["mask"]).mean())

        norm = self._optimize(loss, optimizer, parameters)
        rejected = False
        for attempt in range(retries + 1):
            kl = divergence()
            if np.isfinite(kl) and kl <= limit:
                break
            restore()
            if attempt == retries:
                rejected = True
                kl = divergence()
                break
            for group, rate in zip(optimizer.param_groups, rates):
                group["lr"] = rate * .5 ** (attempt + 1)
            optimizer.step()
        for group, rate in zip(optimizer.param_groups, rates):
            group["lr"] = rate
        return norm, kl, attempt, rejected

    def update(self, rollout, env):
        opts = self.options
        batch = rollout.batch(self.device, self.gamma, opts.get("gae_lambda", .95), opts.get("cost_gae_lambda", .99))
        frame, decisions = batch["frame"], batch["decision"]
        old_logp = decisions["logp_candidate"] + decisions["logp_gate"]
        with torch.no_grad():
            replay_logp, _ = self.actor.evaluate_actions(frame, decisions, entropy=False)
            error = ((replay_logp - old_logp).abs().squeeze(-1) * frame["mask"]).max().item()
            if error > 1e-4:
                raise RuntimeError(f"Historical two-stage likelihood mismatch: {error}")
            q_targets = batch["reward"] + self.gamma * (~batch["terminal"]).float() * self.value_target(batch["next_frame"])
        self.value.update_normalization(batch["returns"])
        self.q.update_normalization(q_targets)
        epochs = int(opts.get("epochs", 4))
        minibatch = int(opts.get("minibatch_steps", 64))
        if epochs < 1 or minibatch < 1:
            raise ValueError("PPO epochs and minibatch_steps must be positive")
        size = len(rollout.rows)
        terminal_updates = int(opts.get("terminal_q_updates", 0))
        terminal_interleave = float(opts.get("terminal_q_interleave_weight", 0.))
        if terminal_updates < 0 or terminal_interleave < 0:
            raise ValueError("Terminal Q update count and weight must be nonnegative")
        if terminal_updates or terminal_interleave:
            self.terminal_calibration.add(rollout.rows)
        critic_losses = []
        cost_losses = []
        for _ in range(epochs):
            for indices in torch.randperm(size, device=self.device).split(minibatch):
                subset = select_frame(frame, indices)
                v_loss = F.smooth_l1_loss(self.value(subset, normalized=True), self.value.normalize_targets(batch["returns"][indices]))
                q_loss = F.smooth_l1_loss(self.q(subset, decisions["executed"][indices], normalized=True), self.q.normalize_targets(q_targets[indices]))
                if terminal_interleave and len(self.terminal_calibration):
                    cf, actions, labels = self.terminal_calibration.sample(self.device)
                    q_loss = q_loss + terminal_interleave * F.mse_loss(
                        self.q(cf, actions, normalized=True), self.q.normalize_targets(labels))
                self._optimize(v_loss, self.value_optimizer, self.value.parameters())
                self._optimize(q_loss, self.q_optimizer, self.q.parameters())
                if self.cost_value:
                    cost_loss = F.binary_cross_entropy_with_logits(
                        self.cost_value(subset), batch["cost_returns"][indices].clamp(0., 1.))
                    self._optimize(cost_loss, self.cost_optimizer, self.cost_value.parameters())
                    cost_losses.append(float(cost_loss.detach()))
                critic_losses.append((float(v_loss.detach()), float(q_loss.detach())))

        terminal_stats = dict(terminal_q_samples=0, terminal_q_updates=0,
                              terminal_q_rmse_before=0., terminal_q_rmse_after=0.)
        if terminal_updates or terminal_interleave:
            terminal_stats["terminal_q_samples"] = len(self.terminal_calibration)
            full_probe = bool(opts.get("terminal_q_full_probe", False))
            probe = (self.terminal_calibration.probe(self.device) if full_probe
                     else self.terminal_calibration.sample(self.device))
            if probe is not None:
                def probe_rmse():
                    errors = (self.q(probe[0], probe[1]) - probe[2]).square()
                    return float((errors.mul(probe[3]).sum() if full_probe else errors.mean()).sqrt())

                with torch.no_grad():
                    terminal_stats["terminal_q_rmse_before"] = probe_rmse()
                for _ in range(terminal_updates):
                    cf, actions, labels = self.terminal_calibration.sample(self.device)
                    terminal_loss = F.mse_loss(self.q(cf, actions, normalized=True), self.q.normalize_targets(labels))
                    indices = torch.randint(size, (minibatch,), device=self.device)
                    anchor_loss = F.smooth_l1_loss(
                        self.q(select_frame(frame, indices), decisions["executed"][indices], normalized=True),
                        self.q.normalize_targets(q_targets[indices]))
                    self._optimize(anchor_loss + opts.get("terminal_q_weight", 1.) * terminal_loss,
                                   self.q_optimizer, self.q.parameters())
                with torch.no_grad():
                    terminal_stats["terminal_q_rmse_after"] = probe_rmse()
                terminal_stats["terminal_q_updates"] = terminal_updates
                required = opts.get("terminal_q_required_outcomes", [2, 3])
                self.terminal_q_ready = (all(self.terminal_calibration.groups[outcome] for outcome in required)
                                         and terminal_stats["terminal_q_rmse_after"] <= opts.get("terminal_q_max_rmse", 10.))

        targets, guide_weights, calibration, guidance_stats = self.guidance.prepare(self, rollout, batch, env)
        # Separate labels can calibrate Q, but cannot change PPO transitions,
        # advantages, old likelihoods, or V's on-policy return targets.
        if calibration:
            cf = collate_frames([x[0] for x in calibration], self.device)
            actions = torch.zeros((*cf["mask"].shape, 2), device=self.device)
            for i, (_, action, _) in enumerate(calibration):
                actions[i, :len(action)] = torch.as_tensor(action, device=self.device, dtype=torch.float32)
            labels = torch.tensor([x[2] for x in calibration], device=self.device).unsqueeze(-1)
            q_loss = F.smooth_l1_loss(self.q(cf, actions, normalized=True), self.q.normalize_targets(labels))
            self._optimize(q_loss, self.q_optimizer, self.q.parameters())

        clip = float(opts.get("clip_ratio", .2))
        actor_minibatch = int(opts.get("actor_minibatch_steps", minibatch))
        if actor_minibatch < 1:
            raise ValueError("actor_minibatch_steps must be positive")
        guide_indices = torch.where(guide_weights > 0)[0]
        guide_batch_mode = self.guidance.options.get("batch_mode", "rollout")
        if guide_batch_mode not in ("rollout", "validated"):
            raise ValueError("guidance.batch_mode must be rollout or validated")
        stats = []
        stopped = False
        backtracks = rejections = 0
        actor_samples = 0
        cost_policy_losses = []
        for _ in range(epochs):
            for indices in torch.randperm(size, device=self.device).split(actor_minibatch):
                actor_samples += len(indices)
                subset = select_frame(frame, indices)
                data = {k: v[indices] for k, v in decisions.items()}
                logp, entropy = self.actor.evaluate_actions(subset, data)
                log_ratio = logp - old_logp[indices]
                ratio = log_ratio.clamp(-20., 20.).exp()
                advantage = batch["advantage"][indices, None, None]
                surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
                ppo_loss = -team_mean(surrogate, subset["mask"]).mean()
                cost_policy_loss = torch.zeros((), device=self.device)
                if self.cost_value:
                    # PPO's pessimistic upper bound for minimizing a cost.
                    cost_adv = batch["cost_advantage"][indices, None, None]
                    cost_surrogate = torch.maximum(ratio * cost_adv, ratio.clamp(1 - clip, 1 + clip) * cost_adv)
                    cost_policy_loss = team_mean(cost_surrogate, subset["mask"]).mean()
                cost_policy_losses.append(float(cost_policy_loss.detach()))
                entropy_mean = team_mean(entropy, subset["mask"]).mean()
                guide_loss = torch.zeros((), device=self.device)
                if guide_batch_mode == "validated" and len(guide_indices):
                    # Sparse validated examples would otherwise often be absent
                    # before PPO reaches its KL limit. Each update sees all of
                    # this rollout's accepted labels, never an off-policy replay.
                    guided_frame = select_frame(frame, guide_indices)
                    _, predicted = self.actor.representative(
                        guided_frame, decisions["candidate"][guide_indices],
                        decisions["messages"][guide_indices], adapter_only=True)
                    action_error = ((predicted - targets[guide_indices]) / THRUST_SCALE).square().mean(-1)
                    weights = guide_weights[guide_indices]
                    guide_loss = (team_mean(action_error, guided_frame["mask"]) * weights).sum() / weights.sum()
                elif guide_batch_mode == "rollout" and torch.any(guide_weights[indices] > 0):
                    _, predicted = self.actor.representative(subset, data["candidate"], data["messages"], adapter_only=True)
                    per_agent = ((predicted - targets[indices]) / THRUST_SCALE).square().mean(-1)
                    guide_loss = (team_mean(per_agent, subset["mask"]) * guide_weights[indices]).mean()
                total_loss = (ppo_loss + self.cost_coefficient * cost_policy_loss
                              + self.guidance.options.get("coefficient", .1) * guide_loss
                              - opts.get("entropy_coefficient", .01) * entropy_mean)
                grad_norm, approx_kl, attempts, rejected = self._actor_step(total_loss, frame, decisions, old_logp)
                backtracks += attempts
                rejections += int(rejected)
                with torch.no_grad():
                    clip_fraction = team_mean((ratio.sub(1).abs() > clip).float(), subset["mask"]).mean()
                stats.append((float(ppo_loss.detach()), float(guide_loss.detach()), float(entropy_mean.detach()),
                              float(approx_kl), float(clip_fraction), grad_norm))
                if attempts or rejected or approx_kl >= opts.get("target_kl", .02):
                    stopped = True
                    break
            if stopped:
                break
        tau = opts.get("value_target_tau", .05)
        with torch.no_grad():
            # Polyak averaging must use the same normalized coordinates. First
            # rebase the target without changing its predictions, then blend.
            self.value_target.copy_normalization_from(self.value)
            for source, target in zip(self.value.parameters(), self.value_target.parameters()):
                target.lerp_(source, tau)
        averages = np.mean(stats, axis=0)
        result = dict(zip(("ppo_loss", "guide_loss", "entropy", "approx_kl", "clip_fraction", "actor_grad_norm"), averages.tolist()))
        result.update(value_loss=float(np.mean(critic_losses, axis=0)[0]),
                      q_loss=float(np.mean(critic_losses, axis=0)[1]),
                      likelihood_error=error, kl_early_stop=int(stopped),
                      actor_kl_backtracks=backtracks, actor_updates_rejected=rejections,
                      actor_minibatches=len(stats), actor_samples_processed=actor_samples,
                      max_actor_kl=max(row[3] for row in stats),
                      cost_value_loss=float(np.mean(cost_losses)) if cost_losses else 0.,
                      cost_policy_loss=float(np.mean(cost_policy_losses)),
                      collision_cost_coefficient=self.cost_coefficient,
                      guidance_weight=float(guide_weights.mean()), **guidance_stats, **terminal_stats)
        with torch.no_grad():
            target_variance = batch["returns"].var(unbiased=False)
            residual_variance = (batch["returns"] - self.value(frame)).var(unbiased=False)
            result["value_explained_variance"] = float(1 - residual_variance / target_variance.clamp_min(1e-8))
            result["mean_step_reward"] = float(batch["reward"].mean())
            result["gate_c"] = float(team_mean(decisions["gates"][..., 0], frame["mask"]).mean())
            result["gate_d"] = float(team_mean(decisions["gates"][..., -1], frame["mask"]).mean())
        if not all(np.isfinite(v) for v in result.values()):
            raise FloatingPointError("Non-finite training diagnostics")
        return result

    def save(self, path):
        numpy_state = np.random.get_state()
        rng = dict(python=random.getstate(), numpy=(numpy_state[0], numpy_state[1].tolist(),
                   *numpy_state[2:]), torch=torch.get_rng_state(),
                   cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        torch.save(dict(format=CHECKPOINT_FORMAT, feature_version=FEATURE_VERSION,
                        actor_config=self.actor.config, actor=self.actor.state_dict(),
                        value=self.value.state_dict(), q=self.q.state_dict(),
                        value_target=self.value_target.state_dict(), config=self.config,
                        actor_optimizer=self.actor_optimizer.state_dict(),
                        value_optimizer=self.value_optimizer.state_dict(),
                        q_optimizer=self.q_optimizer.state_dict(), training_steps=self.training_steps,
                        trainer_state=self.trainer_state,
                        training_rng=rng,
                        terminal_q_calibration=self.terminal_calibration.state_dict(),
                        cost_value=self.cost_value.state_dict() if self.cost_value else None,
                        cost_optimizer=self.cost_optimizer.state_dict() if self.cost_optimizer else None,
                        guidance_history=list(self.guidance.validation_history)), path)

    def restore_training_rng(self):
        """Only training resumes restore RNG; evaluation loading has no such side effect."""
        if self.training_rng is None:
            return
        rng = self.training_rng
        random.setstate(rng["python"])
        state = rng["numpy"]
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
        torch.set_rng_state(rng["torch"].cpu())
        if rng["cuda"] is not None and self.device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])

    @classmethod
    def load(cls, path, device="cpu"):
        data = torch.load(path, map_location=device, weights_only=True)
        if data.get("format") != CHECKPOINT_FORMAT or data.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Incompatible MAPPO checkpoint")
        agent = cls(data["config"], device)
        for key in ("actor", "value", "q", "value_target"):
            getattr(agent, key).load_state_dict(data[key])
        for key in ("actor_optimizer", "value_optimizer", "q_optimizer"):
            getattr(agent, key).load_state_dict(data[key])
        if agent.cost_value:
            agent.cost_value.load_state_dict(data["cost_value"])
            agent.cost_optimizer.load_state_dict(data["cost_optimizer"])
        agent.training_steps = data["training_steps"]
        agent.trainer_state = data.get("trainer_state", {})
        agent.training_rng = data.get("training_rng")
        agent.guidance.validation_history.extend(data.get("guidance_history", []))
        agent.terminal_calibration.load_state_dict(data.get("terminal_q_calibration", {}))
        return agent
