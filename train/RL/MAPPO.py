"""On-policy hierarchical MAPPO with optional verified team guidance."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from .channel_networks import ChannelActor, TeamCritic
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
        self.value = TeamCritic(hidden).to(self.device)
        self.q = TeamCritic(hidden, action_conditioned=True).to(self.device)
        self.value_target = copy.deepcopy(self.value).requires_grad_(False)
        lr = options.get("learning_rate", 1e-4)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, eps=1e-5)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr, eps=1e-5)
        self.q_optimizer = torch.optim.Adam(self.q.parameters(), lr=lr, eps=1e-5)
        self.guidance = TeamGuidance(config.get("guidance", {}))
        self.training_steps = 0

    def _optimize(self, loss, optimizer, parameters):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.options.get("max_grad_norm", .5), error_if_nonfinite=True)
        optimizer.step()
        return float(norm)

    def update(self, rollout, env):
        opts = self.options
        batch = rollout.batch(self.device, self.gamma, opts.get("gae_lambda", .95))
        frame, decisions = batch["frame"], batch["decision"]
        old_logp = decisions["logp_candidate"] + decisions["logp_gate"]
        with torch.no_grad():
            replay_logp, _ = self.actor.evaluate_actions(frame, decisions, entropy=False)
            error = ((replay_logp - old_logp).abs().squeeze(-1) * frame["mask"]).max().item()
            if error > 1e-4:
                raise RuntimeError(f"Historical two-stage likelihood mismatch: {error}")
            q_targets = batch["reward"] + self.gamma * (~batch["terminal"]).float() * self.value_target(batch["next_frame"])
        epochs = int(opts.get("epochs", 4))
        minibatch = int(opts.get("minibatch_steps", 64))
        if epochs < 1 or minibatch < 1:
            raise ValueError("PPO epochs and minibatch_steps must be positive")
        size = len(rollout.rows)
        critic_losses = []
        for _ in range(epochs):
            for indices in torch.randperm(size, device=self.device).split(minibatch):
                subset = select_frame(frame, indices)
                v_loss = F.smooth_l1_loss(self.value(subset), batch["returns"][indices])
                q_loss = F.smooth_l1_loss(self.q(subset, decisions["executed"][indices]), q_targets[indices])
                self._optimize(v_loss, self.value_optimizer, self.value.parameters())
                self._optimize(q_loss, self.q_optimizer, self.q.parameters())
                critic_losses.append((float(v_loss.detach()), float(q_loss.detach())))

        targets, guide_weights, calibration, guidance_stats = self.guidance.prepare(self, rollout, batch, env)
        # Separate labels can calibrate Q, but cannot change PPO transitions,
        # advantages, old likelihoods, or V's on-policy return targets.
        if calibration:
            cf = collate_frames([x[0] for x in calibration], self.device)
            actions = torch.zeros((*cf["mask"].shape, 2), device=self.device)
            for i, (_, action, _) in enumerate(calibration):
                actions[i, :len(action)] = torch.as_tensor(action, device=self.device, dtype=torch.float32)
            labels = torch.tensor([x[2] for x in calibration], device=self.device).unsqueeze(-1)
            q_loss = F.smooth_l1_loss(self.q(cf, actions), labels)
            self._optimize(q_loss, self.q_optimizer, self.q.parameters())

        clip = float(opts.get("clip_ratio", .2))
        stats = []
        stopped = False
        for _ in range(epochs):
            for indices in torch.randperm(size, device=self.device).split(minibatch):
                subset = select_frame(frame, indices)
                data = {k: v[indices] for k, v in decisions.items()}
                logp, entropy = self.actor.evaluate_actions(subset, data)
                log_ratio = logp - old_logp[indices]
                ratio = log_ratio.clamp(-20., 20.).exp()
                advantage = batch["advantage"][indices, None, None]
                surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage)
                ppo_loss = -team_mean(surrogate, subset["mask"]).mean()
                entropy_mean = team_mean(entropy, subset["mask"]).mean()
                guide_loss = torch.zeros((), device=self.device)
                if torch.any(guide_weights[indices] > 0):
                    _, predicted = self.actor.representative(subset, data["candidate"], data["messages"], adapter_only=True)
                    per_agent = ((predicted - targets[indices]) / THRUST_SCALE).square().mean(-1)
                    guide_loss = (team_mean(per_agent, subset["mask"]) * guide_weights[indices]).mean()
                total_loss = (ppo_loss + self.guidance.options.get("coefficient", .1) * guide_loss
                              - opts.get("entropy_coefficient", .01) * entropy_mean)
                grad_norm = self._optimize(total_loss, self.actor_optimizer, self.actor.parameters())
                with torch.no_grad():
                    # Measure the entire rollout after the combined PPO + guide
                    # update, so auxiliary drift is included in the KL guard.
                    updated, _ = self.actor.evaluate_actions(frame, decisions, entropy=False)
                    delta = updated - old_logp
                    approx_kl = team_mean(delta.clamp(-20., 20.).exp() - 1 - delta, frame["mask"]).mean()
                    clip_fraction = team_mean((ratio.sub(1).abs() > clip).float(), subset["mask"]).mean()
                stats.append((float(ppo_loss.detach()), float(guide_loss.detach()), float(entropy_mean.detach()),
                              float(approx_kl), float(clip_fraction), grad_norm))
                if approx_kl > opts.get("target_kl", .02):
                    stopped = True
                    break
            if stopped:
                break
        tau = opts.get("value_target_tau", .05)
        with torch.no_grad():
            for source, target in zip(self.value.parameters(), self.value_target.parameters()):
                target.lerp_(source, tau)
        averages = np.mean(stats, axis=0)
        result = dict(zip(("ppo_loss", "guide_loss", "entropy", "approx_kl", "clip_fraction", "actor_grad_norm"), averages.tolist()))
        result.update(value_loss=float(np.mean(critic_losses, axis=0)[0]),
                      q_loss=float(np.mean(critic_losses, axis=0)[1]),
                      likelihood_error=error, kl_early_stop=int(stopped),
                      guidance_weight=float(guide_weights.mean()), **guidance_stats)
        if not all(np.isfinite(v) for v in result.values()):
            raise FloatingPointError("Non-finite training diagnostics")
        return result

    def save(self, path):
        torch.save(dict(format=CHECKPOINT_FORMAT, feature_version=FEATURE_VERSION,
                        actor_config=self.actor.config, actor=self.actor.state_dict(),
                        value=self.value.state_dict(), q=self.q.state_dict(),
                        value_target=self.value_target.state_dict(), config=self.config,
                        actor_optimizer=self.actor_optimizer.state_dict(),
                        value_optimizer=self.value_optimizer.state_dict(),
                        q_optimizer=self.q_optimizer.state_dict(), training_steps=self.training_steps), path)

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
        agent.training_steps = data["training_steps"]
        return agent
