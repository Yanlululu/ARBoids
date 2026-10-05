"""Shared two-stage actor and a separate centralized reward/cost critic."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def initialize(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, math.sqrt(2.))
        nn.init.zeros_(module.bias)


class PredictiveActor(nn.Module):
    def __init__(self, hidden_dim=512, relation_dim=64, coordination_dim=16, edge_dim=18,
                 baseline_initialization=False, baseline_gate_bound=0., theta_space_gate=False,
                 compatibility_prior=False, compatibility_gain=1., compatibility_margin=7.,
                 compatibility_temperature=1., compatibility_neighbor_temperature=2., prediction_distance_scale=5.,
                 compatibility_compact_risk=False, compatibility_task_priority=False,
                 compatibility_priority_temperature=5., compatibility_peer_intent=False,
                 censored_gate_likelihood=False, compatibility_mixture_points=0,
                 compatibility_joint_mixture=False, compatibility_task_prediction=False,
                 compatibility_task_penalty=1., compatibility_context_gain=False,
                 compatibility_terminal_prediction=False, prediction_horizon=2., prediction_terminal_buffer=.4):
        super().__init__()
        h, r, c = hidden_dim, relation_dim, coordination_dim
        if h < 16 or h % 4 or min(r, c) < 1:
            raise ValueError('hidden_dim must be a multiple of four >=16; head dimensions must be positive.')
        self.register_buffer('local_scale', torch.tensor(
            [60., math.pi, 60., math.pi, 5., math.pi,
             5., math.pi, 5., math.pi, 60., math.pi, 1., 1.]))
        self.register_buffer('neighbor_scale', torch.tensor([60., math.pi]))
        self.activation = nn.LeakyReLU() if baseline_initialization else nn.Tanh()
        if baseline_initialization:
            self.local_scale.fill_(1.)
            self.neighbor_scale.fill_(1.)
        self.f1 = nn.Linear(6, h // 2)
        self.f2 = nn.Linear(8, h // 2)
        self.teammate = nn.Linear(2, h // 4)
        self.encoder = nn.Sequential(nn.Linear(h + h // 4, h), self.activation,
                                     nn.Linear(h, h), self.activation)
        self.proposal_mean = nn.Linear(h, 2)
        self.proposal_log_std = nn.Linear(h, 2)
        self.action_encoder = nn.Sequential(nn.Linear(4, h // 4), self.activation)
        self.base_gate = nn.Linear(h + h // 4, 1) if baseline_initialization else None
        self.baseline_gate_bound = float(baseline_gate_bound)
        self.relation_encoder = nn.Sequential(nn.Linear(edge_dim, r), nn.Tanh(),
                                              nn.Linear(r, r), nn.Tanh())
        self.relation_query = nn.Linear(h, r)
        self.priority_head = nn.Sequential(nn.Linear(edge_dim, r), nn.Tanh(), nn.Linear(r, 1))
        self.coordination_encoder = nn.Sequential(nn.Linear(r + 1, c), nn.Tanh())
        self.adapter = nn.Sequential(nn.Linear(h + h // 4 + r + c, h // 2), nn.Tanh())
        self.gate_mean = nn.Linear(h // 2, 1)
        self.gate_log_std = nn.Linear(h // 2, 1)
        self.theta_space_gate = bool(theta_space_gate)
        self.censored_gate_likelihood = bool(censored_gate_likelihood)
        if self.censored_gate_likelihood and not self.theta_space_gate:
            raise ValueError('Censored probabilities require a clipped theta-space Normal.')
        self.prior_gate_mean = nn.Linear(h // 2, 1) if self.theta_space_gate else None
        self.compatibility_gain_raw = None
        self.compatibility_context_head = None
        self.compatibility_margin = float(compatibility_margin)
        self.compatibility_temperature = float(compatibility_temperature)
        self.compatibility_neighbor_temperature = float(compatibility_neighbor_temperature)
        self.compatibility_compact_risk = bool(compatibility_compact_risk)
        self.compatibility_task_priority = bool(compatibility_task_priority)
        self.compatibility_priority_temperature = float(compatibility_priority_temperature)
        self.compatibility_peer_intent = bool(compatibility_peer_intent)
        self.compatibility_mixture_points = int(compatibility_mixture_points)
        self.compatibility_joint_mixture = bool(compatibility_joint_mixture)
        self.compatibility_task_prediction = bool(compatibility_task_prediction)
        self.compatibility_task_penalty = float(compatibility_task_penalty)
        self.compatibility_terminal_prediction = bool(compatibility_terminal_prediction)
        self.prediction_horizon = float(prediction_horizon)
        self.prediction_terminal_buffer = float(prediction_terminal_buffer)
        self._mixture_assignments = {}
        self.prediction_distance_scale = float(prediction_distance_scale)
        self.apply(initialize)
        for output in (self.proposal_mean, self.proposal_log_std, self.gate_mean, self.gate_log_std):
            nn.init.orthogonal_(output.weight, .01)
        nn.init.constant_(self.proposal_log_std.bias, -.5)
        nn.init.constant_(self.gate_log_std.bias, -.5)
        if self.prior_gate_mean is not None:
            nn.init.zeros_(self.prior_gate_mean.weight)
            nn.init.zeros_(self.prior_gate_mean.bias)
            self.prior_gate_mean.requires_grad_(False)
            nn.init.constant_(self.gate_log_std.bias, math.log(.1))
        if compatibility_prior:
            self.enable_compatibility_prior(compatibility_gain, compatibility_compact_risk,
                                            compatibility_task_priority, compatibility_peer_intent)
        if compatibility_context_gain:
            self.enable_contextual_compatibility()

    def enable_compatibility_prior(self, gain=1., compact_risk=False, task_priority=False, peer_intent=False):
        if not self.theta_space_gate or self.compatibility_gain_raw is not None:
            raise ValueError('Enable this prior once on a theta-space actor.')
        if not math.isfinite(gain) or not 0 < gain < 100:
            raise ValueError('The initial compatibility gain must be in (0,100).')
        self.compatibility_compact_risk = bool(compact_risk)
        self.compatibility_task_priority = bool(task_priority)
        self.compatibility_peer_intent = bool(peer_intent)
        self.compatibility_gain_raw = nn.Parameter(self.gate_mean.weight.new_tensor(math.log(math.expm1(gain))))

    def enable_contextual_compatibility(self):
        """A zero-initialized factor of one preserves the existing policy exactly."""
        if self.compatibility_gain_raw is None or self.compatibility_context_head is not None:
            raise ValueError('Enable contextual gain once on an existing compatibility prior.')
        head = nn.Linear(self.gate_mean.in_features, 1).to(self.gate_mean.weight)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        self.compatibility_context_head = head

    def compatibility_responsibility(self, obs, edges):
        """Favor preserving the nearer interceptor using only local observations.

        Reconstruct the peer's attacker distance from its exchanged relative
        position, never from the peer's observation or a centralized critic.
        The two directed responsibilities sum to two for a consistent pair.
        """
        own = obs[..., 2]
        attacker = own.unsqueeze(-1) * torch.stack((torch.cos(obs[..., 3]), torch.sin(obs[..., 3])), dim=-1)
        peer = (attacker.unsqueeze(-2) - edges[..., 12:14] * self.prediction_distance_scale).norm(dim=-1)
        return 2. * torch.sigmoid((own.unsqueeze(-1) - peer) / self.compatibility_priority_temperature)

    def compatibility_signal(self, edges, mask, obs=None, anchor=None):
        """A bounded action contrast from fixed BB/BL/LB/LL predictions.

        This is a policy feature, not a collision label or an execution filter.
        Endpoint risk interpolation is a prior, not a dynamical prediction of
        the continuous mixed action or the peer's final gate sample.
        """
        if self.compatibility_mixture_points:
            if self.compatibility_joint_mixture:
                return self.joint_mixture_signal(edges, mask, obs, anchor)
            return self.mixture_compatibility_signal(edges, mask, obs, anchor)
        distance = edges[..., [0, 3, 6, 9]] * self.prediction_distance_scale
        deficit = (self.compatibility_margin - distance) / self.compatibility_temperature
        # A compact risk feature leaves safe candidate pairs alone. Keep the
        # original kernel explicit so old frozen checkpoints remain reproducible.
        risk = F.relu(deficit) if self.compatibility_compact_risk else F.softplus(deficit)
        if self.compatibility_peer_intent:
            if anchor is None:
                raise ValueError('Peer intent must be recomputed from the exchanged Actor observations.')
            # All receivers use the same current network and saved raw messages.
            # This one-pass anchor is computed before the physical risk residual;
            # it neither observes nor feeds back any final stochastic gate sample.
            peer = anchor.clamp(0., 1.).squeeze(-1).unsqueeze(-2)
            contrast = (1.-peer)*(risk[..., 0]-risk[..., 2]) + peer*(risk[..., 1]-risk[..., 3])
        else:
            contrast = .5 * (risk[..., 0] + risk[..., 1] - risk[..., 2] - risk[..., 3])
        if self.compatibility_task_priority:
            if obs is None:
                raise ValueError('Task-aware compatibility requires the saved local observation.')
            contrast = contrast * self.compatibility_responsibility(obs, edges)
        weights = torch.softmax((-distance.min(-1).values / self.compatibility_neighbor_temperature)
                                .masked_fill(~mask, -1e9), dim=-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return torch.tanh((weights * contrast).sum(-1, keepdim=True))

    def mixture_compatibility_signal(self, edges, mask, obs, anchor):
        """Local risk slope at intended mixed actions, with zero safe-state shift.

        Raw grid distances come from nominal WAMV rollouts. Bilinear lookup is
        an approximation between grid points, not a safety certificate. Gate
        anchors are recomputed under the current actor from exchanged context.
        """
        k = self.compatibility_mixture_points
        if anchor is None or edges.shape[-1] != 18+k*k:
            raise ValueError('Mixed prediction requires saved grid distances and current gate anchors.')
        distance = edges[..., 18:].reshape(*edges.shape[:-1], k, k) * self.prediction_distance_scale
        grid = torch.linspace(0., 1., k, device=edges.device, dtype=edges.dtype)
        theta = anchor.squeeze(-1).clamp(0., 1.)
        def weights(value):
            return (1.-(value.unsqueeze(-1)-grid).abs()*(k-1)).clamp_min(0.)
        peer_weights = weights(theta).unsqueeze(-3).unsqueeze(-2)
        own_distances = (distance*peer_weights).sum(-1)
        own_weights = weights(theta).unsqueeze(-2)
        current_distance = (own_distances*own_weights).sum(-1)
        risk = F.relu((self.compatibility_margin-own_distances)/self.compatibility_temperature)
        lower = (theta-1./(k-1)).clamp(0., 1.)
        upper = (theta+1./(k-1)).clamp(0., 1.)
        contrast = (risk*(weights(lower)-weights(upper)).unsqueeze(-2)).sum(-1)
        contrast = contrast / (upper-lower).unsqueeze(-1).clamp_min(1e-8)
        contrast = contrast * (current_distance < self.compatibility_margin)
        if self.compatibility_task_priority:
            contrast = contrast*self.compatibility_responsibility(obs, edges)
        neighbor_weights = torch.softmax((-current_distance/self.compatibility_neighbor_temperature)
                                         .masked_fill(~mask, -1e9), dim=-1)*mask
        neighbor_weights = neighbor_weights / neighbor_weights.sum(-1, keepdim=True).clamp_min(1e-8)
        return torch.tanh((neighbor_weights*contrast).sum(-1, keepdim=True))

    def joint_mixture_signal(self, edges, mask, obs, anchor):
        """Choose a compatible team blend closest to the original gate means.

        Candidate choices include each current anchor and a fixed thrust grid.
        This prior enters the gate distribution before sampling. Its discrete
        choice is piecewise constant; PPO still differentiates the actual
        conditional distribution, including the learned prior gain. Predicted
        clearance is nominal and cannot certify real collision avoidance.
        """
        k = self.compatibility_mixture_points
        expected = 18+k*k+(2*k if self.compatibility_task_prediction else 0)
        if self.compatibility_terminal_prediction:
            expected += 3*k*k+k
        if anchor is None or edges.shape[-1] != expected:
            raise ValueError('Joint planning requires saved mixed distances and current gate anchors.')
        theta = anchor.squeeze(-1).clamp(0., 1.)
        batch, n = theta.shape
        if (k+1)**n > 10000:
            raise ValueError('Joint prediction is limited to 10000 team combinations per transition.')
        grid = torch.linspace(0., 1., k, device=edges.device, dtype=edges.dtype)
        options = torch.cat((theta.unsqueeze(-1), grid.expand(batch, n, -1)), dim=-1)
        interpolation = (1.-(options.unsqueeze(-1)-grid).abs()*(k-1)).clamp_min(0.)
        distance = edges[..., 18:18+k*k].reshape(batch, n, n, k, k)*self.prediction_distance_scale
        table = torch.einsum('bixp,bijpq,bjyq->bijxy', interpolation, distance, interpolation)
        key = (n, k, str(edges.device))
        if key not in self._mixture_assignments:
            values = torch.arange(k+1, device=edges.device)
            self._mixture_assignments[key] = torch.cartesian_prod(*[values for _ in range(n)])
        assignments = self._mixture_assignments[key]
        choices = torch.stack([options[:, i, assignments[:, i]] for i in range(n)], dim=-1)
        prefix_table = capture_cutoff = None
        if self.compatibility_terminal_prediction:
            start = 18+k*k+(2*k if self.compatibility_task_prediction else 0)
            prefix = edges[..., start:start+3*k*k].reshape(batch, n, n, 3, k, k)*self.prediction_distance_scale
            prefix_table = torch.einsum('bixp,bijtpq,bjyq->bijtxy', interpolation, prefix, interpolation)
            times = (edges[..., -k:]*mask.unsqueeze(-1)).sum(-2)/mask.sum(-1, keepdim=True).clamp_min(1)
            # For an off-grid anchor require BOTH adjacent actions to predict
            # capture; averaging a capture time with "never" is unsafe.
            option_times = torch.where(interpolation > 1e-6, times.unsqueeze(-2), 0.).amax(-1)
            capture_cutoff = torch.stack([option_times[:, i, assignments[:, i]]
                                         for i in range(n)], -1).amin(-1)
            capture_cutoff = capture_cutoff + self.prediction_terminal_buffer/self.prediction_horizon
        priority = torch.ones_like(theta)
        if self.compatibility_task_priority:
            priority = priority + 4.*torch.softmax(-obs[..., 2]/self.compatibility_priority_temperature, dim=-1)
        score = ((choices-theta[:, None]).square()*priority[:, None]).sum(-1)
        for i in range(n):
            for j in range(i+1, n):
                clearance = table[:, i, j, assignments[:, i], assignments[:, j]]
                if prefix_table is not None:
                    for window in (2, 1, 0):
                        clearance = torch.where(capture_cutoff <= (window+1)/4.,
                            prefix_table[:, i, j, window, assignments[:, i], assignments[:, j]], clearance)
                risk = F.relu((self.compatibility_margin-clearance)/self.compatibility_temperature)
                score = score + 100.*risk.square()*mask[:, i, j, None]
        if self.compatibility_task_prediction:
            # Preserve nominal interception opportunities without using future
            # events. These forecasts never supply the task or breach labels.
            raw = (edges[..., 18+k*k:18+k*k+2*k]*mask.unsqueeze(-1)).sum(-2)
            raw = raw/mask.sum(-1, keepdim=True).clamp_min(1)
            metrics = raw.reshape(batch, n, 2, k)*self.prediction_distance_scale
            task_options = torch.einsum('bimk,bilk->biml', metrics, interpolation)
            proposed = torch.stack([task_options[:, i, :, assignments[:, i]] for i in range(n)], -1).amin(-1)
            reference = task_options[..., 0].amin(1)
            deficit = F.relu(proposed-reference.unsqueeze(-1))
            score = score + self.compatibility_task_penalty*(deficit[:, 0].square()+.25*deficit[:, 1].square())
        chosen = assignments[score.argmin(-1)]
        planned = options.gather(-1, chosen.unsqueeze(-1)).squeeze(-1)
        # Retain the exact latent mean when keeping the original action,
        # including its endpoint probability mass in censored-gate mode.
        return torch.where(chosen == 0, torch.zeros_like(theta), planned-anchor.squeeze(-1)).unsqueeze(-1)

    def encode(self, obs):
        local = obs[..., :14] / self.local_scale
        neighbors = obs[..., 14:].reshape(*obs.shape[:-1], -1, 2) / self.neighbor_scale
        if neighbors.shape[-2]:
            teammate = self.activation(self.teammate(neighbors)).mean(dim=-2)
        else:
            teammate = obs.new_zeros(*obs.shape[:-1], self.teammate.out_features)
        return self.encoder(torch.cat((self.activation(self.f1(local[..., :6])),
                                       self.activation(self.f2(local[..., 6:])), teammate), dim=-1))

    def proposal_distribution(self, obs, encoded=None):
        h = self.encode(obs) if encoded is None else encoded
        return Normal(self.proposal_mean(h), self.proposal_log_std(h).clamp(-5., 1.).exp())

    def relation_features(self, h, edges, mask):
        # Additional physical grid features do not change inherited encoders.
        edges = edges[..., :18]
        embedded = self.relation_encoder(edges)
        score = (self.relation_query(h).unsqueeze(-2) * embedded).sum(-1) / math.sqrt(embedded.shape[-1])
        # A masked softmax that also works for a boat with no valid neighbors.
        weights = torch.softmax(score.masked_fill(~mask, -1e9), dim=-1) * mask
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        z_pred = (weights.unsqueeze(-1) * embedded).sum(-2)
        priority = self.priority_head(edges).squeeze(-1)
        eta = torch.sigmoid(priority - priority.transpose(-1, -2))
        coordinate = self.coordination_encoder(torch.cat((embedded, (eta - .5).unsqueeze(-1)), dim=-1))
        z_coord = (weights.unsqueeze(-1) * coordinate).sum(-2)
        return z_pred, z_coord, eta

    def gate_distribution(self, obs, proposals, boids, edges, mask, encoded=None, gate_scale=1., return_anchor=False):
        h = self.encode(obs) if encoded is None else encoded
        z_pred, z_coord, eta = self.relation_features(h, edges, mask)
        actions = self.action_encoder(torch.cat((proposals, boids), dim=-1))
        features = self.adapter(torch.cat((h, actions, z_pred, z_coord), dim=-1))
        mean = self.gate_mean(features)
        inherited = torch.zeros_like(mean)
        if self.base_gate is not None:
            inherited = self.base_gate(torch.cat((actions, h), dim=-1))
            if self.baseline_gate_bound:
                inherited = inherited.clamp(-self.baseline_gate_bound, self.baseline_gate_bound)
        if self.theta_space_gate:
            # Retain the transferred decision as an anchor, while the new head
            # learns a residual in physical theta units through PPO log-prob.
            mean = mean + torch.sigmoid(inherited + self.prior_gate_mean(features))
        else:
            mean = mean + inherited
        anchor = mean
        if self.compatibility_gain_raw is not None:
            gain = F.softplus(self.compatibility_gain_raw)
            if self.compatibility_context_head is not None:
                gain = gain * (2. * torch.sigmoid(self.compatibility_context_head(features)))
            mean = mean + gain * self.compatibility_signal(edges, mask, obs, mean)
        std = self.gate_log_std(features).clamp(-5., 1.).exp()
        # The curriculum can use a known exploration context. Both collection
        # and PPO must condition on the same saved scale; it is not a label or
        # an importance correction from a different behavior policy.
        if isinstance(gate_scale, (int, float)):
            if not math.isfinite(gate_scale) or gate_scale <= 0:
                raise ValueError('Gate exploration scale must be finite and positive.')
            std = std * gate_scale
        else:
            scale = torch.as_tensor(gate_scale, dtype=std.dtype, device=std.device)
            if (scale.numel() not in (1, len(obs)) or not torch.isfinite(scale).all() or (scale <= 0).any()):
                raise ValueError('Expected one positive gate scale per team transition.')
            std = std * scale.reshape(-1, 1, 1)
        distribution = Normal(mean, std)
        return (distribution, eta, anchor) if return_anchor else (distribution, eta)

    def initialize_from_baseline(self, state, proposal_std=.05, gate_std=.05):
        """Transfer the deterministic SAC actor exactly; subsequent learning is PPO.

        sigmoid(2*x) == .5*tanh(x)+.5. The new prediction Adapter initially adds
        zero to this logit, so initialization has no distillation approximation.
        Neither SAC critics nor SAC optimizer state are imported.
        """
        if self.base_gate is None or min(proposal_std, gate_std) <= 0:
            raise ValueError('Enable baseline_initialization and positive exploration scales.')
        mapping = {'f1': self.f1, 'f2': self.f2, 't1': self.teammate,
                   'l1': self.encoder[0], 'l2': self.encoder[2],
                   'mean_layer': self.proposal_mean, 'a1': self.action_encoder[0]}
        with torch.no_grad():
            for prefix, module in mapping.items():
                module.load_state_dict({k: state[prefix + '.' + k] for k in ('weight', 'bias')})
            self.base_gate.weight.copy_(2 * state['adap_layer.weight'])
            self.base_gate.bias.copy_(2 * state['adap_layer.bias'])
            self.gate_mean.weight.zero_()
            self.gate_mean.bias.zero_()
            if self.prior_gate_mean is not None:
                self.prior_gate_mean.weight.zero_()
                self.prior_gate_mean.bias.zero_()
            for layer, std in ((self.proposal_log_std, proposal_std), (self.gate_log_std, gate_std)):
                layer.weight.zero_()
                layer.bias.fill_(math.log(std))

    @staticmethod
    def _draw(distribution, deterministic, generators):
        if deterministic:
            return distribution.mean
        if generators is None:
            return distribution.sample()
        if len(generators) != distribution.mean.shape[0]:
            raise ValueError('Each parallel world needs its own policy RNG.')
        return torch.stack([torch.normal(mean, std, generator=generator)
                            for mean, std, generator in
                            zip(distribution.mean, distribution.stddev, generators)])

    def sample_proposals(self, obs, deterministic=False, encoded=None, generators=None):
        distribution = self.proposal_distribution(obs, encoded)
        raw = self._draw(distribution, deterministic, generators)
        return torch.tanh(raw), raw, distribution.log_prob(raw).sum(-1)

    def sample_gates(self, obs, proposals, boids, edges, mask, deterministic=False, encoded=None, generators=None,
                     gate_scale=1.):
        distribution, eta = self.gate_distribution(obs, proposals, boids, edges, mask, encoded, gate_scale)
        raw = self._draw(distribution, deterministic, generators)
        gates = raw.clamp(0., 1.) if self.theta_space_gate else torch.sigmoid(raw)
        # Raw samples are retained in either mode. Censored likelihoods combine
        # all samples executing the same endpoint into that endpoint's mass.
        return gates, raw, self.gate_log_probability(distribution, raw), eta

    def gate_log_probability(self, distribution, raw):
        if not self.censored_gate_likelihood:
            return distribution.log_prob(raw).sum(-1)
        lower = torch.special.log_ndtr(-distribution.mean/distribution.stddev)
        upper = torch.special.log_ndtr((distribution.mean-1.)/distribution.stddev)
        density = distribution.log_prob(raw.clamp(0., 1.))
        return torch.where(raw <= 0., lower, torch.where(raw >= 1., upper, density)).sum(-1)

    def gate_entropy(self, distribution):
        if not self.censored_gate_likelihood:
            return distribution.entropy().sum(-1)
        mean, std = distribution.mean, distribution.stddev
        a, b = -mean/std, (1.-mean)/std
        log_p0, log_p1 = torch.special.log_ndtr(a), torch.special.log_ndtr(-b)
        p0, p1 = log_p0.exp(), log_p1.exp()
        inside = (1.-p0-p1).clamp(0., 1.)
        phi_a = torch.exp(-.5*a.square())/math.sqrt(2.*math.pi)
        phi_b = torch.exp(-.5*b.square())/math.sqrt(2.*math.pi)
        # Entropy relative to Lebesgue measure on (0,1) plus two endpoint atoms.
        # p*log(p) via log-CDF avoids undefined log(0) derivatives in tiny tails.
        entropy = -p0*log_p0-p1*log_p1 + inside*(std.log()+.5*math.log(2.*math.pi))
        entropy = entropy + .5*(inside+a*phi_a-b*phi_b)
        return entropy.sum(-1)

    def evaluate_actions(self, obs, proposals, boids, edges, mask, proposal_raw, gate_raw, gate_scale=1.):
        # Saved samples/messages are constants; current encodings stay differentiable.
        # Both stages observe the same state. Reuse this graph within this call;
        # never cache detached features across PPO updates or environment steps.
        h = self.encode(obs)
        proposal = self.proposal_distribution(obs, h)
        gate, _ = self.gate_distribution(obs, proposals, boids, edges, mask, h, gate_scale)
        return (proposal.log_prob(proposal_raw).sum(-1), self.gate_log_probability(gate, gate_raw),
                proposal.entropy().sum(-1) + self.gate_entropy(gate))


class CentralizedCritic(nn.Module):
    def __init__(self, global_dim, hidden_dim=512, track_events=False, success_prior=.5, breach_prior=.05,
                 track_breach=False):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(global_dim, hidden_dim), nn.Tanh(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.Tanh())
        self.reward_head = nn.Linear(hidden_dim, 1)
        self.cost_head = nn.Linear(hidden_dim, 1)
        self.apply(initialize)
        self.success_head = self.breach_head = None
        if track_events:
            self.add_event_heads(success_prior, breach_prior)
        elif track_breach:
            self.add_breach_head(breach_prior)

    def add_event_heads(self, success_prior, breach_prior, preserve_breach=False):
        if self.success_head is not None or (self.breach_head is not None and not preserve_breach):
            raise ValueError('Event value heads already exist.')
        existing_breach = self.breach_head
        if existing_breach is not None:
            # Canonical success-before-breach registration also preserves the
            # parameter order expected by Adam when loading this checkpoint.
            del self.breach_head
        for name, prior in (('success_head', success_prior), ('breach_head', breach_prior)):
            if name == 'breach_head' and existing_breach is not None:
                self.breach_head = existing_breach
                continue
            head = nn.Linear(self.reward_head.in_features, 1).to(self.reward_head.weight.device)
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, prior)
            setattr(self, name, head)

    def add_breach_head(self, prior):
        if self.breach_head is not None:
            raise ValueError('The breach value head already exists.')
        self.breach_head = nn.Linear(self.reward_head.in_features, 1).to(self.reward_head.weight.device)
        nn.init.zeros_(self.breach_head.weight)
        nn.init.constant_(self.breach_head.bias, prior)

    def forward(self, state, with_events=False, with_breach=False):
        h = self.encoder(state)
        values = self.reward_head(h).squeeze(-1), self.cost_head(h).squeeze(-1)
        if with_events:
            if self.success_head is None:
                raise ValueError('Event value heads are not enabled.')
            values += (self.success_head(h).squeeze(-1), self.breach_head(h).squeeze(-1))
        elif with_breach:
            if self.breach_head is None:
                raise ValueError('The breach value head is not enabled.')
            values += (self.breach_head(h).squeeze(-1),)
        return values
