
from pathlib import Path
import sys,numpy as np,torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'source/train'
sys.path.insert(0,str(TRAIN))
from policy.networks import ActorAdap
from envs.TADgame import TADEnv
CHECKPOINT=TRAIN/'experiments/author-source-fusion-20261011/two-hour-development-20261026/best-m1-400.pth'
EVIDENCE=TRAIN/'experiments/author-source-fusion-20261011/unified-civ-development-20261027/paper-evidence-20261027'


def mechanism_packet(env, obs, base):
    """Public measurements and the two unchanged author commands, in body frames."""
    motion = np.asarray([[b.x, b.y, b.theta, *b.velocity] for b in env.defender_list])
    p, v, heading = (motion[:, :2], motion[:, 3:5], motion[:, 2])
    c, s = (np.cos(heading), np.sin(heading))

    def local(vector):
        return np.stack((c * vector[:, 0] + s * vector[:, 1], -s * vector[:, 0] + c * vector[:, 1]), -1)
    n = len(p)
    candidates = np.concatenate((base[:, :2], (env.boids_actions - 250.0) / 750.0), -1)
    node = np.concatenate((local(-p) / 30.0, local(env.attacker.pos - p) / 30.0, local(np.broadcast_to(env.attacker.vel, (n, 2))) / 5.0, local(v) / 5.0, motion[:, 5:6], candidates, base[:, 2:3], np.full((n, 1), n / 6.0), np.full((n, 1), env.Current_T / 80.0)), axis=-1)
    dp = p[None] - p[:, None]
    dv = v[None] - v[:, None]
    rot = lambda x: np.stack((c[:, None] * x[..., 0] + s[:, None] * x[..., 1], -s[:, None] * x[..., 0] + c[:, None] * x[..., 1]), -1)
    distance = np.linalg.norm(dp, axis=-1)
    relative_heading = heading[None] - heading[:, None]
    geometry = np.concatenate((rot(dp) / 15.0, rot(dv) / 5.0, np.sin(relative_heading)[..., None], np.cos(relative_heading)[..., None], np.broadcast_to(motion[None, :, 5:6], (n, n, 1)), distance[..., None] / 15.0, ((dp * dv).sum(-1) / np.maximum(distance, 1.0))[..., None] / 5.0), -1)
    thrust = candidates.reshape(n, 2, 2) * 750.0 + 250.0
    u = local(v)[:, 0]
    acceleration = (thrust.sum(-1) - (100.0 + 150.0 * abs(u))[:, None] * u[:, None]) / 380.0
    yaw = 1.25 * (thrust[..., 0] - thrust[..., 1]) / 1430.0
    forward = np.stack((c, s), -1)
    side = np.stack((-s, c), -1)
    response = p[:, None] + 1.2 * v[:, None] + 0.72 * acceleration[..., None] * forward[:, None] + 0.3 * yaw[..., None] * side[:, None]

    def edges(peer):
        other = response if peer else np.broadcast_to((p + 1.2 * v)[:, None], response.shape)
        effects = []
        for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
            delta = other[None, :, b] - response[:, None, a]
            effects.extend((rot(delta) / 15.0, np.linalg.norm(delta, axis=-1)[..., None] / 15.0))
        return np.concatenate((geometry, np.broadcast_to(candidates[:, None], (n, n, 4)), np.broadcast_to((candidates if peer else np.zeros_like(candidates))[None], (n, n, 4)), *effects), -1)
    return {k: np.clip(value, -6.0, 6.0).astype(np.float32) for k, value in dict(node=node, edge=edges(True), edge_no_peer=edges(False), prior=base[:, 2]).items()}

def joint_packet(env, obs):
    packet = mechanism_packet(env, obs, np.zeros((len(obs), 3), dtype=np.float32))
    packet['obs'] = obs.astype(np.float32)
    packet['motion'] = np.asarray([[b.x, b.y, b.theta, *b.velocity] for b in env.defender_list], dtype=np.float32)
    packet['water'] = np.asarray([b.velocity_r for b in env.defender_list], dtype=np.float32) / 5.0
    a = env.attacker
    packet['task'] = np.asarray([a.x / 60.0, a.y / 60.0, np.sin(a.theta), np.cos(a.theta), *a.velocity_r / 5.0, a.agility / 3.0, env.Current_T / env.Total_T], dtype=np.float32)
    if hasattr(env, '_conditional_option_count'):
        packet['_conditional_count'] = np.asarray([env._conditional_option_count], dtype=np.float32)
    return packet

def mechanism_tensor(packet, device='cpu'):
    return {k: torch.as_tensor(value, device=device, dtype=torch.float32).unsqueeze(0) for k, value in packet.items()}

def mechanism_stack(packets, device):
    return {k: torch.tensor(np.stack([p[k] for p in packets]), device=device) for k in packets[0]}

def outcome(env, done):
    return dict(success=int(done > 2), capture=int(done == 3), collision=int(done == 2), source_failure=int(done == 1), timeout=int(done == 4), physical_breach=int(np.linalg.norm(env.attacker.pos) < env.Target_R), duration=float(env.Current_T), outcome=int(done))

def actuator_candidate_grid(preferred, subdivisions=1, *, include_uniform_midpoint=False):
    """Resolve each side of the source preference without changing its hull."""
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError('Candidate subdivisions must be a positive integer.')
    if not isinstance(include_uniform_midpoint, bool):
        raise ValueError('The common midpoint option must be boolean.')
    if subdivisions == 1:
        values = torch.stack((torch.zeros_like(preferred), preferred, torch.ones_like(preferred)), -2)
    else:
        fraction = torch.arange(subdivisions, device=preferred.device, dtype=preferred.dtype) / subdivisions
        lower = preferred[..., None, :] * fraction[:, None]
        upper = preferred[..., None, :] + (1.0 - preferred[..., None, :]) * fraction[:, None]
        values = torch.cat((lower, upper, torch.ones_like(preferred[..., None, :])), -2)
    if include_uniform_midpoint:
        values = torch.cat((values, torch.full_like(preferred[..., None, :], 0.5)), -2)
    levels = values.shape[-2]
    digits = torch.arange(levels * levels, device=preferred.device)
    return torch.stack((values[..., digits // levels, 0], values[..., digits % levels, 1]), -1)

def refine_pair_blocks(unary, risk, plans, risk_weight, sweeps=1):
    """Optimize two vessels together while retaining every other fleet choice."""
    plans = plans.clone()
    b, n, k = unary.shape
    for _ in range(sweeps):
        previous = plans.clone()
        for first in range(n):
            for second in range(first + 1, n):
                first_cost = unary[:, first, None].expand(-1, plans.shape[1], -1).clone()
                second_cost = unary[:, second, None].expand(-1, plans.shape[1], -1).clone()
                for other in range(n):
                    if other in (first, second):
                        continue
                    choices = plans[:, None, :, other].expand(-1, k, -1)
                    first_cost += risk_weight * risk[:, first, other].gather(2, choices).transpose(1, 2)
                    second_cost += risk_weight * risk[:, second, other].gather(2, choices).transpose(1, 2)
                cost = first_cost[..., None] + second_cost[..., None, :] + risk_weight * risk[:, first, second, None]
                flat = cost.flatten(-2)
                choice = flat.argmin(-1)
                current = plans[:, :, first] * k + plans[:, :, second]
                better = flat.gather(2, choice[..., None]).squeeze(-1) < flat.gather(2, current[..., None]).squeeze(-1)
                plans[:, :, first] = torch.where(better, choice // k, plans[:, :, first])
                plans[:, :, second] = torch.where(better, choice % k, plans[:, :, second])
        if torch.equal(previous, plans):
            break
    return plans

def pareto_fleet_refinement(unary, risk, plans, reference, risk_weight, componentwise=False):
    """Replace a fleet plan only when both control deviation and risk improve."""
    b, n, k = unary.shape
    catalog = torch.cat((reference[:, None], plans), 1)
    deviation = unary.new_zeros(catalog.shape[:2])
    interaction = torch.zeros_like(deviation)
    protected = torch.ones_like(deviation, dtype=torch.bool)
    for vessel in range(n):
        own = unary[:, vessel].gather(1, catalog[:, :, vessel])
        deviation += own
        if componentwise:
            protected &= own <= own[:, :1]
        for other in range(vessel):
            code = catalog[:, :, other] * k + catalog[:, :, vessel]
            pair = risk[:, other, vessel].flatten(1).gather(1, code)
            interaction += pair
            if componentwise:
                protected &= pair <= pair[:, :1]
    score = deviation + risk_weight * interaction
    admissible = (deviation <= deviation[:, :1]) & (interaction <= interaction[:, :1]) & (score < score[:, :1]) & protected
    admissible[:, 0] = True
    selected = score.masked_fill(~admissible, float('inf')).argmin(-1)
    return catalog[torch.arange(b, device=unary.device), selected]

def public_return_context(packet, return_order=False, version=1, reference_action=None):
    """Canonical public fleet state for factual continuation-value support."""
    motion = packet['motion']
    heading = motion[..., 2]
    c, s = (heading[:, 0].cos(), heading[:, 0].sin())
    relative = packet['node'][:, 0, 2:4] * 30.0
    target = motion[:, 0, :2] + torch.stack((c * relative[:, 0] - s * relative[:, 1], s * relative[:, 0] + c * relative[:, 1]), -1)
    direction = -target / target.norm(dim=-1, keepdim=True).clamp_min(1.0)

    def rotate(vector):
        return torch.stack(((vector * direction[:, None]).sum(-1), direction[:, None, 0] * vector[..., 1] - direction[:, None, 1] * vector[..., 0]), -1)
    relative = rotate(motion[..., :2] - target[:, None]) / 30.0
    body = rotate(torch.stack((heading.cos(), heading.sin()), -1))
    rows = torch.cat((relative, rotate(motion[..., 3:5]) / 3.0, body, motion[..., 5:6], motion[..., :2].norm(dim=-1, keepdim=True) / 30.0), -1)
    if version >= 3:
        if reference_action is None:
            raise ValueError('Control-conditioned values require the current predecessor action.')
        action = reference_action
        first = action[..., 3:4] if action.shape[-1] > 3 else torch.full_like(action[..., :1], -1.0)
        second = action[..., 5:6] if action.shape[-1] > 5 else first
        gate = torch.where(first >= 0.0, first, action[..., 2:3]) if version >= 4 else action[..., 2:3]
        controls = torch.cat((action[..., :2], packet['node'][..., 11:13], gate, first, second), -1)
        rows = torch.cat((rows, 0.25 * controls), -1)
    order = torch.atan2(relative[..., 1], relative[..., 0]).argsort(-1)
    context = rows.gather(1, order[..., None].expand_as(rows)).flatten(1)
    if version >= 2:
        velocity = packet['node'][:, 0, 4:6] * 5.0
        velocity = torch.stack((c * velocity[:, 0] - s * velocity[:, 1], s * velocity[:, 0] + c * velocity[:, 1]), -1)
        context = torch.cat((context, rotate(velocity[:, None])[:, 0] / 5.0, packet['node'][:, 0, 15:16]), -1)
    if version >= 3:
        held = (first[..., 0] >= 0.0).any(-1) | (second[..., 0] >= 0.0).any(-1) | (second[..., 0] == -2.0).all(-1)
        duration = torch.where(held, reference_action[:, 0, 4] / 20.0, torch.full_like(context[:, 0], 0.05)) if reference_action.shape[-1] > 4 else torch.full_like(context[:, 0], 0.05)
        context = torch.cat((context, duration[:, None]), -1)
    if version >= 5:
        switch = reference_action[:, 0, 8] if reference_action.shape[-1] >= 9 else torch.floor(reference_action[:, 0, 4] / 2.0) if reference_action.shape[-1] > 4 else torch.zeros_like(context[:, 0])
        context = torch.cat((context, (switch / 20.0)[:, None]), -1)
    return (context, order) if return_order else context


class CIActor(nn.Module):
 def __init__(self):
  super().__init__()
  r=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
  self.base=ActorAdap(6,8,3,512);self.base.load_state_dict({k[5:]:v for k,v in r['actor'].items() if k.startswith('base.')},strict=True)
  self.base.eval().requires_grad_(False);self.config=r['actuator_fusion_refinement'].copy()
  self.peer=True;self.levels=3;self.risk_horizon=r['risk_horizon'];self.risk_clearance=r['risk_clearance'];self.prediction_step=.1
  self.risk_feedback=True;self.interaction_enabled=True;self.pair_feedback_prediction=False;self.prediction_iterations=0
  self.current_marginal_prediction=False;self.conditional_enabled=False;self.prediction_relaxation=0.;self.terminal_prediction=False
  self.target_motion=None;self.interception_weight=0.;self.interception_horizon=0.;self.prediction_discount=0.;self.risk_barrier=0.
  self.environment_protocol=r.get('environment_protocol','source');self.eval()
 def source(self,packet):
  mean,_,hidden=self.proposal_distribution(packet['obs']);u=mean.tanh();rule=packet['node'][...,11:13]
  own=self.base.activation(self.base.a1(torch.cat((u,rule),-1)))
  raw=self.base.adap_layer(torch.cat((own,hidden),-1)).tanh()*.5+.5
  g=(2*torch.tanh(torch.logit(.02+.96*raw)/2)).sigmoid()
  return torch.cat((u,g),-1)


 @torch.no_grad()
 def candidate_risk(self, packet, proposals, return_distance=False, pairwise=False, reversal_at=None, candidate_grid=None, proposal_offsets=None, return_paths=False):
     motion = packet['motion']
     b, n = motion.shape[:2]
     actuator_grid = candidate_grid is not None and candidate_grid.ndim == 4
     k = self.levels if candidate_grid is None else candidate_grid.shape[-2 if actuator_grid else -1]
     if not self.interaction_enabled:
         if return_paths:
             return motion.new_zeros((b, n, k, round(self.risk_horizon / self.prediction_step), 2))
         return motion.new_zeros((b, n, n, k, k) if self.peer or pairwise else (b, n, k))
     if self.pair_feedback_prediction and (not return_paths):
         if candidate_grid is not None or proposal_offsets is not None or reversal_at is not None:
             raise ValueError('Pair feedback prediction requires the common fixed blending candidates.')
         return self.pair_feedback_risk(packet, proposals, return_distance)
     iterations = packet.get('_forecast_iteration', self.prediction_iterations)
     if iterations:
         previous = dict(packet, _forecast_iteration=iterations - 1)
         trajectories = self.candidate_risk(previous, proposals, reversal_at=reversal_at, candidate_grid=candidate_grid, proposal_offsets=proposal_offsets, return_paths=True)
         hidden = self.proposal_distribution(packet['obs'])[2]
         own = self.base.activation(self.base.a1(torch.cat((proposals, packet['node'][..., 11:13]), -1)))
         prior = 0.02 + 0.96 * (self.base.adap_layer(torch.cat((own, hidden), -1)).tanh().squeeze(-1) * 0.5 + 0.5)
         center = (2.0 * torch.tanh(torch.logit(prior) / 2.0)).sigmoid()
         reference_grid = torch.linspace(0.0, 1.0, k, device=motion.device) if candidate_grid is None else candidate_grid
         reference_grid = reference_grid[None, None] if reference_grid.ndim == 1 else reference_grid
         departure = ((reference_grid - center[..., None, None]) / 0.4).square().mean(-1) if actuator_grid else ((reference_grid - center[..., None]) / 0.4).square()
         weights = (-departure).softmax(-1)
         positions = (trajectories * weights[..., None, None]).sum(2)
         velocities = (positions[:, :, 1:] - positions[:, :, :-1]) / self.prediction_step
         velocities = torch.cat((velocities, velocities[:, :, -1:]), 2) if velocities.shape[2] else motion[:, :, None, 3:5]
         packet['_candidate_peer_reference'] = (positions, velocities)
     grid = torch.linspace(0.0, 1.0, k, device=motion.device) if candidate_grid is None else candidate_grid
     grid = grid[None, None] if grid.ndim == 1 else grid
     candidate = proposals[:, :, None] if proposal_offsets is None else (proposals[:, :, None] + proposal_offsets[None, None]).clamp(-1.0, 1.0)
     mixture = grid if actuator_grid else grid[..., None]
     force = 750.0 * (mixture * candidate + (1.0 - mixture) * packet['node'][:, :, None, 11:13]) + 250.0
     predicted_motion = self.prediction_motion(packet)
     x, y, heading, vx, vy, r = [predicted_motion[..., i, None].expand(-1, -1, k).clone() for i in range(6)]
     paths = []
     dt = self.prediction_step
     feedback_steps = max(1, round(0.2 / dt))
     for step in range(round(self.risk_horizon / dt)):
         if self.risk_feedback and step and (step % feedback_steps == 0):
             state = torch.stack((x, y, heading, vx, vy, r), -1).transpose(1, 2)
             forecast_obs, boids = self.feedback_observation(packet, state, step * dt)
             mean, _, hidden = self.proposal_distribution(forecast_obs.reshape(b * k, n, -1))
             future = mean.tanh().reshape(b, k, n, 2).transpose(1, 2)
             if self.conditional_enabled and '_proposal_offset' in packet:
                 future = (future + packet['_proposal_offset'][:, :, None]).clamp(-1.0, 1.0)
             if proposal_offsets is not None:
                 future = (future + proposal_offsets[None, None]).clamp(-1.0, 1.0)
             current_grid = 1.0 - grid if reversal_at is not None and step * dt >= reversal_at else grid
             mixture = current_grid if actuator_grid else current_grid[..., None]
             relaxation = getattr(self, 'prediction_relaxation', 0.0)
             if relaxation > 0.0:
                 own = self.base.activation(self.base.a1(torch.cat((future.transpose(1, 2), boids), -1)))
                 native = self.base.adap_layer(torch.cat((own, hidden.reshape(b, k, n, -1)), -1)).tanh() * 0.5 + 0.5
                 continuation = (2.0 * torch.tanh(torch.logit(0.02 + 0.96 * native) / 2.0)).sigmoid().transpose(1, 2)
                 weight = 1.0 - np.exp(-step * dt / relaxation)
                 relaxed = (1.0 - weight) * mixture + weight * continuation
                 committed = packet.get('_committed_gates')
                 mixture = torch.where((committed >= 0.0)[:, :, None, None], mixture, relaxed) if committed is not None else relaxed
             force = 750.0 * (mixture * future + (1.0 - mixture) * boids.transpose(1, 2)) + 250.0
         x = x + dt * vx
         y = y + dt * vy
         heading = heading + dt * r
         c, s = (heading.cos(), heading.sin())
         u = c * vx + s * vy
         v = -s * vx + c * vy
         du = (400.0 * r * v - (100.0 + 150.0 * u.abs()) * u + force.sum(-1)) / 380.0
         dv = (-380.0 * r * u - (100.0 + 150.0 * v.abs()) * v) / 400.0
         dr = (-20.0 * u * v - (980.0 + 950.0 * r.abs()) * r + 1.25 * (force[..., 0] - force[..., 1])) / 1430.0
         u = u + dt * du
         v = v + dt * dv
         vx = c * u - s * v
         vy = s * u + c * v
         r = r + dt * dr
         paths.append(torch.stack((x, y), -1))
     paths = torch.stack(paths, -2)
     if packet.get('_retain_candidate_paths', False):
         packet['_retained_candidate_paths'] = paths
     if return_paths:
         return paths
     if self.terminal_prediction:
         target, target_velocity = self.public_target_motion(packet)
         times = torch.arange(1, paths.shape[-2] + 1, device=motion.device) * dt
         target_path = torch.stack([self.target_motion.predict(packet, float(t))[0] for t in times], 1) if self.target_motion is not None else target[:, None] + times[None, :, None] * target_velocity[:, None]
         packet['_terminal_paths'] = (paths, target_path)
     if self.interception_weight:
         target, target_velocity = self.public_target_motion(packet)
         times = torch.arange(1, paths.shape[-2] + 1, device=motion.device) * dt
         target_path = torch.stack([self.target_motion.predict(packet, float(t))[0] for t in times], 1) if self.target_motion is not None else target[:, None] + times[None, :, None] * target_velocity[:, None]
         reach = (paths - target_path[:, None, None]).norm(dim=-1)
         if self.interception_horizon:
             reach = reach[..., :max(1, round(self.interception_horizon / dt))]
         packet['_candidate_interception'] = reach.amin(-1).clamp_min(5.0)
     if self.peer or pairwise:
         delta = paths[:, :, None, :, None] - paths[:, None, :, None]
         distance = delta.norm(dim=-1)
         minimum = distance.amin(-1)
         if return_distance:
             return minimum
         if self.prediction_discount:
             time = torch.arange(1, distance.shape[-1] + 1, device=motion.device) * dt
             weight = torch.exp(-(time - self.prediction_discount_start).clamp_min(0.0) / self.prediction_discount)
             return (self.distance_risk(distance) * weight).amax(-1)
         return self.distance_risk(minimum)
     time = torch.arange(1, len(paths[0, 0, 0]) + 1, device=motion.device) * dt
     constant = motion[:, :, :2][:, :, None] + time[None, None, :, None] * motion[:, :, 3:5][:, :, None]
     delta = paths[:, :, None] - constant[:, None, :, None]
     distance = delta.norm(dim=-1)
     minimum = distance.amin(-1)
     mask = ~torch.eye(n, dtype=torch.bool, device=motion.device)[None, :, :, None]
     if self.prediction_discount:
         weight = torch.exp(-(time - self.prediction_discount_start).clamp_min(0.0) / self.prediction_discount)
         return ((self.distance_risk(distance) * weight).amax(-1) * mask).amax(2)
     return (self.distance_risk(minimum) * mask).amax(2)

 def prediction_motion(self, packet):
     """Predict new-step motion after marginalizing the observed current.
 
         The source task draws a new zero-mean current at each control step.
         Water-relative velocity is persistent; the last ground-relative
         velocity also contains the previous transient disturbance.
         """
     motion = packet['motion']
     if not self.current_marginal_prediction:
         return motion
     return torch.cat((motion[..., :3], 5.0 * packet['water']), -1)

 def public_target_motion(self, packet):
     motion = packet['motion']
     c = motion[:, 0, 2].cos()
     s = motion[:, 0, 2].sin()
 
     def rotate(vector):
         return torch.stack((c * vector[:, 0] - s * vector[:, 1], s * vector[:, 0] + c * vector[:, 1]), -1)
     target = motion[:, 0, :2] + rotate(packet['node'][:, 0, 2:4] * 30.0)
     return (target, rotate(packet['node'][:, 0, 4:6] * 5.0))

 def distance_risk(self, minimum):
     risk = (self.risk_clearance - minimum).clamp_min(0.0).square()
     if self.risk_barrier:
         risk = risk + self.risk_barrier * F.softplus((5.5 - minimum) / 0.3).square()
     return risk

 def feedback_observation(self, packet, state, time, joint=False, target_override=None):
     """Original source observations at nominal candidate poses.
 
         Each candidate refreshes the unchanged source neural and Boids commands;
         neighbors follow public constant-velocity forecasts. These are predictor
         inputs only. No new controller or simulator-private state is executed.
         """
     motion = packet['motion']
     b, k, n = state.shape[:3]
     position = state[..., :2]
     velocity = state[..., 3:5]
     heading = state[..., 2]
     peers = motion[:, :, :2] + time * motion[:, :, 3:5]
     peer_velocity = motion[:, :, 3:5]
     reference = packet.get('_candidate_peer_reference')
     if reference is not None and (not joint):
         index = min(max(round(time / self.prediction_step) - 1, 0), reference[0].shape[2] - 1)
         peers = reference[0][:, :, index]
         peer_velocity = reference[1][:, :, index]
     relative = position[:, :, None] - position[:, :, :, None] if joint else peers[:, None, None] - position[:, :, :, None]
     distance = relative.norm(dim=-1)
     mask = ~torch.eye(n, dtype=torch.bool, device=state.device)[None, None]
     separation = (-relative / (distance - 3.5).clamp_min(0.01)[..., None].square() * ((distance < 15.0) & mask)[..., None]).sum(-2)
     alignment = velocity.mean(2, keepdim=True).expand_as(velocity) if joint else (peer_velocity.sum(1)[:, None, None] - peer_velocity[:, None] + velocity) / n
     cohesion = position.mean(2, keepdim=True) - position if joint else (peers.sum(1)[:, None, None] - peers[:, None] + position) / n - position
     h0 = motion[:, 0, 2]
     c0, s0 = (h0.cos(), h0.sin())
 
     def world(vector):
         return torch.stack((c0 * vector[:, 0] - s0 * vector[:, 1], s0 * vector[:, 0] + c0 * vector[:, 1]), -1)
     attacker = motion[:, 0, :2] + world(packet['node'][:, 0, 2:4] * 30.0)
     att_velocity = world(packet['node'][:, 0, 4:6] * 5.0)
     target = attacker + time * att_velocity
     if self.target_motion is not None and self.interaction_enabled and time:
         target, att_velocity = self.target_motion.predict(packet, time)
     if target_override is not None:
         target, att_velocity = target_override
         if target.shape not in ((b, 2), (b, k, 2)) or att_velocity.shape != target.shape:
             raise ValueError('A public feedback scenario needs one target state per batch or fleet plan.')
     target_position = target[:, None, None] if target.ndim == 2 else target[:, :, None]
     target_velocity = att_velocity[:, None, None] if att_velocity.ndim == 2 else att_velocity[:, :, None]
     attraction = target_position - position
     field = 0.5 * attraction + 10.0 * separation + 0.1 * alignment + 0.1 * cohesion
     field = field / field.norm(dim=-1, keepdim=True).clamp_min(1e-08)
     c, s = (heading.cos(), heading.sin())
     forward = c * field[..., 0] + s * field[..., 1]
     turn = torch.atan2(-s * field[..., 0] + c * field[..., 1], forward)
     thrust = (1000.0 * torch.stack((forward + 0.75 * turn, forward - 0.75 * turn), -1)).clamp(-500.0, 1000.0)
     boids = (thrust - 250.0) / 750.0
 
     def polar(vector):
         angle = torch.atan2(vector[..., 1], vector[..., 0]) - heading
         return torch.stack((vector.norm(dim=-1), torch.remainder(angle + np.pi, 2 * np.pi) - np.pi), -1)
     neighbor_angle = torch.atan2(relative[..., 1], relative[..., 0]) - heading[..., None]
     neighbor_polar = torch.stack((distance, torch.remainder(neighbor_angle + np.pi, 2 * np.pi) - np.pi), -1)
     neighbors = neighbor_polar[:, :, mask[0, 0]].reshape(b, k, n, 2 * (n - 1))
     obs = torch.cat((polar(-position), polar(attraction), polar(target_velocity.expand_as(position)), polar(separation), polar(alignment), polar(cohesion), boids, neighbors), -1)
     return (obs, boids)

 def proposal_distribution(self, obs):
     b, n, d = obs.shape
     net = self.base
     act = net.activation
     x = torch.cat((act(net.f1(obs[..., :6])), act(net.f2(obs[..., 6:14])), act(net.t1(obs[..., 14:].reshape(b, n, -1, 2))).mean(-2)), -1)
     h = act(net.l2(act(net.l1(x))))
     pm = net.mean_layer(h)
     ps = net.log_std_layer(h).clamp(-3.0, -0.3).exp()
     return (pm, ps, h)
@torch.no_grad()
def factorize(actor,packet):
 source=actor.source(packet);preferred=source[...,2:3].expand(-1,-1,2);grid=actuator_candidate_grid(preferred)
 p=dict(packet,_retain_candidate_paths=True)
 risk=actor.candidate_risk(p,source[...,:2],pairwise=True,candidate_grid=grid);paths=p['_retained_candidate_paths']
 _,order=public_return_context(packet,True,2);b,n,k=grid.shape[:3];batch=torch.arange(b)
 sorted_grid=grid.gather(1,order[...,None,None].expand_as(grid))
 center=preferred.gather(1,order[...,None].expand_as(preferred))
 risk=risk[batch[:,None,None],order[:,:,None],order[:,None,:]]
 unary=((sorted_grid-center[:,:,None])/.4).square().mean(-1)
 dup=(sorted_grid[:,:,:,None]-sorted_grid[:,:,None,:]).abs().amax(-1)<1e-6
 dup=(dup&torch.tril(torch.ones(k,k,dtype=torch.bool),-1)).any(-1)
 unary=unary.masked_fill(dup,float('inf'))
 return dict(source=source,grid=grid,grid_sorted=sorted_grid,unary=unary,risk=risk,order=order,paths=paths,weight=32.)

def score_plans(f,plans):
 unary,risk,weight=f['unary'],f['risk'],f['weight'];b,n,k=unary.shape
 score=unary.new_zeros(plans.shape[:2])
 for i in range(n):
  score+=unary[:,i].gather(1,plans[:,:,i])
  for j in range(i):score+=weight*risk[:,j,i].flatten(1).gather(1,plans[:,:,j]*k+plans[:,:,i])
 return score

@torch.no_grad()
def solve(f,beam_width=128):
 unary,risk,w=f['unary'],f['risk'],f['weight'];b,n,k=unary.shape
 scores=unary.new_zeros((b,1));plans=torch.empty((b,1,0),dtype=torch.long)
 for i in range(n):
  cost=scores[:,:,None]+unary[:,i,None]
  for j in range(i):cost=cost+w*risk[:,j,i].gather(1,plans[:,:,j,None].expand(-1,-1,k))
  scores,ix=cost.flatten(1).topk(min(beam_width,cost[0].numel()),largest=False)
  plans=torch.cat((plans.gather(1,(ix//k)[...,None].expand(-1,-1,i)),(ix%k)[...,None]),-1)
 beam=plans.clone();ref=plans[:,0].clone()
 for _ in range(3):
  for i in range(n):
   cost=unary[:,i].clone()
   for j in range(n):
    if j!=i:cost+=w*risk[:,i,j].gather(2,ref[:,j,None,None].expand(-1,k,1)).squeeze(-1)
   ref[:,i]=cost.argmin(-1)
 for _ in range(3):
  for i in range(n):
   cost=unary[:,i,None].expand(-1,plans.shape[1],-1).clone()
   for j in range(n):
    if j!=i:cost+=w*risk[:,i,j].gather(2,plans[:,None,:,j].expand(-1,k,-1)).transpose(1,2)
   plans[:,:,i]=cost.argmin(-1)
 plans=refine_pair_blocks(unary,risk,plans,w,1)
 selected=pareto_fleet_refinement(unary,risk,plans,ref,w)
 return dict(reference=selected,beam=beam,refined=plans,energies=score_plans(f,plans))

def physical_gates(f,plans):
 b,c,n=plans.shape
 gates=f['grid_sorted'][:,None].expand(-1,c,-1,-1,-1).gather(3,plans[...,None,None].expand(-1,-1,-1,1,2)).squeeze(3)
 return gates.gather(2,f['order'].argsort(-1)[:,None,:,None].expand(b,c,n,2))
