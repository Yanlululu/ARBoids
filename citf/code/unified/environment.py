"""Batched common task from the 2026-11-06 unified manuscript.

WAM-V coefficients follow third_party/arboids_release/train/envs/modules.py.
Every event is checked after each 0.05 s integration step. No radial early loss,
legacy policy, learned teacher, or privileged future enters this environment.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch


@dataclass
class TaskConfig:
    defenders: int = 3
    duration: float = 60.0
    control_dt: float = 0.2
    integration_dt: float = 0.05
    target_radius: float = 15.0
    capture_radius: float = 5.0
    collision_radius: float = 5.0
    thrust_delta: float = 300.0
    agility: float = 2.0
    apf_noise: float = 0.3
    collision_penalty: float = 50.0

    def __post_init__(self):
        scales = (self.duration, self.control_dt, self.integration_dt, self.thrust_delta,
                  self.agility, self.target_radius, self.capture_radius, self.collision_radius,
                  self.collision_penalty)
        if self.defenders < 2 or any(not math.isfinite(value) or value <= 0 for value in scales):
            raise ValueError('The common task needs at least two defenders and positive time/action scales.')
        if not math.isfinite(self.apf_noise) or self.apf_noise < 0:
            raise ValueError('APF noise scale must be finite and nonnegative.')
        for ratio in (self.duration/self.control_dt, self.control_dt/self.integration_dt):
            if not math.isclose(ratio, round(ratio), rel_tol=0., abs_tol=1e-7):
                raise ValueError('Duration and control period must contain complete integration steps.')


def execute(nominal, previous, delta=300.0):
    low = (previous-delta).clamp(min=-500.0)
    high = (previous+delta).clamp(max=1000.0)
    return torch.maximum(torch.minimum(nominal, high), low)


def fusion(x, a, b, g, delta=300.0):
    """Supports [B,n] and [B,K,n] coefficients; one scalar per vessel."""
    previous = x[:, :-1, 6:8]
    if g.ndim == 3:
        a, b, previous = a[:, None], b[:, None], previous[:, None]
    return execute(b+g[..., None]*(a-b), previous, delta)


def force_to_thrust(force, yaw):
    direction = force/force.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    c, s = yaw.cos(), yaw.sin()
    longitudinal = c*direction[..., 0]+s*direction[..., 1]
    lateral = -s*direction[..., 0]+c*direction[..., 1]
    angle = torch.atan2(lateral, longitudinal)
    thrust = 1000*torch.stack((longitudinal+.75*angle, longitudinal-.75*angle), -1)
    thrust = thrust.clamp(-500, 1000)
    return torch.where((force.norm(dim=-1) > 1e-8)[..., None], thrust, 0.0)


def boids_components(x):
    state = x[..., :6]
    p, v = state[:, :-1, :2], state[:, :-1, 3:5]
    delta = p[:, :, None]-p[:, None, :]
    dist = delta.norm(dim=-1)
    neighbor = (dist > 1e-6) & (dist < 15.0)
    count = neighbor.sum(-1, keepdim=True).clamp_min(1)
    separation = 10*(delta/dist.clamp_min(1e-6)[..., None]*neighbor[..., None]).sum(2)
    alignment = .1*((v[:, None]*neighbor[..., None]).sum(2)/count-v)
    cohesion = .1*((p[:, None]*neighbor[..., None]).sum(2)/count-p)
    exists = neighbor.any(-1, keepdim=True)
    alignment = torch.where(exists, alignment, 0.0)
    cohesion = torch.where(exists, cohesion, 0.0)
    return separation, alignment, cohesion


def boids(x, components=None):
    separation, alignment, cohesion = boids_components(x) if components is None else components
    state = x[..., :6]
    force = separation+alignment+cohesion+.5*(state[:, -1:, :2]-state[:, :-1, :2])
    return force_to_thrust(force, state[:, :-1, 2])


def integrate(state, thrust, dt):
    p = state[..., :2]+dt*state[..., 3:5]
    # Recover nu_t using R(psi_t), then integrate BOTH nu and psi. Converting
    # the old world velocity with psi_{t+1} would apply the frame rotation
    # before the Coriolis terms and omit its matching world-frame transport.
    c, s = state[..., 2].cos(), state[..., 2].sin()
    u = c*state[..., 3]+s*state[..., 4]
    v = -s*state[..., 3]+c*state[..., 4]
    r = state[..., 5]
    du = (400*r*v-(100+150*u.abs())*u+thrust.sum(-1))/380
    dv = (-380*r*u-(100+150*v.abs())*v)/400
    dr = (-20*u*v-(980+950*r.abs())*r+1.25*(thrust[..., 0]-thrust[..., 1]))/1430
    u, v, r = u+dt*du, v+dt*dv, r+dt*dr
    yaw = torch.remainder(state[..., 2]+dt*state[..., 5], 2*math.pi)
    c, s = yaw.cos(), yaw.sin()
    return torch.cat((p, yaw[..., None], torch.stack((c*u-s*v, s*u+c*v, r), -1)), -1)


class FleetEnv:
    def __init__(self, count, cfg=None, seed=1, device='cuda', curriculum=False):
        if count < 1:
            raise ValueError('A vector environment needs at least one lane.')
        self.cfg = cfg or TaskConfig()
        self.count, self.n, self.device = count, self.cfg.defenders, torch.device(device)
        self.seed, self.curriculum = int(seed), curriculum
        self.horizon = round(self.cfg.duration/self.cfg.control_dt)
        self.substeps = round(self.cfg.control_dt/self.cfg.integration_dt)
        self.episodes = np.zeros(count, dtype=np.int64)
        self.state = torch.zeros(count, self.n+1, 6, device=self.device)
        self.previous = torch.zeros(count, self.n+1, 2, device=self.device)
        self.order = torch.zeros(count, self.n, dtype=torch.long, device=self.device)
        self.noise = torch.zeros(count, self.horizon, 2, device=self.device)
        self.agility = torch.ones(count, device=self.device)*self.cfg.agility
        self.tick = torch.zeros(count, dtype=torch.long, device=self.device)
        self.elapsed = torch.zeros(count, device=self.device)
        self.outcome = torch.zeros(count, dtype=torch.long, device=self.device)
        self.reset()

    @torch.no_grad()
    def reset(self, indices=None, transitions=0):
        ids = np.arange(self.count) if indices is None else indices.detach().cpu().numpy()
        if not len(ids):
            return
        states, orders, disturbances, agility = [], [], [], []
        mean_agility = 2.0+.25*min(3, int(transitions)//250000)
        for lane in ids:
            episode = int(self.episodes[lane])
            layout = np.random.default_rng(np.random.SeedSequence([self.seed, int(lane), episode, 11]))
            perturbation = np.random.default_rng(np.random.SeedSequence([self.seed, int(lane), episode, 23]))
            permutation = np.random.default_rng(np.random.SeedSequence([self.seed, int(lane), episode, 37]))
            state = np.zeros((self.n+1, 6), np.float32)
            azimuth, radius = layout.uniform(0, 2*np.pi), layout.uniform(60, 65)
            state[-1, :2] = radius*np.array([np.cos(azimuth), np.sin(azimuth)])
            state[-1, 2] = (azimuth+np.pi) % (2*np.pi)
            phases = layout.uniform(0, 2*np.pi)+np.arange(self.n)*2*np.pi/self.n
            radii = layout.uniform(7, 8, self.n)
            state[:-1, :2] = radii[:, None]*np.stack((np.cos(phases), np.sin(phases)), -1)
            state[:-1, 2] = layout.uniform(-np.pi, np.pi)+layout.uniform(-np.pi/3, np.pi/3, self.n)
            states.append(state)
            orders.append(permutation.permutation(self.n))
            disturbances.append(perturbation.normal(0, self.cfg.apf_noise, (self.horizon, 2)))
            agility.append(layout.uniform(mean_agility-.5, mean_agility+.5) if self.curriculum else self.cfg.agility)
            self.episodes[lane] += 1
        index = torch.as_tensor(ids, device=self.device)
        self.state[index] = torch.as_tensor(np.asarray(states), device=self.device)
        self.order[index] = torch.as_tensor(np.asarray(orders), device=self.device)
        self.noise[index] = torch.as_tensor(np.asarray(disturbances), dtype=torch.float32, device=self.device)
        self.agility[index] = torch.as_tensor(agility, device=self.device)
        self.previous[index] = 0
        self.tick[index] = 0
        self.elapsed[index] = 0
        self.outcome[index] = 0

    def observation(self):
        # State is in the episode-fixed generation order, including its rank.
        index = self.order[..., None]
        defenders = self.state[:, :-1].gather(1, index.expand(-1, -1, 6))
        previous = self.previous[:, :-1].gather(1, index.expand(-1, -1, 2))
        state = torch.cat((defenders, self.state[:, -1:]), 1)
        # The attacker's next rate-limited action also depends on its last command.
        previous = torch.cat((previous, self.previous[:, -1:]), 1)
        remaining = (1-self.elapsed/self.cfg.duration).clamp(0, 1)[:, None, None]
        return torch.cat((state, previous, remaining.expand(-1, self.n+1, 1)), -1)

    def executed_defender_action(self):
        return self.previous[:, :-1].gather(1, self.order[..., None].expand(-1, -1, 2))

    def replay_metadata(self):
        return dict(order=self.order, inverse_order=self.order.argsort(-1),
                    mask=torch.ones(self.count, self.n+1, dtype=torch.bool, device=self.device),
                    event=self.outcome, seconds=self.elapsed)

    def state_dict(self):
        result = dict(config=asdict(self.cfg), count=self.count, seed=self.seed,
                      curriculum=self.curriculum, episodes=self.episodes.copy())
        for name in ('state', 'previous', 'order', 'noise', 'agility', 'tick', 'elapsed', 'outcome'):
            result[name] = getattr(self, name).detach().cpu().clone()
        return result

    @torch.no_grad()
    def load_state_dict(self, saved):
        if saved['config'] != asdict(self.cfg) or saved['count'] != self.count:
            raise ValueError('A resume must retain the saved task and vector environment count.')
        self.seed, self.curriculum = saved['seed'], saved['curriculum']
        self.episodes = saved['episodes'].copy()
        for name in ('state', 'previous', 'order', 'noise', 'agility', 'tick', 'elapsed', 'outcome'):
            getattr(self, name).copy_(saved[name])

    def attacker(self):
        position = self.state[:, -1, :2]
        delta = position[:, None]-self.state[:, :-1, :2]
        distance = delta.norm(dim=-1).clamp_min(.1)
        support = distance < 15
        repel = 3000*(1/distance-1/15)*support/distance.pow(3)
        force = -.1*position+(repel[..., None]*delta).sum(1)
        noise = self.noise[torch.arange(self.count, device=self.device), self.tick.clamp_max(self.horizon-1)]
        nominal = force_to_thrust(force+noise, self.state[:, -1, 2])*self.agility[:, None]
        # APF and learned attackers share the single actuator mapping in step().
        return nominal

    @torch.no_grad()
    def step(self, ordered_action, attacker_action=None, active_mask=None):
        active_before = self.outcome == 0
        if active_mask is not None:
            active_before &= active_mask
        if not bool(torch.isfinite(ordered_action[active_before]).all()):
            raise FloatingPointError('Non-finite defender command; the episode cannot be scored as a timeout win.')
        action = torch.zeros_like(self.previous[:, :-1])
        action.scatter_(1, self.order[..., None].expand(-1, -1, 2), ordered_action)
        action = execute(action, self.previous[:, :-1], self.cfg.thrust_delta)
        attacker_action = self.attacker() if attacker_action is None else attacker_action
        if not bool(torch.isfinite(attacker_action[active_before]).all()):
            raise FloatingPointError('Non-finite attacker command.')
        attacker_action = execute(attacker_action/self.agility[:, None],
                                  self.previous[:, -1]/self.agility[:, None],
                                  self.cfg.thrust_delta)*self.agility[:, None]
        thrust = torch.cat((action, attacker_action[:, None]), 1)
        self.previous = torch.where(active_before[:, None, None], thrust, self.previous)
        eye = torch.eye(self.n, device=self.device, dtype=torch.bool)[None]
        collision_members = torch.zeros(self.count, self.n, dtype=torch.bool, device=self.device)
        for substep in range(self.substeps):
            active = (self.outcome == 0) & active_before
            advanced = integrate(self.state, thrust, self.cfg.integration_dt)
            self.state = torch.where(active[:, None, None], advanced, self.state)
            self.elapsed += active*self.cfg.integration_dt
            p = self.state[:, :-1, :2]
            distances = (p[:, :, None]-p[:, None, :]).norm(dim=-1)
            colliding = (distances <= self.cfg.collision_radius) & ~eye
            collision_members = colliding.any(-1)
            collision = collision_members.any(-1)
            breach = self.state[:, -1, :2].norm(dim=-1) <= self.cfg.target_radius
            capture = (p-self.state[:, -1:, :2]).norm(dim=-1).amin(-1) < self.cfg.capture_radius
            timeout = (self.tick+1 >= self.horizon) & (substep == self.substeps-1)
            event = torch.where(breach, 1, torch.where(collision, 2, torch.where(capture, 3, torch.where(timeout, 4, 0))))
            self.outcome = torch.where(active, event, self.outcome)
        if not bool(torch.isfinite(self.state).all()):
            raise FloatingPointError('Non-finite physical state; discard this failed run, not the failed episode.')
        self.tick += active_before.long()
        done = self.outcome != 0
        distance = (self.state[:, :-1, :2]-self.state[:, -1:, :2]).norm(dim=-1)
        individual = torch.zeros_like(distance)
        individual = torch.where((self.outcome == 1)[:, None], -100., individual)
        capture_reward = torch.where(distance < self.cfg.capture_radius, 100., torch.where(distance <= 15, 50., 0.))
        individual = torch.where((self.outcome == 3)[:, None], capture_reward, individual)
        individual = torch.where((self.outcome == 2)[:, None] & collision_members, -self.cfg.collision_penalty, individual)
        directions = self.state[:, -1:, :2]-self.state[:, :-1, :2]
        directions = directions/directions.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        summed = directions.sum(1)
        goal_to_attacker = self.state[:, -1, :2]
        goal_to_attacker = goal_to_attacker/goal_to_attacker.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        formation = .5*(goal_to_attacker*summed/summed.norm(dim=-1, keepdim=True).clamp_min(1e-8)).sum(-1)-summed.norm(dim=-1)/self.n
        individual += torch.where(done, 0., formation)[:, None]
        individual *= active_before[:, None]
        return self.observation(), individual.mean(-1), done, individual.gather(1, self.order)

    def records(self):
        return [dict(scene_seed=self.seed, scene_index=i, outcome=int(code),
                     success=int(code in (3, 4)), capture=int(code == 3), timeout=int(code == 4),
                     collision=int(code == 2), breach=int(code == 1), seconds=float(seconds))
                for i, (code, seconds) in enumerate(zip(self.outcome.cpu().tolist(), self.elapsed.cpu().tolist()))]
