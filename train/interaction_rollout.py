"""Exact paper-protocol interventions; no online planning in deployment."""
import copy
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import multiprocessing as mp

import numpy as np
import torch

from envs.TADgame import TADEnv
from envs.snapshot import SimulationSnapshot, preserved_random_state
from policy.interaction_sac import InteractionActor, TwinTeamCritic, tensor_packet


def public_packet(env, observations=None):
    if observations is None:
        observations, _ = env._get_obs()
    return dict(obs=np.asarray(observations, dtype=np.float32).copy(),
                motion=np.asarray([[b.x, b.y, b.theta, *b.velocity] for b in env.defender_list],
                                  dtype=np.float64),
                central=env.centralized_state().copy())


@lru_cache(maxsize=8)
def safety_controller(defenders):
    from cbf_source_baseline import CBFController
    # CBFpy validates a new specialization using random example states.
    # A cold process (including resume) must not advance the training RNG.
    with preserved_random_state():
        return CBFController(defenders)


def execute(env, packet, action, *, safety=True, attacker_action=None):
    if safety:
        _, thrust, info = safety_controller(env.defender_num).control(np.asarray(packet['motion'], dtype=float), action, env.boids_actions)
    else:
        # Used only by unfiltered reference arms and explicit structural tests.
        thrust = action[:, 2:3] * (750. * action[:, :2] + 250.) + (1. - action[:, 2:3]) * env.boids_actions
        info = {}
    obs, reward, outcome, att_obs = env.step(action, 'AdaRes', att_action=attacker_action, defender_thrust=thrust)
    if env.LearningSide == 'Att':
        info['attacker_reward'] = float(reward)
        reward = np.sum(env.paper_reward_components(), axis=0)
    return public_packet(env, obs), reward, int(outcome), thrust, info, att_obs


def compact_snapshot(env):
    snapshot = SimulationSnapshot.capture(env)
    # These arrays are append-only output histories, not the Markov state.
    for name in ('Pos_Att', 'Phi_Att', 'Pos_Def', 'Phi_Def', 'Rewards'):
        value = getattr(snapshot.environment, name, None)
        if isinstance(value, np.ndarray) and value.ndim > 1:
            setattr(snapshot.environment, name, value[-1:].copy())
    return snapshot


class SnapshotPool:
    def __init__(self, capacity=256):
        self.capacity, self.items, self.count = int(capacity), [], 0

    def add(self, env):
        snapshot = compact_snapshot(env)
        if len(self.items) < self.capacity:
            self.items.append(snapshot)
        else:
            self.items[self.count % self.capacity] = snapshot
        self.count += 1

    def sample(self, size):
        if not self.items:
            raise ValueError('No intervention states.')
        return [self.items[i] for i in np.random.randint(len(self.items), size=size)]


def frozen_payload(agent, *, online_critic=False, attacker=None):
    if agent.legacy:
        raise ValueError('Intervention supervision requires the joint critic.')
    cpu_state = lambda model: {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return dict(config=copy.deepcopy(agent.config), actor=cpu_state(agent.actor),
                critic=cpu_state(agent.critic if online_critic else agent.target),
                alpha=float(agent.alpha), gamma=agent.gamma, attacker=attacker)


class FrozenPolicy:
    def __init__(self, payload):
        config = payload['config']
        hidden, relation = config['rl']['hidden_dim'], config['interaction']['relation_dim']
        self.actor = InteractionActor(hidden, relation,
            peer_candidates=config['interaction'].get('peer_candidates', True)).eval().requires_grad_(False)
        self.critic = TwinTeamCritic(hidden, relation).eval().requires_grad_(False)
        self.actor.load_state_dict(payload['actor'])
        self.critic.load_state_dict(payload['critic'])
        self.alpha, self.gamma = payload['alpha'], payload['gamma']
        self.safety = config['interaction'].get('safety', True)
        self.attacker = None
        if payload.get('attacker') is not None:
            from policy.networks import ActorAtt
            self.attacker = ActorAtt(2, 6, 2, hidden).eval().requires_grad_(False)
            self.attacker.load_state_dict(payload['attacker'])

    @torch.no_grad()
    def action(self, packet, generator):
        packet = {k: v.unsqueeze(0) for k, v in tensor_packet(packet).items()}
        noise = torch.randn((*packet['obs'].shape[:2], 3), generator=generator)
        action, logp = self.actor(packet['obs'], packet['motion'], noise=noise)
        return action[0].numpy(), float(logp.item())

    @torch.no_grad()
    def value(self, packet, generator):
        action, logp = self.action(packet, generator)
        p = {k: v.unsqueeze(0) for k, v in tensor_packet(packet).items()}
        q1, q2 = self.critic(p, torch.as_tensor(action).unsqueeze(0))
        return float(torch.minimum(q1, q2).item()) - self.alpha * logp

    @torch.no_grad()
    def attacker_action(self, env):
        if self.attacker is None:
            return None
        _, obs = env._get_obs()
        action, _ = self.attacker(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0),
                                  deterministic=True, with_logprob=False)
        return action[0].numpy()


def branch_return(policy, snapshot, first_action, future_seed, policy_seed, steps, *, trace=False, record_states=False):
    if record_states and not trace:
        raise ValueError('State records require a return trace.')
    env = snapshot.restore(future_seed=future_seed)
    generator = torch.Generator().manual_seed(policy_seed)
    # Both branches advance past the noise used to draw the common first action.
    torch.randn((1, env.defender_num, 3), generator=generator)
    packet, total, discount, done = public_packet(env), 0., 1., 0
    traces = []
    for k in range(steps):
        if k == 0:
            action, logp = first_action.copy(), 0.
        else:
            action, logp = policy.action(packet, generator)
        following, reward, done, thrust, _, _ = execute(env, packet, action, safety=policy.safety,
                                                        attacker_action=policy.attacker_action(env))
        total += discount * (float(np.mean(reward)) - policy.alpha * logp)
        if trace:
            traces.append(dict(action=action.copy(), thrust=thrust.copy(), motion=following['motion'].copy(),
                               reward=float(np.mean(reward)), logp=logp, outcome=done))
            if record_states:
                traces[-1]['packet'] = {key: value.copy() for key, value in packet.items()}
        discount *= policy.gamma
        packet = following
        if done:
            break
    if not done:
        total += discount * policy.value(packet, generator)
    return total, k + 1, traces


def intervention_pair(policy, snapshot, future_seed, policy_seed, boat, reference, steps, *, trace=False):
    if steps < 1 or reference not in (0., .5, 1.):
        raise ValueError('Positive horizon and reference in {0,.5,1} required.')
    with preserved_random_state():
        env = snapshot.restore(future_seed=future_seed)
        if env.protocol != 'paper-parameters-v1':
            raise ValueError('Cannot mix the old source protocol into intervention labels.')
        packet = public_packet(env)
        generator = torch.Generator().manual_seed(policy_seed)
        action, _ = policy.action(packet, generator)
        alternative = action.copy()
        alternative[boat, 2] = reference
        original, n1, t1 = branch_return(policy, snapshot, action, future_seed, policy_seed, steps, trace=trace)
        changed, n2, t2 = branch_return(policy, snapshot, alternative, future_seed, policy_seed, steps, trace=trace)
    result = {**packet, 'action': action, 'counterfactual': alternative,
              'return': np.asarray([original], dtype=np.float32),
              'counterfactual_return': np.asarray([changed], dtype=np.float32),
              'simulated_steps': n1 + n2, 'boat': boat, 'reference': reference}
    if trace:
        result['trace'] = (t1, t2)
    return result


def _worker_pairs(payload, jobs, steps):
    import study_runtime  # Set CPU limits before the first JAX initialization.
    torch.set_num_threads(1)
    policy = FrozenPolicy(payload)
    return [intervention_pair(policy, snapshot, future, noise, boat, ref, steps)
            for snapshot, future, noise, boat, ref in jobs]


class InterventionSampler:
    def __init__(self, workers=4):
        self.workers = int(workers)
        self.executor = None

    def generate(self, agent, pool, *, attacker=None, step=0):
        config = agent.config['interaction']
        with preserved_random_state():
            # Matched, independent seed stream across methods. Physical snapshots
            # and continuations belong to each method's own current policy.
            rng = np.random.default_rng([600000000, agent.config['training']['seed'], int(step)])
            snapshots = [pool.items[i] for i in rng.integers(len(pool.items), size=config['pairs_per_batch'])]
            jobs = [(s, int(rng.integers(1000000000,2000000000)), int(rng.integers(2000000000,3000000000)),
                     int(rng.integers(s.environment.defender_num)), float(rng.choice([0., .5, 1.])))
                    for s in snapshots]
            payload = frozen_payload(agent, attacker=attacker)
            steps = config['horizon_steps']
            if self.workers <= 0:
                rows = _worker_pairs(payload, jobs, steps)
            else:
                if self.executor is None:
                    self.executor = ProcessPoolExecutor(self.workers, mp_context=mp.get_context('spawn'))
                batches = [jobs[i::self.workers] for i in range(self.workers)]
                futures = [self.executor.submit(_worker_pairs, payload, part, steps) for part in batches if part]
                rows = [row for future in futures for row in future.result()]
        fields = ('obs', 'motion', 'central', 'action', 'counterfactual', 'return', 'counterfactual_return')
        return {k: np.stack([row[k] for row in rows]) for k in fields}, sum(r['simulated_steps'] for r in rows)

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None


class DeploymentPolicy:
    """Load only the deployment artifact. Never creates a critic or a rollout."""
    def __init__(self, checkpoint, device='cpu'):
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if saved.get('format') != 'interaction-sac-deployment-v1':
            raise ValueError('Expected an IA-CRRL deployment checkpoint.')
        self.config, self.device = saved['config'], torch.device(device)
        if self.config['environment']['protocol'] != 'paper-parameters-v1':
            raise ValueError('Checkpoint task protocol mismatch.')
        self.legacy = self.config['interaction']['arm'] == 'arboids_cbf'
        if self.legacy:
            from policy.networks import ActorAdap
            self.actor = ActorAdap(6, 8, 3, self.config['rl']['hidden_dim']).to(device)
        else:
            self.actor = InteractionActor(self.config['rl']['hidden_dim'],
                self.config['interaction']['relation_dim'],
                peer_candidates=self.config['interaction'].get('peer_candidates', True)).to(device)
        self.actor.load_state_dict(saved['actor'], strict=True)
        self.actor.eval().requires_grad_(False)
        self.last_action = None

    @torch.no_grad()
    def choose_action(self, packet, deterministic=True):
        obs = torch.as_tensor(packet['obs'], dtype=torch.float32, device=self.device)
        if self.legacy:
            action, _ = self.actor(obs, deterministic=deterministic, with_logprob=False)
        else:
            motion = torch.as_tensor(packet['motion'], dtype=torch.float32, device=self.device)
            action, _ = self.actor(obs.unsqueeze(0), motion.unsqueeze(0), deterministic=deterministic)
            action = action[0]
        return action.cpu().numpy(), 0.

    def control(self, observations, motion, boids_thrust, timestamp=0.):
        if not np.isfinite(timestamp):
            raise ValueError('Finite control timestamp required.')
        obs = np.asarray(observations, dtype=np.float32).copy()
        obs[:, 12:14] = (np.asarray(boids_thrust) - 250.) / 750.
        action, _ = self.choose_action(dict(obs=obs, motion=np.asarray(motion)))
        _, thrust, _ = safety_controller(len(action)).control(np.asarray(motion, dtype=float), action, boids_thrust)
        self.last_action = action.copy()
        return thrust
