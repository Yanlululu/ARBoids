"""Evaluation-only per-link delay/loss emulation; local observations stay current."""
from collections import deque
import numpy as np
import torch
from policy.interaction_prediction import CandidateMessages


class MessageStress:
    def __init__(self, delay_steps=0, drop_probability=0., seed=0):
        if not isinstance(delay_steps, int) or delay_steps < 0 or not 0 <= drop_probability <= 1:
            raise ValueError('Invalid message impairment.')
        self.delay_steps = delay_steps
        self.drop_probability = drop_probability
        self.rng = np.random.default_rng(seed)
        self.history = deque(maxlen=delay_steps+1)
        self.received = None
        self.attempts = self.drops = 0
        self.maximum_age = 0.

    def receive(self, packet):
        """Reliable startup snapshot, then independent directed-link packet loss."""
        self.history.append({k: np.asarray(v).copy() for k, v in packet.items()})
        n = len(packet['states'])
        if self.received is None:
            self.received = [{k: v.copy() for k, v in packet.items()} for _ in range(n)]
        available = self.history[0]
        for i, cache in enumerate(self.received):
            for j in range(n):
                if i == j:
                    source = packet
                else:
                    self.attempts += 1
                    dropped = self.rng.random() < self.drop_probability
                    self.drops += int(dropped)
                    if dropped:
                        continue
                    source = available
                for key in cache:
                    cache[key][j] = source[key][j]
            self.maximum_age = max(self.maximum_age, float(packet['timestamps'].max()-cache['timestamps'].min()))
        return self.received

    @torch.no_grad()
    def act(self, agent, observations, states, boids, timestamp):
        if self.delay_steps == 0 and self.drop_probability == 0:
            return agent.act(observations, states, boids, timestamp, deterministic=True)[0]
        if agent.settings.prediction_disabled:
            return agent.act(observations, states, boids, timestamp, deterministic=True)[0]
        obs = np.asarray(observations, dtype=np.float32)
        proposals = agent.actor.sample_proposals(agent.tensor(obs).unsqueeze(0), True)[0][0].cpu().numpy()
        n = len(obs)
        packet = dict(states=np.asarray(states), boids=np.asarray(boids), proposals=proposals,
                      observations=obs, timestamps=np.full(n, timestamp))
        caches = self.receive(packet)
        action = np.zeros((n, 3))
        action[:, :2] = proposals
        for i, cache in enumerate(caches):
            # This explicit evaluation path intentionally permits unequal timestamps.
            # No extrapolation, hidden future state, retraining or execution-time shield.
            message = CandidateMessages(np.arange(n), cache['timestamps'], cache['states'], cache['boids'],
                                        cache['proposals'], cache['observations'] if agent.settings.compatibility_peer_intent else None)
            edges, mask = agent.predictor.features(message)
            gate_obs = agent.tensor(cache['observations']).unsqueeze(0)
            gate = agent.actor.sample_gates(gate_obs, agent.tensor(cache['proposals']).unsqueeze(0),
                        agent.tensor(cache['boids']).unsqueeze(0), agent.tensor(edges).unsqueeze(0),
                        agent.tensor(mask, True).unsqueeze(0), deterministic=True)[0]
            action[i, 2] = float(gate[0, i, 0])
        return action


def payload_bytes(n, peer_observations=True):
    # ID int64 + timestamp float64 + 6 state float64 + two 2-D float64 candidates.
    per_boat = 8 + 8 + 6*8 + 4*8 + (4*(14+2*(n-1)) if peer_observations else 0)
    return dict(broadcast_per_step=n*per_boat, unicast_per_step=n*(n-1)*per_boat,
                excludes_network_headers=True)
