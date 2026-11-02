"""Physical pair queries with candidate-conditioned fleet cross-attention.

The prediction target remains the observed stopped minimum separation. Pair
queries expose the complete nominal separation curve, while attention reads
the candidate-dependent geometry of every other vessel. No event label or
unobserved learned trajectory tail enters the online risk.
"""
from dataclasses import dataclass
import math
import sys

import torch
from torch import nn
import torch.nn.functional as F

from citf.model import Config, Predictor
from citf.risk import pair_indices, rectified_gaussian_risk
from citf.contracts import canonical_hash, file_hash, tree_hash


@dataclass(frozen=True)
class PairGeometryConfig(Config):
    revision: str = 'CITF-pair-geometry-v1'


class PairContext(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qnorm = nn.LayerNorm(width)
        self.knorm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2 * width)
        self.bias = nn.Sequential(nn.Linear(5, width), nn.GELU(), nn.Linear(width, heads))
        self.output = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))

    def forward(self, pair, entity, valid, geometry):
        batch, pairs, width = pair.shape
        entities = entity.shape[1]
        query = self.query(self.qnorm(pair)).reshape(batch, pairs, self.heads, -1).transpose(1, 2)
        key, value = self.key_value(self.knorm(entity)).reshape(batch, entities, 2, self.heads, -1).permute(2, 0, 3, 1, 4)
        logits = query @ key.transpose(-1, -2) / math.sqrt(width // self.heads)
        logits = logits + self.bias(geometry).permute(0, 3, 1, 2)
        weight = logits.masked_fill(~valid[:, None, None], -torch.inf).softmax(-1)
        pair = pair + self.output((weight @ value).transpose(1, 2).reshape(batch, pairs, width))
        return pair + self.ff(self.norm(pair))


class PairGeometryPredictor(Predictor):
    def __init__(self, cfg=PairGeometryConfig()):
        super().__init__(cfg)
        if cfg.variant != 'candidate_graph':
            raise ValueError('Pair geometry uses the registered candidate graph backbone')
        del self.pair
        width = cfg.width
        self.pair_embed = nn.Sequential(
            nn.Linear(2 * width + 2 * cfg.horizon + 2, width), nn.GELU(), nn.Linear(width, width))
        self.pair_context = PairContext(width, cfg.heads)
        self.distance_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.zeros_(self.distance_head[-1].weight)
        nn.init.zeros_(self.distance_head[-1].bias)
        with torch.no_grad():
            self.distance_head[-1].bias[1] = -1.5

    def forward(self, b, history_cache=None):
        nominal, candidate, positions = b['nominal'], b['candidate'], b['positions']
        valid, known = b['entity_valid'], b['known_time_valid']
        batch, candidates, entities, steps, _ = nominal.shape
        width = self.cfg.width
        if steps != self.cfg.horizon or not valid[:, -1].all() or not known.any(-1).all():
            raise ValueError('Invalid physical pair query schema')
        history = self.encode_history(b) if history_cache is None else history_cache
        displacement = (nominal - positions[:, None, :, None]) / 10
        displacement = displacement * known[:, None, None, :, None]
        z = (history[:, None] + self.candidate_embed(torch.cat((candidate, displacement.flatten(-2)), -1))).reshape(batch * candidates, entities, width)
        mask = valid[:, None].expand(batch, candidates, entities).reshape(batch * candidates, entities)
        paths = nominal.reshape(batch * candidates, entities, steps, 2)
        current = positions[:, None].expand(batch, candidates, entities, 2).reshape(batch * candidates, entities, 2)
        known_flat = known[:, None].expand(batch, candidates, steps).reshape(batch * candidates, steps)
        distances = (paths[:, None] - paths[:, :, None]).norm(dim=-1)
        nearest, when = distances.masked_fill(~known_flat[:, None, None], torch.inf).min(-1)
        last_index = (known_flat.sum(-1) - 1)[:, None, None, None].expand(-1, entities, entities, 1)
        last_distance = distances.gather(-1, last_index).squeeze(-1)
        delta = current[:, None] - current[:, :, None]
        current_distance = delta.norm(dim=-1)
        edges = torch.cat((delta / 20, nearest[..., None] / 10,
                           ((last_distance - current_distance) / 10)[..., None], (when / steps)[..., None]), -1)
        if self.cfg.fleet_candidate_context:
            for block in self.context:
                z = block(z, mask, edges)
        z = z * mask[..., None]
        times = torch.arange(1, steps + 1, device=nominal.device, dtype=nominal.dtype) * self.cfg.dt
        envelope = (times - self.cfg.dt).clamp_min(0).square() / 3
        correction = self.cfg.residual_scale * torch.tanh(self.motion(z)).reshape(batch, candidates, entities, steps, 2)
        xy = nominal + correction * envelope[None, None, None, :, None] * valid[:, None, :, None, None]
        i, j = pair_indices(entities - 1, nominal.device)
        pair_count = len(i)
        curve = distances[:, i, j]
        initial = current_distance[:, i, j]
        closing = (curve - torch.cat((initial[..., None], curve[..., :-1]), -1)) / self.cfg.dt
        feature = torch.cat((z[:, i] + z[:, j], (z[:, i] - z[:, j]).abs(),
                             curve * known_flat[:, None] / 10, closing * known_flat[:, None] / 5,
                             initial[..., None] / 10, nearest[:, i, j, None] / 10), -1)
        pair = self.pair_embed(feature)
        # The matched ablation retains all historical state and all parameters,
        # but the context keys and geometric bias contain no other candidate.
        if self.cfg.fleet_candidate_context:
            keys = z
            geometric_distance = nearest
            geometric_time = when.to(nominal.dtype) / steps
        else:
            keys = history[:, None].expand(batch, candidates, entities, width).reshape(batch * candidates, entities, width)
            geometric_distance = current_distance
            geometric_time = torch.zeros_like(current_distance)
        di, dj = geometric_distance[:, i], geometric_distance[:, j]
        ti, tj = geometric_time[:, i], geometric_time[:, j]
        ids = torch.arange(entities, device=nominal.device)
        own = ((ids[None] == i[:, None]) | (ids[None] == j[:, None])).to(nominal.dtype)
        attacker = (ids == entities - 1).to(nominal.dtype)[None].expand(pair_count, -1)
        geometry = torch.stack((torch.minimum(di, dj) / 10, torch.maximum(di, dj) / 20,
                                torch.minimum(ti, tj), own[None].expand(batch * candidates, -1, -1),
                                attacker[None].expand(batch * candidates, -1, -1)), -1)
        pair = self.pair_context(pair, keys, mask, geometry)
        q = self.distance_head(pair).reshape(batch, candidates, pair_count, 2)
        nominal_mu = nearest[:, i, j].reshape(batch, candidates, pair_count)
        mu = nominal_mu + q[..., 0]
        sigma = (F.softplus(q[..., 1]) + .03).clamp_max(4)
        mean = mu * torch.special.ndtr(mu / sigma) + sigma * torch.exp(-.5 * (mu / sigma).square()) / math.sqrt(2 * math.pi)
        return dict(xy=xy, mu=mu, sigma=sigma,
                    risk=rectified_gaussian_risk(mu, sigma, self.cfg.clearance),
                    mean_distance=mean, det_risk=(self.cfg.clearance - mean).clamp_min(0).square(),
                    nominal_mu=nominal_mu, pair_valid=valid[:, i] & valid[:, j])


def implementation_hash(path):
    return canonical_hash({'core': tree_hash(path), 'pair_geometry': file_hash(__file__)})


def train_main():
    import citf.train as training
    training.Config = PairGeometryConfig
    training.Predictor = PairGeometryPredictor
    training.tree_hash = implementation_hash
    training.main()


def evaluate_main():
    import citf.collect as collection
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity
    def identity(*args, **kwargs):
        result = original_identity(*args, **kwargs)
        result['method'] = 'CITF-PairGeometry-development'
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'], 'pair_geometry': file_hash(__file__)})
        return result
    collection.Config = PairGeometryConfig
    collection.Predictor = PairGeometryPredictor
    evaluation.runtime_identity = identity
    evaluation.main()


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        train_main()
    else:
        evaluate_main()
