"""Candidate-conditioned coherent samples of the short stopped team motion.

Each predicted sample contains every vessel at every known short-horizon
time. The target at t is the observed position at min(t, actual stop): after
the observed stop it is the same measured endpoint, not an unobserved
continuation. Thus all outputs used for geometry have a supervised target.
Future stopping times and actions remain labels only.

The three matched simulator futures are an unordered set. One permutation
per root matches complete samples across all candidates, retaining the
common-random-number candidate response. Nonlinear team geometry is computed
within each coherent sample before taking the expectation.
"""
from dataclasses import dataclass
import itertools
import math

import torch
from torch import nn
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash
from citf.model import Predictor, Attention
from citf.risk import bounded_mean, pair_indices
from experiments import joint_task_geometry as geometry


@dataclass(frozen=True)
class StoppedMotionConfig(geometry.TaskGeometryConfig):
    revision: str = 'CITF-coherent-stopped-team-motion-v1'
    trajectory_hypotheses: int = 3
    trajectory_scale_m: float = 2.
    response_scale_m: float = .5
    response_weight: float = 2.
    pursuit_weight: float = .025
    trajectory_target: str = 'observed X(min(t, branch stop)); never future continuation'
    geometry_expectation: str = 'nonlinear synchronized geometry within each joint sample, then mean'


def stopped_positions(batch):
    xy, valid = batch['real_positions'], batch['time_valid']
    steps = xy.shape[-2]
    index = torch.arange(steps, device=xy.device)[None, None, None]
    index = torch.minimum(index, valid.sum(-1)[..., None] - 1)
    index = index[..., None, :, None].expand(*xy.shape[:-2], steps, 2)
    return xy.gather(-2, index)


def motion_geometry(paths, batch, clearance=7., margin=6., capture_radius=5.):
    # paths: root, candidate, coherent future, entity, time, coordinate.
    defenders = paths.shape[-3] - 1
    i, j = pair_indices(defenders, paths.device)
    known = batch['known_time_valid'][:, None, None, None]
    distance = (paths[:, :, :, i] - paths[:, :, :, j]).norm(dim=-1)
    distance = distance.masked_fill(~known, torch.inf).amin(-1)
    pair_valid = batch['entity_valid'][:, i] & batch['entity_valid'][:, j]
    collision = (clearance - distance).clamp_min(0).square()
    radius = paths.norm(dim=-1)
    radial = radius[:, :, :, :-1] - radius[:, :, :, -1:]
    radial = radial.masked_fill(~batch['entity_valid'][:, None, None, :-1, None], torch.inf).amin(-2)
    radial = radial.masked_fill(~known.squeeze(-2), -torch.inf).amax(-1)
    pursuit = (paths[:, :, :, :-1] - paths[:, :, :, -1:]).norm(dim=-1)
    pursuit = pursuit.masked_fill(~known, torch.inf).amin(-1)
    pursuit = pursuit.masked_fill(~batch['entity_valid'][:, None, None, :-1], torch.inf)
    team_pursuit = (pursuit.amin(-1) - capture_radius).clamp_min(0).square()
    return dict(distance=distance, collision=collision,
                radial=radial, radial_risk=(radial + margin).clamp_min(0).square(),
                pursuit=team_pursuit, pair_valid=pair_valid)


class StoppedMotionPredictor(Predictor):
    def __init__(self, cfg=StoppedMotionConfig()):
        super().__init__(cfg)
        del self.pair
        width = cfg.width
        self.order_embed = nn.Linear(3, width, bias=False)
        nn.init.zeros_(self.order_embed.weight)
        self.mode_embed = nn.Parameter(torch.randn(cfg.trajectory_hypotheses, width) * .2)
        self.mode_context = Attention(width, cfg.heads, relational=True)
        self.motion = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                    nn.GELU(), nn.Linear(width, 2 * cfg.horizon))
        nn.init.normal_(self.motion[-1].weight, std=.01)
        nn.init.zeros_(self.motion[-1].bias)

    def encode_history(self, batch):
        history = super().encode_history(batch)
        entities = history.shape[1]
        index = torch.arange(entities, device=history.device, dtype=history.dtype)
        defender = index < entities - 1
        code = torch.stack((index / max(1, entities - 2), index / 8,
                            defender.to(history.dtype)), -1) * defender[:, None]
        return history + self.order_embed(code)[None] * batch['entity_valid'][..., None]

    def forward(self, batch, history_cache=None):
        nominal, positions = batch['nominal'], batch['positions']
        valid, known = batch['entity_valid'], batch['known_time_valid']
        roots, candidates, entities, steps, _ = nominal.shape
        width, modes = self.cfg.width, self.cfg.trajectory_hypotheses
        history = self.encode_history(batch) if history_cache is None else history_cache
        displacement = ((nominal - positions[:, None, :, None]) / 10
                        * known[:, None, None, :, None])
        z = history[:, None] + self.candidate_embed(torch.cat(
            (batch['candidate'], displacement.flatten(-2)), -1))
        z = z.reshape(roots * candidates, entities, width)
        mask = valid[:, None].expand(roots, candidates, entities).reshape(-1, entities)
        paths = nominal.reshape(-1, entities, steps, 2)
        current = positions[:, None].expand(roots, candidates, entities, 2).reshape(-1, entities, 2)
        kt = known[:, None].expand(roots, candidates, steps).reshape(-1, steps)
        distances = (paths[:, None] - paths[:, :, None]).norm(dim=-1)
        nearest, when = distances.masked_fill(~kt[:, None, None], torch.inf).min(-1)
        last = (kt.sum(-1) - 1)[:, None, None, None].expand(-1, entities, entities, 1)
        last_distance = distances.gather(-1, last).squeeze(-1)
        delta = current[:, None] - current[:, :, None]
        edges = torch.cat((delta / 20, nearest[..., None] / 10,
            ((last_distance - delta.norm(dim=-1)) / 10)[..., None],
            (when / steps)[..., None]), -1)
        if self.cfg.fleet_candidate_context:
            for block in self.context:
                z = block(z, mask, edges)
        z = z[:, None] + self.mode_embed[None, :, None]
        z = z.reshape(roots * candidates * modes, entities, width)
        mode_mask = mask[:, None].expand(-1, modes, -1).reshape(-1, entities)
        if self.cfg.fleet_candidate_context:
            mode_edges = edges[:, None].expand(-1, modes, -1, -1, -1).reshape(-1, entities, entities, 5)
            z = self.mode_context(z, mode_mask, mode_edges)
        residual = torch.tanh(self.motion(z)).reshape(roots, candidates, modes, entities, steps, 2)
        time = torch.arange(1, steps + 1, device=nominal.device, dtype=nominal.dtype) * self.cfg.dt
        envelope = self.cfg.residual_scale * time.square() / 3
        motion = nominal[:, :, None] + residual * envelope[None, None, None, None, :, None]
        measurements = motion_geometry(motion, batch, self.cfg.clearance,
                                       self.cfg.guard_margin, self.cfg.pursuit_radius)
        collision = measurements['collision'].mean(2)
        pursuit = measurements['pursuit'].mean(2)
        return dict(stopped_motion=motion, xy=motion.mean(2),
            mean_distance=measurements['distance'].mean(2),
            risk=collision, pair_valid=measurements['pair_valid'],
            team_mu=measurements['radial'].mean(2),
            team_risk=measurements['radial_risk'].mean(2),
            # A team quantity is broadcast solely for the shared controller API.
            pursuit_risk=pursuit[..., None].expand(-1, -1, entities - 1),
            pursuit_weight=self.cfg.pursuit_weight, pursuit_radius=self.cfg.pursuit_radius,
            guard_weight=self.cfg.guard_weight, guard_margin=self.cfg.guard_margin,
            collision_source=self.cfg.collision_source,
            objective_risk_weight=self.cfg.objective_risk_weight,
            trajectory_scale_m=self.cfg.trajectory_scale_m,
            response_scale_m=self.cfg.response_scale_m,
            response_weight=self.cfg.response_weight)


def match_futures(prediction, target, valid):
    modes, futures = prediction.shape[2], target.shape[2]
    if modes != futures or modes > 4:
        raise ValueError('Joint stopped motion requires the registered matched small future set')
    pair = .5 * (prediction[:, :, :, None] - target[:, :, None]).square()
    mask = valid[:, :, None, None]
    # One permutation for every candidate at a root, not per-candidate matching.
    error = (pair * mask).sum((1, 4, 5, 6)) / mask.expand_as(pair).sum((1, 4, 5, 6)).clamp_min(1)
    permutations = torch.tensor(list(itertools.permutations(range(futures))), device=prediction.device)
    rows = torch.arange(modes, device=prediction.device)
    energy = torch.stack([error[:, rows, p].mean(-1) for p in permutations], -1)
    chosen = permutations[energy.detach().argmin(-1)]
    index = chosen[:, None, :, None, None, None].expand_as(target)
    return target.gather(2, index)


def stopped_motion_objective(out, batch, clearance=7., mode='full', risk_weight=1.,
                             rank_weight=.5, temperature=2., **kwargs):
    target = stopped_positions(batch)
    predicted = out['stopped_motion']
    cv, ev, known = batch['candidate_valid'], batch['entity_valid'], batch['known_time_valid']
    valid = cv[:, :, None, None, None] & ev[:, None, :, None, None] & known[:, None, None, :, None]
    matched = match_futures(predicted, target, valid)
    residual = predicted - matched
    # Fixed-scale Gaussian assignment score, separately from geometric regret.
    nll = bounded_mean(.5 * (residual / out['trajectory_scale_m']).square()
        + math.log(out['trajectory_scale_m'] * math.sqrt(2 * math.pi)), valid[:, :, None])
    motion = bounded_mean(F.smooth_l1_loss(predicted, matched, reduction='none'), valid[:, :, None])
    center = (residual * cv[:, :, None, None, None, None]).sum(1, keepdim=True) / cv.sum(-1)[:, None, None, None, None, None].clamp_min(1)
    contrast = bounded_mean(F.smooth_l1_loss((residual - center) / out['response_scale_m'],
        torch.zeros_like(residual), reduction='none'), valid[:, :, None])
    truth = motion_geometry(target, batch, clearance, out['guard_margin'], out['pursuit_radius'])
    collision = truth['collision'].mean(2)
    radial = truth['radial_risk'].mean(2)
    pursuit = truth['pursuit'].mean(2)
    risk = bounded_mean(F.smooth_l1_loss(out['risk'] / 49, collision / 49, reduction='none'),
                        cv[..., None] & truth['pair_valid'][:, None])
    risk = risk + bounded_mean(F.smooth_l1_loss(out['team_risk'] / 36, radial / 36, reduction='none'), cv)
    risk = risk + bounded_mean(F.smooth_l1_loss(out['pursuit_risk'][..., 0] / 100, pursuit / 100, reduction='none'), cv)
    physical = geometry.guard.physical_pair_cost(batch, clearance)
    empirical_collision = physical if out['collision_source'] == 'nominal' else (collision * truth['pair_valid'][:, None]).sum(-1)
    empirical = (batch['unary'] + out['objective_risk_weight'] * empirical_collision
        + out['guard_weight'] * radial + out['pursuit_weight'] * pursuit)
    estimated = geometry.costs(out, batch, clearance)
    target_prob = (-empirical / temperature).masked_fill(~cv, -torch.inf).softmax(-1).detach()
    logp = (-estimated / temperature).masked_fill(~cv, -torch.inf).log_softmax(-1)
    rank = -(target_prob * torch.where(cv, logp, torch.zeros_like(logp))).sum(-1).mean()
    total = nll
    if mode == 'full':
        total = total + risk_weight * risk + rank_weight * rank + out['response_weight'] * contrast
    return dict(total=total, motion=motion, nll=nll, risk=risk, rank=rank,
                context=contrast, empirical_cost=empirical, estimated_cost=estimated)


def main():
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = previous(*args, **kwargs)
        value['controller'].update(trajectory_target=StoppedMotionConfig().trajectory_target,
            geometry_expectation=StoppedMotionConfig().geometry_expectation,
            trajectory_hypotheses=StoppedMotionConfig().trajectory_hypotheses)
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'],
                                                    joint_stopped_motion=file_hash(__file__)))
        return value

    evaluation.runtime_identity = identity
    previous_hash = geometry.tree_hash
    geometry.tree_hash = lambda path: canonical_hash(dict(core=previous_hash(path),
                                                         joint_stopped_motion=file_hash(__file__)))
    geometry.TaskGeometryConfig, geometry.TaskGeometryPredictor = StoppedMotionConfig, StoppedMotionPredictor
    geometry.task_objective = stopped_motion_objective
    geometry.main()


if __name__ == '__main__':
    main()
