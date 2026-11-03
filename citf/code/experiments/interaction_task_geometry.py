"""Learn third-vessel effects and motion behind complete task geometry.

Matched candidate pairs keep the queried pair's actions fixed while another
vessel changes its candidate. Their stopped-distance differences supervise
the candidate interaction itself, alongside valid-prefix motion and the
existing direct stopped-geometry likelihood and selection objective.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from citf.contracts import canonical_hash, file_hash
from citf.risk import bounded_mean
from experiments import joint_task_geometry as geometry


@dataclass(frozen=True)
class InteractionConfig(geometry.TaskGeometryConfig):
    revision: str = 'CITF-third-vessel-task-geometry-v1'
    auxiliary_motion_weight: float = 1.
    third_vessel_weight: float = 4.
    third_vessel_scale_m: float = .25
    interaction_target: str = 'stopped pair distance change with queried pair candidates fixed'


class InteractionPredictor(geometry.TaskGeometryPredictor):
    def forward(self, batch, history_cache=None):
        out = super().forward(batch, history_cache=history_cache)
        out.update(auxiliary_motion_weight=self.cfg.auxiliary_motion_weight,
                   third_vessel_weight=self.cfg.third_vessel_weight,
                   third_vessel_scale_m=self.cfg.third_vessel_scale_m)
        return out


ordinary_objective = geometry.task_objective


def interaction_objective(out, batch, mode='full', **kwargs):
    result = ordinary_objective(out, batch, mode=mode, **kwargs)
    real, tv = batch['real_positions'], batch['time_valid']
    ev, cv = batch['entity_valid'], batch['candidate_valid']
    count = tv.sum(2).clamp_min(1)
    target = (real * tv[:, :, :, None, :, None]).sum(2) / count[:, :, None, :, None]
    valid = ev[:, None, :, None, None] & tv.any(2)[:, :, None, :, None] & cv[:, :, None, None, None]
    motion = bounded_mean(F.smooth_l1_loss(out['xy'], target, reduction='none'), valid)
    predicted_relative = out['xy'][:, :, :-1] - out['xy'][:, :, -1:]
    actual_relative = target[:, :, :-1] - target[:, :, -1:]
    motion = motion + .5 * bounded_mean(F.smooth_l1_loss(predicted_relative, actual_relative,
                                                      reduction='none'), valid[:, :, :-1])
    result['motion'] = motion
    result['total'] = result['total'] + out['auxiliary_motion_weight'] * motion
    if mode == 'full':
        distance, _, _, _ = geometry.targets(batch)
        i, j = geometry.all_pairs(ev.shape[-1] - 1, ev.device)
        gates = batch['candidate'][..., :2]
        pair_gates = torch.cat((gates[:, :, i], gates[:, :, j]), -1)
        matched = (pair_gates[:, :, None] == pair_gates[:, None, :]).all(-1)
        changed = (gates[:, :, None] != gates[:, None, :]).any(-1).any(-1)
        matched = matched & changed[..., None] & cv[:, :, None, None] & cv[:, None, :, None]
        matched = matched & out['all_pair_valid'][:, None, None]
        error = out['all_mu'] - distance.mean(2)
        contrast = (error[:, :, None] - error[:, None, :]) / out['third_vessel_scale_m']
        contrast = bounded_mean(F.smooth_l1_loss(contrast, torch.zeros_like(contrast), reduction='none'), matched)
        result['context'] = contrast
        result['total'] = result['total'] + out['third_vessel_weight'] * contrast
    return result


def main():
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = previous(*args, **kwargs)
        value['controller']['interaction_target'] = InteractionConfig().interaction_target
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'], interaction_geometry=file_hash(__file__)))
        return value

    evaluation.runtime_identity = identity
    previous_hash = geometry.tree_hash
    geometry.tree_hash = lambda path: canonical_hash(dict(core=previous_hash(path),
                                                         interaction_geometry=file_hash(__file__)))
    geometry.TaskGeometryConfig, geometry.TaskGeometryPredictor = InteractionConfig, InteractionPredictor
    geometry.task_objective = interaction_objective
    geometry.main()


if __name__ == '__main__':
    main()
