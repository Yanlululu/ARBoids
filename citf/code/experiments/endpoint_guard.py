"""Learn radial coverage at the last observed time of a short branch.

The endpoint is the known prediction limit or an earlier actual stop in the
labels. Future stopping time never enters online inputs. This target retains
candidate-dependent progress that a worst-over-time margin can suppress when
the largest gap occurs before the candidate has materially changed motion.
"""
from dataclasses import dataclass

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import guard_prediction as guard
from experiments import synchronized_guard as team
from experiments import synchronized_delta as precise


TARGET = 'minimum defender signed radial gap at last valid short-branch time'


def nominal_endpoint(batch):
    radius = batch['nominal'].norm(dim=-1)
    gap = radius[:, :, :-1] - radius[:, :, -1:]
    gap = gap.masked_fill(~batch['entity_valid'][:, None, :-1, None], torch.inf)
    curve = gap.amin(-2)
    index = (batch['known_time_valid'].sum(-1) - 1)[:, None, None]
    index = index.expand(-1, curve.shape[1], 1)
    endpoint = curve.gather(-1, index).squeeze(-1)
    defenders = gap.shape[-2]
    return (curve[:, :, None].expand(-1, -1, defenders, -1),
            endpoint[:, :, None].expand(-1, -1, defenders))


def actual_endpoint(batch):
    radius = batch['real_positions'].detach().norm(dim=-1)
    gap = radius[:, :, :, :-1] - radius[:, :, :, -1:]
    gap = gap.masked_fill(~batch['entity_valid'][:, None, None, :-1, None], torch.inf)
    curve = gap.amin(-2)
    index = batch['time_valid'].sum(-1) - 1
    endpoint = curve.gather(-1, index[..., None]).squeeze(-1)
    return endpoint[..., None].expand(-1, -1, -1, gap.shape[-2])


@dataclass(frozen=True)
class EndpointConfig(precise.PreciseTeamConfig):
    revision: str = 'CITF-synchronized-first-action-endpoint-geometry-v1'
    geometric_target: str = TARGET


class EndpointPredictor(precise.PreciseTeamPredictor):
    pass


def main():
    team.nominal_team, team.actual_team = nominal_endpoint, actual_endpoint
    team.TARGET = TARGET
    precise.PreciseTeamConfig, precise.PreciseTeamPredictor = EndpointConfig, EndpointPredictor
    original_hash = guard.implementation_hash
    guard.implementation_hash = lambda path: canonical_hash({
        'guard': original_hash(path), 'endpoint_guard': file_hash(__file__),
        'geometric_target': TARGET})
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['method'] = ('CITF-EndpointGuard-development' if result['checkpoint_sha256']
                            else 'EndpointGuard-Phys-development')
        result['controller']['guard_target'] = TARGET
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
                                                 'endpoint_guard': file_hash(__file__)})
        return result

    evaluation.runtime_identity = identity
    precise.main()


if __name__ == '__main__':
    main()
