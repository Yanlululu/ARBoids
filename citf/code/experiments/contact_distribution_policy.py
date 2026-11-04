"""Joint response-distribution supervision for retained contact decisions.

Each future sample contains every current candidate branch. The energy score
therefore keeps common-future dependence without assigning modes from a nearly
unchanged reference trajectory. Labels remain stopped short physical paths.
"""
from dataclasses import dataclass
import sys

import torch

from experiments import contact_event_policy as contact
from experiments.candidate_response import response_objective, stopped_positions


@dataclass(frozen=True)
class ContactDistributionConfig(contact.ContactEventConfig):
    revision: str = 'CITF-retained-contact-joint-response-distribution-v1'
    physical_objective: str = 'joint-candidate-energy-reference3m-relative-response005m'


_initialize_contact_worker = contact.initialize


def initialize_distribution_worker(project, source, config):
    # Spawned workers import modules afresh and do not inherit main's bindings.
    contact.ContactEventConfig = ContactDistributionConfig
    _initialize_contact_worker(project, source, config)


def joint_response_energy(output, batch):
    prediction = output['xy']
    truth = stopped_positions(batch)[:, :, :prediction.shape[2]]
    known = batch['known_time_valid'][:, None, None, :, None]
    candidate_valid = batch['candidate_valid'].clone()
    candidate_valid[:, 0] = False
    distance = (batch['positions'][:, :-1] - batch['positions'][:, -1:]).norm(dim=-1)
    near = 1 + 3 * torch.exp(-(distance - 5).clamp_min(0) / 3)
    response_mask = (known[:, None] * candidate_valid[:, :, None, None, None, None]
                     * near[:, None, None, :, None, None])
    reference_scale = (known.sum((-1, -2, -3), keepdim=True)
                       * prediction.shape[3] * 2).clamp_min(1).sqrt()
    response_scale = (response_mask.sum((1, 3, 4, 5)) * 2).clamp_min(1).sqrt()

    def features(paths):
        reference = (paths[:, 0] * known / (3 * reference_scale)).flatten(2)
        relative = paths[..., :-1, :, :] - paths[..., -1:, :, :]
        response = relative - relative[:, :1]
        response = response * response_mask.sqrt() / .05
        response = response.permute(0, 2, 1, 3, 4, 5).flatten(2)
        response = response / response_scale[..., None]
        return torch.cat((.5 * reference, response), -1)

    predicted, actual = features(prediction), features(truth)
    cross = torch.cdist(predicted, actual).mean((1, 2))
    spread = torch.cdist(predicted, predicted).mean((1, 2))
    return (cross - .5 * spread).mean()


def distribution_objective(output, batch, cfg):
    result = response_objective(output, batch, cfg)
    energy = joint_response_energy(output, batch)
    if cfg.supervision == 'response':
        result['total'] = result['absolute'] + .1 * result['geometry'] + energy
    result['joint_response_energy'] = energy
    return result


def main():
    contact.ContactEventConfig = ContactDistributionConfig
    contact.contact_objective = distribution_objective
    contact.initialize = initialize_distribution_worker
    if len(sys.argv) <= 1 or sys.argv[1] not in ('prepare', 'train'):
        import citf.evaluate as evaluation
        from citf.contracts import canonical_hash, file_hash
        original = evaluation.runtime_identity

        def identity(*args, **kwargs):
            value = original(*args, **kwargs)
            value['runtime_code'] = canonical_hash(dict(
                core=value['runtime_code'], response_distribution=file_hash(__file__)))
            return value

        evaluation.runtime_identity = identity
    contact.main()


if __name__ == '__main__':
    main()
