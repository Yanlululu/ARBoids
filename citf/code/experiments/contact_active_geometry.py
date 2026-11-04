"""Give the added contact-geometry attention a trainable nonzero start.

This controlled initialization experiment retains the joint physical response
objective, the original candidate attention, and the frozen teacher. It changes
only the initial gain of the added relative-velocity attention from zero to .25.
"""
from dataclasses import dataclass
import math
import sys

import torch

from experiments import contact_event_policy as contact
from experiments import contact_distribution_policy as distribution


@dataclass(frozen=True)
class ActiveGeometryConfig(distribution.ContactDistributionConfig):
    revision: str = 'CITF-contact-active-geometry-initialization-v1'
    geometry_attention_initial_gain: float = .25


class ActiveGeometryModel(contact.ContactEventModel):
    def __init__(self, cfg=ActiveGeometryConfig()):
        super().__init__(cfg)
        with torch.no_grad():
            self.capture_strength.fill_(math.atanh(cfg.geometry_attention_initial_gain))


_initialize_contact_worker = contact.initialize


def initialize_active_geometry_worker(project, source, config):
    contact.ContactEventConfig = ActiveGeometryConfig
    contact.ContactEventModel = ActiveGeometryModel
    _initialize_contact_worker(project, source, config)


def main():
    contact.ContactEventModel = ActiveGeometryModel
    distribution.ContactDistributionConfig = ActiveGeometryConfig
    distribution.initialize_distribution_worker = initialize_active_geometry_worker
    if len(sys.argv) <= 1 or sys.argv[1] not in ('prepare', 'train'):
        import citf.evaluate as evaluation
        from citf.contracts import canonical_hash, file_hash
        original = evaluation.runtime_identity

        def identity(*args, **kwargs):
            value = original(*args, **kwargs)
            value['runtime_code'] = canonical_hash(dict(
                core=value['runtime_code'], active_geometry=file_hash(__file__)))
            return value

        evaluation.runtime_identity = identity
    distribution.main()


if __name__ == '__main__':
    main()
