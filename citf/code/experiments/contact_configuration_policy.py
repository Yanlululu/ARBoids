"""Closed-loop control ablation motivated by contact-root interventions.

Keep the frozen learned policy and use the unchanged strong physical choice
on first contact, while near the attacker, or after first contact.
This isolates control authority; it is not a newly trained predictor.
"""
from dataclasses import dataclass
import numpy as np
import torch

from citf.search import candidate_grid
from experiments import synchronized_configuration_policy as synchronized


@dataclass(frozen=True)
class ContactPolicyConfig(synchronized.SynchronizedPolicyConfig):
    revision: str = 'CITF-contact-authority-control-ablation-v1'
    contact_distance: float = 8.
    contact_mode: str = 'while_near'


class ContactPolicyController(synchronized.joint.JointPolicyController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.contact_seen = False

    @torch.no_grad()
    def act(self, world):
        state = world.state()
        distance = float(np.linalg.norm(state[:-1, :2] - state[-1, :2], axis=-1).min())
        near = distance <= self.model.cfg.contact_distance
        first_contact = near and not self.contact_seen
        self.contact_seen = self.contact_seen or near
        mode = self.model.cfg.contact_mode
        if mode not in ('while_near', 'after_contact', 'entry_pulse'):
            raise ValueError('Unknown contact-authority ablation')
        physical = (first_contact if mode == 'entry_pulse' else
                    near if mode == 'while_near' else self.contact_seen)
        force, trace = super().act(world)
        learned_indices = trace['selected_indices']
        if physical:
            indices = trace['physical_selected_indices']
            a, b, prior = self.source.commands(world.env)
            grid, _ = candidate_grid(prior)
            gate = grid[torch.arange(len(indices)), torch.as_tensor(indices)]
            force = (750 * (gate * a + (1 - gate) * b) + 250).numpy()
            trace.update(selected_indices=indices, gates=gate.tolist(), physical_thrust=force.tolist())
        trace.update(contact_mode=mode, contact_distance=self.model.cfg.contact_distance,
            observed_nearest_distance=distance, contact_seen=self.contact_seen,
            contact_physical_choice=physical, original_learned_indices=learned_indices,
            selection='declared contact authority on the unchanged original candidate bank')
        return force, trace


def main():
    synchronized.install_semantics()
    joint = synchronized.joint
    joint.JointPolicyConfig = ContactPolicyConfig
    joint.JointPolicyController = ContactPolicyController
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    original = evaluation.runtime_identity
    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        checkpoint = args[2] if len(args) > 2 else kwargs['checkpoint']
        config = torch.load(checkpoint, map_location='cpu', weights_only=False)['config']
        value['controller'].update(contact_mode=config['contact_mode'],
            contact_distance=config['contact_distance'], selection='declared contact authority ablation')
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'],
            contact_configuration_policy=file_hash(__file__)))
        return value
    evaluation.runtime_identity = identity
    joint.main()


if __name__ == '__main__':
    main()
