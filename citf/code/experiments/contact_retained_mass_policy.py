"""Preserve all joint proposals while controlling total prior replacement mass.

This matches the pair policy's retained-action prior without removing any
alternative. Ordinary complete-task policy gradients can change every logit.
"""
from dataclasses import dataclass

import numpy as np
import torch

from experiments import contact_event_policy as contact


@dataclass(frozen=True)
class RetainedMassConfig(contact.ContactEventConfig):
    revision: str = 'CITF-contact-full-bank-retained-mass-policy-v1'
    decision_domain: str = 'all-original-joint-proposals-with-fixed-total-alternative-prior'
    retained_prior_logodds: float = 1.5


class RetainedMassController(contact.ContactEventController):
    @torch.no_grad()
    def act(self, world):
        stochastic = self.stochastic
        self.stochastic = False
        try:
            force, trace = super().act(world)
        finally:
            self.stochastic = stochastic
        if self.transition is None:
            return force, trace
        values = self.transition['inputs']
        plans = np.asarray(trace['configurations'], dtype=np.int64)
        retained = next(i for i, p in enumerate(plans)
                        if p.tolist() == trace['retained_teacher_indices'])
        source_logits = values['teacher_logits'].copy()
        alternative = np.arange(len(plans)) != retained
        prior = np.zeros(len(plans), dtype=np.float32)
        if alternative.any():
            scores = torch.from_numpy(source_logits[alternative])
            prior[alternative] = (scores - torch.logsumexp(scores, dim=0)
                                  - self.model.cfg.retained_prior_logodds).numpy()
        values['teacher_logits'] = prior
        batch = {k: torch.from_numpy(v[None]).to(self.device) for k, v in values.items()}
        logits = self.model(batch)['logits'][0]
        distribution = torch.distributions.Categorical(logits=logits)
        choice = int(distribution.sample()) if stochastic else min(
            range(len(plans)), key=lambda i: (-float(logits[i]), i != retained, tuple(plans[i])))
        phases = self.transition['phases']
        self.transition = dict(inputs={k: v.copy() for k, v in values.items()},
            action=choice,
            old_logp=float(distribution.log_prob(torch.tensor(choice, device=logits.device))),
            changed=int(choice != retained), entropy=float(distribution.entropy()), phases=phases)
        from citf.search import candidate_grid
        learned, rules, prior_gate = self.source.commands(world.env)
        grid, _ = candidate_grid(prior_gate)
        gate = grid[torch.arange(len(plans[choice])), torch.as_tensor(plans[choice])]
        force = (750 * (gate * learned + (1 - gate) * rules) + 250).numpy()
        trace.update(scores=(-logits).cpu().tolist(), selected_indices=plans[choice].tolist(),
            gates=gate.tolist(), physical_thrust=force.tolist(), teacher_logits=prior.tolist(),
            source_teacher_logits=source_logits.tolist(), retained_proposal_index=retained,
            selection='all original proposals; fixed total alternative prior plus learned current-action logits')
        return force, trace


_initialize_contact_worker = contact.initialize


def initialize_retained_mass_worker(project, source, config):
    contact.ContactEventConfig = RetainedMassConfig
    contact.ContactEventController = RetainedMassController
    _initialize_contact_worker(project, source, config)


def main():
    contact.ContactEventConfig = RetainedMassConfig
    contact.ContactEventController = RetainedMassController
    contact.initialize = initialize_retained_mass_worker
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        value['controller']['prior_alternative_mass'] = 'exp(-1.5) relative to retained proposal'
        value['runtime_code'] = canonical_hash(dict(
            core=value['runtime_code'], retained_mass=file_hash(__file__)))
        return value

    evaluation.runtime_identity = identity
    contact.main()


if __name__ == '__main__':
    main()
