"""Learn whether to retain the teacher or use its matched physical proposal.

This controlled action-space experiment uses ordinary complete-task policy
gradients and the existing short-motion replay, with no outcome predictor.
The teacher and physical proposals retain their full joint configurations.
"""
from dataclasses import dataclass

import numpy as np
import torch

from experiments import contact_event_policy as contact


@dataclass(frozen=True)
class ContactPairConfig(contact.ContactEventConfig):
    revision: str = 'CITF-contact-teacher-physical-pair-policy-v1'
    decision_domain: str = 'retained-teacher-and-unchanged-physical-joint-proposals'
    retained_prior_logodds: float = 1.5


class ContactPairController(contact.ContactEventController):
    @torch.no_grad()
    def act(self, world):
        # Reuse the observed event state and exact teacher candidate set. The
        # provisional choice is never executed or used as a training action.
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
        teacher_plan = trace['retained_teacher_indices']
        teacher = next(i for i, p in enumerate(plans) if p.tolist() == teacher_plan)
        selected = [0] if teacher == 0 else [0, teacher]
        candidate_fields = {'nominal', 'candidate', 'unary', 'candidate_valid', 'teacher_logits'}
        values = {k: (v[selected] if k in candidate_fields else v)
                  for k, v in values.items()}
        source_logits = values['teacher_logits'].copy()
        values['teacher_logits'] = np.full(len(selected),
            -self.model.cfg.retained_prior_logodds, dtype=np.float32)
        retained = 0 if teacher == 0 else 1
        values['teacher_logits'][retained] = 0.
        batch = {k: torch.from_numpy(v[None]).to(self.device) for k, v in values.items()}
        logits = self.model(batch)['logits'][0]
        distribution = torch.distributions.Categorical(logits=logits)
        choice = int(distribution.sample()) if stochastic else min(
            range(len(selected)), key=lambda i: (-float(logits[i]), i != retained))
        phases = self.transition['phases']
        self.transition = dict(inputs={k: v.copy() for k, v in values.items()},
            action=choice,
            old_logp=float(distribution.log_prob(torch.tensor(choice, device=logits.device))),
            changed=int(choice != retained), entropy=float(distribution.entropy()), phases=phases)
        plans = plans[selected]
        from citf.search import candidate_grid
        learned, rules, prior = self.source.commands(world.env)
        grid, _ = candidate_grid(prior)
        gate = grid[torch.arange(len(plans[choice])), torch.as_tensor(plans[choice])]
        force = (750 * (gate * learned + (1 - gate) * rules) + 250).numpy()
        trace.update(configurations=plans.tolist(), scores=(-logits).cpu().tolist(),
            selected_indices=plans[choice].tolist(), gates=gate.tolist(), physical_thrust=force.tolist(),
            teacher_logits=values['teacher_logits'].tolist(), source_teacher_logits=source_logits.tolist(),
            policy_candidate_count=len(selected), retained_proposal_index=retained,
            selection='ordinary policy choice between retained and unchanged physical joint proposals')
        return force, trace


_initialize_contact_worker = contact.initialize


def initialize_pair_worker(project, source, config):
    contact.ContactEventConfig = ContactPairConfig
    contact.ContactEventController = ContactPairController
    _initialize_contact_worker(project, source, config)


def main():
    contact.ContactEventConfig = ContactPairConfig
    contact.ContactEventController = ContactPairController
    contact.initialize = initialize_pair_worker
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        value['controller']['policy_proposals'] = 'teacher and unchanged physical'
        value['runtime_code'] = canonical_hash(dict(
            core=value['runtime_code'], contact_pair=file_hash(__file__)))
        return value

    evaluation.runtime_identity = identity
    contact.main()


if __name__ == '__main__':
    main()
