"""Closed-loop development diagnostic for the feedback short-risk objective.

This expensive reference uses observed initial velocities and zero future
current, with the registered opponent model. Its computation time is measured;
it is not CITF-Phys or a claim of deployable real-time performance.
"""
import time

import numpy as np
import torch

from citf.bridge import Controller, factorize
from citf.contracts import canonical_hash, file_hash
from citf.opponents import APFOpponent
from citf.search import nominal_beam, initialize_bank, whole_configuration_search
from experiments.branch_diagnosis import feedback_distances


class FeedbackController(Controller):
    @torch.no_grad()
    def act(self, world):
        if self.model is not None:
            raise ValueError('Feedback diagnostic does not load a predictor')
        start = time.perf_counter()
        f = factorize(world, self.source)
        p = world.protocol
        ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), p.risk_weight, 128)
        initial, centre = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(),
                                          f['prior'].numpy(), self.bank, self.budget)
        branch_steps = 0
        def score(plans):
            nonlocal branch_steps
            distances, steps = feedback_distances(world, self.source, f, plans, APFOpponent())
            branch_steps += steps
            unary = f['unary'].numpy()[np.arange(len(centre))[None], plans].sum(-1)
            return unary + p.risk_weight * (np.maximum(p.risk_clearance - distances, 0) ** 2).sum(-1)
        indices, trace = whole_configuration_search(initial, score,
            [row.nonzero()[:, 0].numpy() for row in f['valid']], self.budget, self.rounds)
        gates = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * f['learned'] + (1 - gates) * f['rules']) + 250).numpy()
        trace.update(gates=gates.tolist(), selected_indices=indices.tolist(),
                     physical_thrust=force.tolist(), initial_configurations=initial.tolist(),
                     source_center_indices=centre.tolist(),
                     source_center_scored=centre.tolist() in trace['configurations'],
                     nominal_branch_transitions=branch_steps,
                     compute_seconds=time.perf_counter() - start,
                     reference='observed initial velocity; zero future current; APF opponent model')
        return force, trace


def main():
    from pathlib import Path
    import citf.evaluate as evaluation
    original_identity = evaluation.runtime_identity
    def identity(*args, **kwargs):
        result = original_identity(*args, **kwargs)
        if result['opponent'].get('kind') != 'apf':
            raise ValueError('This diagnostic supports the registered APF opponent only')
        result['method'] = 'Feedback-short-risk-development-diagnostic'
        result['controller']['reference'] = 'observed initial velocity; zero future current; registered APF and agility'
        result['controller']['deployment_claim'] = False
        result['runtime_code'] = canonical_hash({'core': result['runtime_code'],
            'feedback_control': file_hash(__file__),
            'branch_diagnosis': file_hash(Path(__file__).with_name('branch_diagnosis.py'))})
        return result
    evaluation.Controller = FeedbackController
    evaluation.runtime_identity = identity
    evaluation.main()


if __name__ == '__main__':
    main()
