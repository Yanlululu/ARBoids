"""Let the learned current-action distribution guide candidate expansion.

The original bank and one-coordinate neighborhood budget are unchanged.
All accumulated configurations are rescored together after an expansion,
because within-bank relative features must use a common reference set.
Zero action residuals recover the same monotone physical search decisions.
"""
from dataclasses import dataclass
import sys
import time

import numpy as np
import torch

from citf.bridge import factorize, inputs
from citf.contracts import canonical_hash, file_hash
from citf.search import nominal_beam, initialize_bank
from experiments import joint_configuration_policy as joint


@dataclass(frozen=True)
class SearchPolicyConfig(joint.JointPolicyConfig):
    revision: str = 'CITF-learned-current-configuration-search-v1'
    search_condition: str = 'greedy current action logits guide neighborhood; rescore full accumulated bank'


class SearchPolicyController(joint.JointPolicyController):
    @torch.no_grad()
    def act(self, world):
        started = time.perf_counter()
        self.model.capture_inputs = self.stochastic
        f, cfg = factorize(world, self.source), self.model.cfg
        ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), cfg.objective_risk_weight, 128)
        initial, center = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(),
                                         f['prior'].numpy(), self.bank, self.budget)
        choices = [row.nonzero()[:, 0].numpy() for row in f['valid']]
        plans = [tuple(map(int, row)) for row in initial]
        known = set(plans)
        history_cache = None
        forwards = 0
        expansion_centers = []
        for iteration in range(self.rounds + 1):
            batch, _, _ = inputs(world, f, np.asarray(plans, dtype=np.int64))
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if history_cache is None:
                history_cache = self.model.encode_history(batch)
            output = self.model(batch, history_cache=history_cache)
            forwards += 1
            logits = output['logits'][0]
            best = min(range(len(plans)), key=lambda k: (-float(logits[k]), plans[k]))
            if iteration == self.rounds or len(plans) == self.budget:
                break
            base = plans[best]
            expansion_centers.append(list(base))
            previous = len(plans)
            for i, values in enumerate(choices):
                for value in values:
                    proposal = list(base)
                    proposal[i] = int(value)
                    key = tuple(proposal)
                    if key not in known and len(plans) < self.budget:
                        plans.append(key)
                        known.add(key)
            if len(plans) == previous:
                break
        distribution = torch.distributions.Categorical(logits=logits)
        choice = int(distribution.sample()) if self.stochastic else best
        indices = np.asarray(plans[choice], dtype=np.int64)
        gates = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * f['learned'] + (1 - gates) * f['rules']) + 250).numpy()
        physical = joint.policy.physical_components(batch, cfg)[0][0]
        physical_choice = min(range(len(plans)), key=lambda k: (float(physical[k]), plans[k]))
        self.transition = dict(features=output['features'][0].cpu().half(),
            baseline=output['baseline'][0].cpu(), action=choice,
            log_probability=float(distribution.log_prob(torch.as_tensor(choice, device=logits.device))),
            changed=int(choice != physical_choice), entropy=float(distribution.entropy()))
        if self.stochastic:
            self.transition['inputs'] = self.model.rollout_inputs
            self.model.rollout_inputs = None
        trace = dict(cost=float(-logits[choice]), evaluated_unique=len(plans), forward_batches=forwards,
            configurations=[list(row) for row in plans], scores=(-logits).cpu().tolist(),
            gates=gates.tolist(), selected_indices=indices.tolist(), physical_thrust=force.tolist(),
            physical_selected_indices=list(plans[physical_choice]),
            source_center_indices=center.tolist(), source_center_scored=tuple(center) in known,
            initial_configurations=initial.tolist(), expansion_centers=expansion_centers,
            selection=cfg.search_condition, compute_seconds=time.perf_counter() - started)
        return force, trace


_original_worker = joint.initialize_worker


def initialize_worker(*args):
    joint.JointPolicyConfig = SearchPolicyConfig
    joint.JointPolicyController = SearchPolicyController
    _original_worker(*args)


def main():
    joint.JointPolicyConfig = SearchPolicyConfig
    joint.JointPolicyController = SearchPolicyController
    joint.initialize_worker = initialize_worker
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        joint.train_main()
        return
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        result['controller']['search_condition'] = SearchPolicyConfig().search_condition
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            search_configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    joint.main()


if __name__ == '__main__':
    main()
