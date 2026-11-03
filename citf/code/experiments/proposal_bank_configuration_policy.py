"""Learn current choices on a stable physical and learned proposal union.

A frozen task policy expands a learned neighborhood alongside the complete
physical search bank. The union respects the declared unique-configuration
budget. Only the current configuration selector and its interaction encoder
are refined by task policy gradients and stopped-geometry replay. Proposal
weights remain fixed throughout training and are stored in every checkpoint.
"""
from dataclasses import dataclass, replace
from functools import partial
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from citf.bridge import factorize, inputs
from citf.contracts import canonical_hash, file_hash
from citf.search import nominal_beam, initialize_bank, whole_configuration_search
from experiments import joint_configuration_policy as joint
from experiments.search_configuration_policy import SearchPolicyConfig


@dataclass(frozen=True)
class ProposalBankConfig(joint.JointPolicyConfig):
    revision: str = 'CITF-frozen-policy-proposal-union-v1'
    search_condition: str = 'complete physical bank plus frozen task-policy neighborhood; unique union budget'
    proposal_training: str = 'frozen pretrained task policy; selector and geometry encoder train jointly'
    rollout_device: str = 'cuda'

    def __post_init__(self):
        object.__setattr__(self, 'action_bank', 'physical_and_frozen_policy_union')


BaseJointPolicy = joint.JointConfigurationPolicy


class ProposalBankPolicy(BaseJointPolicy):
    def __init__(self, cfg=ProposalBankConfig()):
        # The base class normally interprets proposal_config as a motion model.
        # This registered variant instead stores a complete fixed task policy.
        super().__init__(replace(cfg, proposal_config=None))
        self.cfg = cfg
        if cfg.proposal_config is None:
            raise ValueError('The proposal union requires its fixed task-policy checkpoint')
        self.proposal = BaseJointPolicy(SearchPolicyConfig(**cfg.proposal_config))
        self.proposal.requires_grad_(False)


class ProposalBankController(joint.JointPolicyController):
    @torch.no_grad()
    def act(self, world):
        started = time.perf_counter()
        self.model.capture_inputs = self.stochastic
        f, cfg = factorize(world, self.source), self.model.cfg
        ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), cfg.objective_risk_weight, 128)
        initial, center = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(),
                                         f['prior'].numpy(), self.bank, self.budget)
        choices = [row.nonzero()[:, 0].numpy() for row in f['valid']]

        def physical_score(plans):
            batch, _, _ = inputs(world, f, plans)
            return joint.policy.physical_components(batch, cfg)[0][0].cpu().numpy()

        physical_choice, physical_trace = whole_configuration_search(
            initial, physical_score, choices, self.budget, self.rounds)
        plans = [tuple(map(int, row)) for row in physical_trace['configurations']]
        known = set(plans)
        proposed = [tuple(map(int, row)) for row in initial]
        proposal_known = set(proposed)
        proposal_cache = None
        expansion_centers = []
        proposal_scored = 0
        # Only the frozen proposal model picks expansion centers. Its bank
        # cannot drift as a side effect of a selector policy-gradient update.
        for iteration in range(self.rounds):
            batch, _, _ = inputs(world, f, np.asarray(proposed, dtype=np.int64))
            batch = {k: v.to(self.device) for k, v in batch.items()}
            if proposal_cache is None:
                proposal_cache = self.model.proposal.encode_history(batch)
            logits = self.model.proposal(batch, history_cache=proposal_cache)['logits'][0]
            proposal_scored += len(proposed)
            best = min(range(len(proposed)), key=lambda k: (-float(logits[k]), proposed[k]))
            base = proposed[best]
            expansion_centers.append(list(base))
            neighborhoods = [sorted(map(int, values), key=lambda value: (abs(value - base[i]), value))
                             for i, values in enumerate(choices)]
            # Interleave vessels so a tight remaining union budget does not
            # consume every slot on the first few vessel coordinates.
            for neighbor_rank in range(max(map(len, neighborhoods))):
                for i, values in enumerate(neighborhoods):
                    if neighbor_rank >= len(values):
                        continue
                    value = values[neighbor_rank]
                    candidate = list(base)
                    candidate[i] = int(value)
                    key = tuple(candidate)
                    if key not in known and len(plans) < self.budget:
                        plans.append(key)
                        known.add(key)
                    if key in known and key not in proposal_known:
                        proposed.append(key)
                        proposal_known.add(key)
            if len(plans) == self.budget:
                break

        batch, _, _ = inputs(world, f, np.asarray(plans, dtype=np.int64))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        output = self.model(batch)
        logits = output['logits'][0]
        distribution = torch.distributions.Categorical(logits=logits)
        choice = int(distribution.sample()) if self.stochastic else min(
            range(len(plans)), key=lambda k: (-float(logits[k]), plans[k]))
        indices = np.asarray(plans[choice], dtype=np.int64)
        gates = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * f['learned'] + (1 - gates) * f['rules']) + 250).numpy()
        self.transition = dict(features=output['features'][0].cpu().half(),
            baseline=output['baseline'][0].cpu(), action=choice,
            log_probability=float(distribution.log_prob(torch.as_tensor(choice, device=logits.device))),
            changed=int(not np.array_equal(indices, physical_choice)), entropy=float(distribution.entropy()))
        if self.stochastic:
            self.transition['inputs'] = self.model.rollout_inputs
            self.model.rollout_inputs = None
        trace = dict(cost=float(-logits[choice]), evaluated_unique=len(plans),
            configurations=[list(row) for row in plans], scores=(-logits).cpu().tolist(),
            gates=gates.tolist(), selected_indices=indices.tolist(), physical_thrust=force.tolist(),
            physical_selected_indices=physical_choice.tolist(),
            physical_scored_unique=physical_trace['evaluated_unique'],
            proposal_predictor_configurations=proposal_scored,
            selector_predictor_configurations=len(plans),
            source_center_indices=center.tolist(), source_center_scored=tuple(center) in known,
            initial_configurations=initial.tolist(), expansion_centers=expansion_centers,
            selection=cfg.search_condition, compute_seconds=time.perf_counter() - started)
        return force, trace


_original_worker = joint.initialize_worker


def initialize_worker(*args):
    joint.JointPolicyConfig = ProposalBankConfig
    joint.JointConfigurationPolicy = ProposalBankPolicy
    joint.JointPolicyController = ProposalBankController
    _original_worker(*args)
    device = joint.policy._worker['model'].cfg.rollout_device
    joint.policy._worker['model'].to(device)
    joint.policy.PolicyController = partial(ProposalBankController, device=device)


def main():
    joint.JointPolicyConfig = ProposalBankConfig
    joint.JointConfigurationPolicy = ProposalBankPolicy
    joint.JointPolicyController = ProposalBankController
    joint.initialize_worker = initialize_worker
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        implementation = file_hash(__file__)
        save = joint.atomic_save
        recorded_protocol = False

        def save_checkpoint(value, path):
            nonlocal recorded_protocol
            protocol = dict(value['protocol'])
            protocol['frozen_proposal_union'] = dict(implementation=implementation,
                proposal_training=ProposalBankConfig().proposal_training,
                unique_configuration_cap=True, extra_proposal_inference_accounted_separately=True)
            protocol['runtime_code'] = canonical_hash(dict(core=protocol['runtime_code'],
                proposal_bank_configuration_policy=implementation))
            if not recorded_protocol:
                (Path(path).parent / 'protocol.json').write_text(json.dumps(protocol, indent=2))
                recorded_protocol = True
            save(dict(value, protocol=protocol), path)

        joint.atomic_save = save_checkpoint
        joint.train_main()
        return
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        result['controller']['search_condition'] = ProposalBankConfig().search_condition
        result['controller']['proposal_training'] = ProposalBankConfig().proposal_training
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            proposal_bank_configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    joint.main()


if __name__ == '__main__':
    main()
