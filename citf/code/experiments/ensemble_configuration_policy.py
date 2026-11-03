"""Combine every declared current-action policy on one configuration bank.

All members receive the same available configurations, so their relative
features refer to the same set. Logits have fixed equal weights. If a declared
member learned neighborhood expansion, its frozen proposal augments the
complete physical bank under the same unique-configuration cap. Otherwise
the original physical search bank is used. No evaluation labels select a
member or gate its current actions.
"""
import argparse
from dataclasses import asdict
import sys

import torch
from torch import nn

from citf.contracts import canonical_hash, file_hash
from experiments import joint_configuration_policy as joint
from experiments.proposal_bank_configuration_policy import ProposalBankController
from experiments.search_configuration_policy import SearchPolicyConfig


class EnsembleConfigurationPolicy(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.members = nn.ModuleList(members)
        reference = members[0].cfg
        shared = ('objective_risk_weight', 'guard_weight', 'guard_margin',
                  'clearance', 'policy_prior_scale', 'policy_logit_scale')
        if any(any(getattr(member.cfg, key) != getattr(reference, key) for key in shared)
               for member in members[1:]):
            raise ValueError('Ensemble members need the same physical reference and action scale')
        proposals = [member for member in members if hasattr(member.cfg, 'search_condition')]
        if len(proposals) > 1:
            raise ValueError('This proposal union declares exactly one frozen expansion policy')
        self.proposal = proposals[0] if proposals else None
        values = asdict(reference)
        values['search_condition'] = ('complete physical bank plus frozen task-policy neighborhood; '
            'equal-weight current-action logits on the entire common bank' if proposals else
            'complete physical search bank; equal-weight current-action logits')
        self.cfg = SearchPolicyConfig(**values)
        self.capture_inputs = False

    def forward(self, batch):
        outputs = [member(batch) for member in self.members]
        return dict(features=outputs[0]['features'], baseline=outputs[0]['baseline'],
                    logits=torch.stack([output['logits'] for output in outputs]).mean(0))


class EnsembleConfigurationController(joint.policy.PolicyController):
    @torch.no_grad()
    def act(self, world):
        if self.model.proposal is None:
            force, trace = super().act(world)
        else:
            force, trace = ProposalBankController.act(self, world)
        trace.update(ensemble_members=len(self.model.members),
            ensemble_rule='fixed equal-weight mean current-action logits on one common bank',
            selector_predictor_configurations=len(self.model.members) * trace['evaluated_unique'],
            selection=self.model.cfg.search_condition)
        return force, trace


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ensemble-checkpoints', nargs='+', required=True)
    args, remaining = parser.parse_known_args()
    if len(args.ensemble_checkpoints) < 2 or len(set(args.ensemble_checkpoints)) != len(args.ensemble_checkpoints):
        raise ValueError('Declare at least two distinct complete current-action policies')
    sys.argv = [sys.argv[0], *remaining]
    primary = argparse.ArgumentParser(add_help=False)
    primary.add_argument('--checkpoint', required=True)
    checkpoint, _ = primary.parse_known_args()
    if checkpoint.checkpoint != args.ensemble_checkpoints[0]:
        raise ValueError('The primary checkpoint must be the first declared member')

    import citf.collect as collection
    import citf.evaluate as evaluation
    original_load = evaluation.load_behavior

    def load(path, source, world, device):
        members = []
        for filename in args.ensemble_checkpoints:
            revision = torch.load(filename, map_location='cpu', weights_only=False)['config']['revision']
            if revision == joint.JointPolicyConfig().revision:
                collection.Config = joint.JointPolicyConfig
            elif revision == SearchPolicyConfig().revision:
                collection.Config = SearchPolicyConfig
            else:
                raise ValueError('Unregistered current-action ensemble architecture: ' + revision)
            collection.Predictor = joint.JointConfigurationPolicy
            members.append(original_load(filename, source, world, device))
        return EnsembleConfigurationPolicy(members).eval()

    evaluation.load_behavior = load
    evaluation.Controller = EnsembleConfigurationController
    original_identity = evaluation.runtime_identity

    def identity(*positional, **kwargs):
        result = original_identity(*positional, **kwargs)
        result['method'] = 'CITF-CurrentConfigurationEnsemble-development'
        result['controller'].update(ensemble_checkpoints=[
            dict(path=path, sha256=file_hash(path)) for path in args.ensemble_checkpoints],
            ensemble_rule='fixed equal-weight mean current-action logits; every declared member',
            common_bank='physical bank plus frozen learned expansion when a search-policy member is declared',
            proposal_count='at most one declared frozen search-policy member',
            unique_configuration_budget_unchanged=True,
            additional_inference='each member scores every common candidate; proposal inference recorded separately')
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            ensemble_configuration_policy=file_hash(__file__),
            configuration_policy=file_hash(joint.policy.__file__),
            joint_configuration_policy=file_hash(joint.__file__),
            proposal_bank_configuration_policy=file_hash(sys.modules[ProposalBankController.__module__].__file__)))
        return result

    evaluation.runtime_identity = identity
    evaluation.main()


if __name__ == '__main__':
    main()
