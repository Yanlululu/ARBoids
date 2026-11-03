"""Average every declared policy's action distribution on one common bank.

The original ensemble averages logits, equivalent to a geometric probability
pool. This variant uses an equal arithmetic mixture of normalized categorical
probabilities. Every independent seed remains present with the same weight;
no outcome labels or per-task expert selection enter the controller.
"""
from dataclasses import replace
import math

import torch

from citf.contracts import canonical_hash, file_hash
from experiments import ensemble_configuration_policy as ensemble


RULE = 'fixed equal-weight arithmetic mean categorical probabilities on one common bank'


class ProbabilityEnsemble(ensemble.EnsembleConfigurationPolicy):
    def __init__(self, members):
        super().__init__(members)
        self.cfg = replace(self.cfg, search_condition=RULE)

    def forward(self, batch):
        outputs = [member(batch) for member in self.members]
        log_probabilities = torch.stack([
            torch.log_softmax(output['logits'], dim=-1) for output in outputs])
        logits = torch.logsumexp(log_probabilities, dim=0) - math.log(len(outputs))
        return dict(features=outputs[0]['features'], baseline=outputs[0]['baseline'],
                    logits=logits.masked_fill(~batch['candidate_valid'], -torch.inf))


class ProbabilityController(ensemble.EnsembleConfigurationController):
    @torch.no_grad()
    def act(self, world):
        force, trace = super().act(world)
        trace.update(ensemble_rule=RULE)
        return force, trace


def main():
    ensemble.EnsembleConfigurationPolicy = ProbabilityEnsemble
    ensemble.EnsembleConfigurationController = ProbabilityController
    import citf.evaluate as evaluation
    original_main = evaluation.main

    def evaluate():
        # Install this identity after the base ensemble has registered its
        # loading contract, so the saved runtime names the actual pooling rule.
        previous = evaluation.runtime_identity

        def identity(*args, **kwargs):
            result = previous(*args, **kwargs)
            result['method'] = 'CITF-CurrentConfigurationProbabilityMixture-development'
            result['controller']['ensemble_rule'] = RULE
            result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
                probability_ensemble_configuration_policy=file_hash(__file__)))
            return result

        evaluation.runtime_identity = identity
        original_main()

    evaluation.main = evaluate
    ensemble.main()


if __name__ == '__main__':
    main()
