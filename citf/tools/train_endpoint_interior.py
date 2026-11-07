"""Joint online learning with a small endpoint saturation penalty.

The physical mapping remains the original tanh in the executable thrust box.
An explicit penalty on its log-Jacobian stops the actor objective from driving
the endpoint's logits arbitrarily far into saturation. All Q targets remain
ordinary actual task returns; the only modified loss is endpoint learning.
"""
import math
from torch import nn
from torch.nn import functional as F

from train_shared_team_interaction import main
from train_endpoint_learning import TrainableEndpoint
from unified.learner import Learner
from unified.networks import CITF, JointGenerator, ModelConfig, TeamCritic


class InteriorEndpoint(TrainableEndpoint):
    def map_logits(self, logits):
        self.last_logits = logits
        return logits.tanh()


class InteriorCITF(CITF):
    def __init__(self, cfg=None):
        nn.Module.__init__(self)
        self.cfg = cfg or ModelConfig()
        self.endpoint = InteriorEndpoint(self.cfg)
        self.generator = JointGenerator(self.cfg)
        self.q1, self.q2 = TeamCritic(self.cfg, 1), TeamCritic(self.cfg, 2)


class InteriorLearner(Learner):
    interior_weight = .001

    def endpoint_objective(self, x, a, weights, value):
        base = super().endpoint_objective(x, a, weights, value)
        z = self.model.endpoint.last_logits
        # -log(1-tanh(z)^2), evaluated without subtracting nearly equal
        # floating-point values. Its derivative is the actual 2*tanh(z).
        penalty = self.interior_weight*(2*(F.softplus(2*z)-z-math.log(2))).mean()
        self.last_interior_penalty = penalty.detach()
        return base+penalty

    def update(self, *args, **kwargs):
        result = super().update(*args, **kwargs)
        if 'endpoint_loss' in result:
            result['endpoint_interior_penalty'] = self.last_interior_penalty
        return result

    def payload(self):
        result = super().payload()
        result['endpoint_interior_weight'] = self.interior_weight
        return result

    def restore(self, payload):
        if ('endpoint_interior_weight' in payload and
                payload['endpoint_interior_weight'] != self.interior_weight):
            raise ValueError('Continuation retains its registered endpoint penalty.')
        super().restore(payload)


if __name__ == '__main__':
    main(candidate_type=InteriorCITF, candidate_name='endpoint_interior', learner_type=InteriorLearner,
         candidate_definition='original tanh and r25 critic retained; endpoint objective adds0.001 times mean negative log(1-tanh(logit)^2), evaluated stably; no additional reward or value target')
