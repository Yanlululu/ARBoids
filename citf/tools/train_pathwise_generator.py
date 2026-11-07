"""Hybrid gradients for the existing autoregressive fusion distribution.

Continuous Beta samples and their physical prefixes carry path derivatives
of the actual-action team Q. Discrete endpoint/interior choices retain their
categorical score gradient. Both estimate the existing single-configuration
expected-value objective; optimized-configuration feedback remains intact.
"""
import torch
from torch import nn
from torch.distributions import Beta, Categorical

from train_shared_team_interaction import main
from unified.environment import fusion
from unified.learner import Learner
from unified.networks import (CITF, ModelConfig, Endpoint, JointGenerator,
                              TeamCritic, relative_geometry, frozen)


class PathwiseJointGenerator(JointGenerator):
    def sample_pathwise(self, x, a, b, count):
        a, b = a.detach(), b.detach()
        batch, n = a.shape[:2]
        pair = relative_geometry(x)
        z = self.encode(x, a, b, pair)[:, None].expand(-1, count, -1, -1).reshape(batch*count, n, -1)
        pair = pair[:, None, :n, :n].expand(-1, count, -1, -1, -1).reshape(batch*count, n, n, 8)
        xx = x[:, None].expand(-1, count, -1, -1).reshape(batch*count, n+1, -1)
        aa = a[:, None].expand(-1, count, -1, -1).reshape(batch*count, n, 2)
        bb = b[:, None].expand(-1, count, -1, -1).reshape(batch*count, n, 2)
        prefix = z.new_zeros(batch*count, n, 4)
        coefficients, category_logs = [], []
        for j in range(n):
            logits, shape = self.parameters_at(z, prefix, pair)
            category = Categorical(logits=logits[:, j])
            kind = category.sample()
            limits = torch.finfo(shape.dtype)
            interior = Beta(shape[:, j, 0], shape[:, j, 1]).rsample().clamp(limits.tiny, 1-limits.eps/2)
            coefficient = torch.where(kind == 0, 0., torch.where(kind == 1, 1., interior))
            coefficients.append(coefficient)
            category_logs.append(category.log_prob(kind))
            partial = torch.stack(coefficients, -1)
            g = torch.cat((partial, partial.new_zeros(batch*count, n-j-1)), -1)
            u = fusion(xx, aa, bb, g, self.cfg.thrust_delta)[:, j]
            token = torch.cat((coefficient[:, None], u/1000, torch.ones_like(coefficient[:, None])), -1)
            # Preserve the previous computation graph; later categorical
            # probabilities and continuous shapes depend on this actual prefix.
            prefix = torch.cat((prefix[:, :j], token[:, None], prefix[:, j+1:]), 1)
        return (torch.stack(coefficients, -1).reshape(batch, count, n),
                torch.stack(category_logs, -1).sum(-1).reshape(batch, count))


class PathwiseGeneratorCITF(CITF):
    def __init__(self, cfg=None):
        nn.Module.__init__(self)
        self.cfg = cfg or ModelConfig()
        self.endpoint = Endpoint(self.cfg)
        self.generator = PathwiseJointGenerator(self.cfg)
        self.q1, self.q2 = TeamCritic(self.cfg, 1), TeamCritic(self.cfg, 2)


class PathwiseGeneratorLearner(Learner):
    generator_gradient_kind = 'categorical-score-and-autoregressive-beta-path-v1'

    def generator_base_objective(self, x, result):
        a, b = result['a'].detach(), result['b'].detach()
        g, category_logp = self.model.generator.sample_pathwise(x, a, b, self.model.cfg.samples)
        with frozen(self.model.q1), frozen(self.model.q2):
            with torch.no_grad():
                encoded = self.model.encode_value(x)
            actions = fusion(x, a, b, g, self.model.cfg.thrust_delta)
            value = self.model.value_from_encoded(encoded, actions)/100.
        baseline = (value.sum(-1, keepdim=True)-value)/(self.model.cfg.samples-1)
        path_loss = -value.mean()
        category_loss = -((value-baseline).detach()*category_logp).mean()
        self.last_path_loss, self.last_category_loss = path_loss.detach(), category_loss.detach()
        return path_loss+category_loss

    def update(self, *args, **kwargs):
        report = super().update(*args, **kwargs)
        if 'base_loss' in report:
            report.update(generator_path_loss=self.last_path_loss,
                          generator_category_loss=self.last_category_loss)
        return report

    def payload(self):
        result = super().payload()
        result['generator_gradient_kind'] = self.generator_gradient_kind
        return result

    def restore(self, payload):
        kind = payload.get('generator_gradient_kind')
        if kind is not None and kind != self.generator_gradient_kind:
            raise ValueError('Resume retains the registered generator gradient estimator.')
        if (payload.get('continuation', {}).get('variant') == 'pathwise_generator'
                and kind != self.generator_gradient_kind):
            raise ValueError('The saved pathwise-generator candidate requires its learning identity.')
        super().restore(payload)


if __name__ == '__main__':
    main(candidate_type=PathwiseGeneratorCITF, candidate_name='pathwise_generator',
         learner_type=PathwiseGeneratorLearner,
         candidate_definition='r25 value targets and aggregation retained; the original expected-Q generator objective uses Beta path derivatives through actual thrust and all generated prefixes plus categorical score gradients with an independent-sample leave-one-out baseline; four fresh independent base-learning configurations per policy update; original optimized-configuration feedback and deployment pool retained')
