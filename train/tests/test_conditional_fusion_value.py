import unittest

import torch

from conditional_fusion_value import (
    component_relation_features, component_response_features, conditional_auxiliary, conditional_edges, paired_targets, paired_covariance_targets,
)


class ConditionalFusionValueTests(unittest.TestCase):
    def setUp(self):
        self.parts = torch.tensor([[0, 0], [1, 0], [2, 0], [1, 2]])
        self.options = torch.tensor([[[[0., 0.], [0., 0.]],
            [[.5, .5], [0., 0.]], [[0., 0.], [.5, -.5]], [[.5, .5], [.5, -.5]]]])

    def test_every_credit_edge_changes_at_most_one_vessel(self):
        edges = conditional_edges(self.parts)
        changes = (self.options[:, edges[:, 0]] - self.options[:, edges[:, 1]]).abs().amax(-1) > 0
        self.assertTrue((changes.sum(-1) == 1).all())
        self.assertEqual(edges.tolist(), [[1, 0], [2, 0], [3, 1], [3, 2]])

    def test_common_future_return_shift_cancels(self):
        generator = torch.Generator().manual_seed(33)
        costs = torch.randn(3, 8, 4, generator=generator, dtype=torch.double)
        shifts = 20. * torch.randn(3, 8, 1, generator=generator, dtype=torch.double)
        edges = conditional_edges(self.parts)
        for first, second in zip(paired_targets(costs, edges), paired_targets(costs+shifts, edges)):
            for a, b in zip(first, second):
                torch.testing.assert_close(a, b)

    def test_credit_gradient_prefers_lower_real_cost(self):
        edges = torch.tensor([[1, 0]])
        costs = torch.tensor([[[5., 2.]]]*1).repeat(1, 8, 1).double()
        energy = torch.zeros(1, 2, dtype=torch.double, requires_grad=True)
        loss = conditional_auxiliary(energy, edges, paired_targets(costs, edges), torch.ones(1, 1)).sum()
        loss.backward()
        self.assertGreater(float(energy.grad[0, 1]), 0.)
        self.assertLess(float(energy.grad[0, 0]), 0.)
        self.assertAlmostEqual(float(energy.grad.sum()), 0.)

    def test_mean_credit_preserves_asymmetric_return_magnitude(self):
        # Opposite folds have differences -2 and +3: the candidate costs
        # +0.5 in expectation, even though a saturated robust loss cancels.
        edges = torch.tensor([[1, 0]])
        folds = [(torch.tensor([[-2.]]), torch.ones(1, 1)),
            (torch.tensor([[3.]]), torch.ones(1, 1))]
        energy = torch.zeros(1, 2, requires_grad=True)
        conditional_auxiliary(energy, edges, folds, torch.ones(1, 1), quadratic=True).sum().backward()
        self.assertLess(float(energy.grad[0, 1]), 0.)
        self.assertAlmostEqual(float(energy.grad.sum()), 0.)

    def test_relations_depend_on_current_peer_control_and_vanish_for_singletons(self):
        obs = torch.zeros(1, 2, 14)
        obs[..., 2] = torch.tensor([15., 18.])
        obs[..., 12:14] = -.2
        packet = dict(obs=obs, motion=torch.tensor([[[0., 0., 0., 1., 0., 0.],
            [8., 2., .3, 1., .5, 0.]]]))
        base = torch.tensor([[[.8, .4, .5], [.5, .8, .5]]])
        original = component_relation_features(packet, base, self.options, self.parts)
        changed = base.clone(); changed[:, 1, :2] = -.6
        other = component_relation_features(packet, changed, self.options, self.parts)
        self.assertTrue(torch.isfinite(original).all())
        self.assertEqual(float(original[:, :3].abs().max()), 0.)
        self.assertGreater(float((original[:, 3]-other[:, 3]).abs().max()), 1e-4)

    def test_unary_value_does_not_read_joint_candidate_features(self):
        from identify_fusion_mechanisms import reactive_monotone_estimates
        packet = dict(obs=torch.zeros(2, 4, 20), time_remaining=torch.ones(2))
        parts = torch.stack((torch.arange(17), torch.zeros(17, dtype=torch.long)), -1)
        prior = torch.ones(2, 4, 17)
        weights = torch.zeros(3, 10, 16)
        value = reactive_monotone_estimates(weights, prior, parts, packet, False)
        other = reactive_monotone_estimates(weights, prior, parts,
            dict(packet, candidate_interactions=torch.full((2, 113, 12), float('nan'))), False)
        torch.testing.assert_close(value, other)
        self.assertTrue(torch.isfinite(value).all())

    def test_response_interaction_cancels_additive_motion(self):
        packet = dict(obs=torch.zeros(1, 2, 14), motion=torch.tensor([[[0.,0.,0.,1.,0.,0.], [8.,2.,.3,1.,.5,0.]]]))
        packet['obs'][..., 2] = torch.tensor([15., 18.])
        base = torch.tensor([[[.8,.4,.5], [.5,.8,.5]]])
        response = torch.randn(1,4,2,2,2)
        response[:,3] = response[:,1]+response[:,2]-response[:,0]
        encoded = component_response_features(packet,base,self.options,self.parts,response)
        self.assertLess(float(encoded[...,12:].abs().max()),1e-6)
        response[:,3,0,:,0] += 2.
        encoded = component_response_features(packet,base,self.options,self.parts,response)
        self.assertGreater(float(encoded[:,3,12:].abs().max()),.1)
        self.assertEqual(float(encoded[:,:3,12:].abs().max()),0.)

    def test_paired_noise_precision_is_positive_and_common_noise_cancels(self):
        costs = torch.randn(5,12,4,dtype=torch.double)
        edges = conditional_edges(self.parts)
        first = paired_covariance_targets(costs,edges)
        shifted = paired_covariance_targets(costs+torch.randn(5,12,1,dtype=torch.double)*50.,edges)
        for (target,precision),(other,shifted_precision) in zip(first,shifted):
            torch.testing.assert_close(target,other)
            torch.testing.assert_close(precision,shifted_precision)
            self.assertGreater(float(torch.linalg.eigvalsh(precision).min()),0.)
        energy = torch.zeros(5,4,dtype=torch.double,requires_grad=True)
        conditional_auxiliary(energy,edges,first,torch.ones(5,len(edges)),covariance=True).sum().backward()
        self.assertTrue(torch.isfinite(energy.grad).all())


if __name__ == '__main__':
    unittest.main()
