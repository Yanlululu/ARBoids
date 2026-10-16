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


class JointExecutionRegressionTests(unittest.TestCase):
    def test_learning_data_identity_distinguishes_active_predictors(self):
        from develop_source_fusion import joint_interaction_identity
        base={'levels':3,'risk_prior':32.,'coupled_shortlist':16,'risk_feedback':True}
        changed=dict(base,prediction_discount=1.)
        self.assertNotEqual(joint_interaction_identity(base),joint_interaction_identity(changed))
        base['mechanism_ablation']=(True,True,True)
        changed['mechanism_ablation']=(True,False,True)
        self.assertEqual(joint_interaction_identity(base),joint_interaction_identity(changed))

    def test_continuation_identity_tracks_value_changes_not_fit_labels(self):
        import copy
        from develop_source_fusion import joint_continuation_identity
        model={'actor':{'weight':torch.ones(2)},'joint_intervention_value':
            {'values':torch.tensor([[.2,.7]]),'learning_objective':'conditional'}}
        reference=joint_continuation_identity(model)
        relabeled=copy.deepcopy(model)
        relabeled['joint_intervention_fit']={'conditional':False,'roots':500}
        relabeled['joint_intervention_value']['learning_objective']='absolute'
        self.assertEqual(reference,joint_continuation_identity(relabeled))
        relabeled['joint_intervention_value']['values'][0,1]=.3
        self.assertNotEqual(reference,joint_continuation_identity(relabeled))

    def test_value_fitting_reenables_learning_but_preserves_interaction_removal(self):
        from types import SimpleNamespace
        from develop_source_fusion import require_clean_mechanism_start
        args=SimpleNamespace(joint_no_cf=False,without_peer=False,joint_no_interaction=False)
        model={'mechanism_ablation':(True,True,True)}
        require_clean_mechanism_start(model,args)
        self.assertEqual(model['mechanism_ablation'],(True,False,True))
        model={'mechanism_ablation':(False,True,False)}
        require_clean_mechanism_start(model,args)
        self.assertEqual(model['mechanism_ablation'],(False,False,False))

    def test_absolute_appended_head_keeps_conditional_ancestry_visible(self):
        from develop_source_fusion import joint_value_objectives
        conditional={'learning_objective':'conditional'}
        absolute={'learning_objective':'absolute','prior':conditional}
        model={'joint_intervention_value':absolute,'joint_intervention_fit':{'conditional':False}}
        self.assertEqual(joint_value_objectives(model),{'absolute','conditional'})
        absolute.pop('prior')
        self.assertEqual(joint_value_objectives(model),{'absolute'})
        model['program_proposer']={'values':[conditional]}
        self.assertEqual(joint_value_objectives(model),{'absolute','conditional'})

    def test_empty_supported_layer_preserves_predecessor_without_context_query(self):
        from unittest.mock import patch
        from conditional_fusion_value import JointInterventionValue
        value=JointInterventionValue(contexts=[[0.]*8],values=[[0.]*4],observed=[[False]*4],
            span=10,anchor={'contexts':[[0.]*8],'trajectories':[0],'returns':[0.]})
        action=torch.zeros(2,1,6);action[...,3]=.5;action[...,4]=20;action[...,5]=1.
        with patch('conditional_fusion_value.public_return_context',side_effect=AssertionError('inactive query')):
            result,gain=value({},action,return_gain=True)
        torch.testing.assert_close(result,action)
        self.assertTrue((gain==-1.).all())

    def test_competing_windows_share_one_frozen_predecessor(self):
        from conditional_fusion_value import JointInterventionOptions
        record=dict(contexts=[[0.]*8],values=[[0.]*4],observed=[[False]*4],span=10,
            anchor={'contexts':[[0.]*8],'trajectories':[0],'returns':[0.]})
        options=JointInterventionOptions([dict(record,prior=record),dict(record,span=20,prior=record)])
        self.assertTrue(options.shared_prior)
        self.assertIs(options.values[0].prior,options.values[1].prior)
        action=torch.zeros(1,1,6)
        torch.testing.assert_close(options({},action),action)
        torch.testing.assert_close(JointInterventionOptions([])({},action),action)

    def test_rejected_high_gain_window_cannot_hide_an_accepted_window(self):
        from conditional_fusion_value import JointInterventionOptions
        from develop_source_fusion import JointLearningActor
        class Window(torch.nn.Module):
            def __init__(self,gain,margin):
                super().__init__();self.gain=gain;self.margin=margin
            def forward(self,packet,action,**kwargs):
                candidate=action.clone()
                if self.gain>self.margin:candidate[...,2]=1.
                return candidate,action.new_full((len(action),),self.gain)
        values=torch.nn.ModuleList([Window(.2,.25),Window(.15,.05)])
        options=JointInterventionOptions.__new__(JointInterventionOptions)
        torch.nn.Module.__init__(options);options.values=values;options.shared_prior=False
        action=torch.zeros(2,2,6)
        self.assertTrue((options({},action)[...,2]==1.).all())
        from types import SimpleNamespace
        actor=SimpleNamespace(shared_intervention_prior=False,joint_intervention_values=values)
        self.assertTrue((JointLearningActor.choose_joint_value_option(actor,{},action)[...,2]==1.).all())

    def test_complete_joint_prediction_obeys_feedback_and_discount(self):
        from unittest.mock import patch
        from develop_source_fusion import JointLearningActor
        torch.manual_seed(120)
        actor=JointLearningActor(peer=True,graph=True,levels=3,risk_prior=2.,risk_horizon=.6,
            risk_feedback=True,coupled_shortlist=9)
        node=torch.zeros(1,2,16);node[:,:,11:13]=.25;node[:,0,2]=.8;node[:,1,2]=.7
        motion=torch.tensor([[[0.,0.,0.,.5,0.,0.],[4.,0.,.3,-.5,0.,0.]]])
        packet={'node':node,'motion':motion,'obs':torch.zeros(1,2,16)}
        codes=torch.arange(9);plans=torch.stack((codes%3,codes//3),-1).float()/2
        proposals=torch.tensor([[[.6,-.3],[-.3,.6]]]);scores=torch.zeros(1,9)
        with torch.no_grad(),patch.object(actor,'feedback_observation',wraps=actor.feedback_observation) as feedback:
            actor.risk_feedback=False
            frozen=actor.refine_joint_scores(dict(packet),proposals,plans,scores,scores)
            self.assertEqual(feedback.call_count,0)
            actor.risk_feedback=True
            refreshed=actor.refine_joint_scores(dict(packet),proposals,plans,scores,scores)
            self.assertGreater(feedback.call_count,0)
            self.assertGreater(float((frozen-refreshed).abs().max()),1e-5)
            actor.prediction_discount=.2;actor.prediction_discount_start=0.
            discounted=actor.refine_joint_scores(dict(packet),proposals,plans,scores,scores)
            self.assertGreater(float((refreshed-discounted).abs().max()),1e-5)


class SharedTemporalProgramTests(unittest.TestCase):
    def test_temporal_value_loads_with_its_matching_shared_estimator(self):
        from conditional_fusion_value import JointInterventionValue,SharedProgramPotential,temporal_program_options
        options=temporal_program_options().tolist()
        net=SharedProgramPotential(width=8,program_options=options)
        context=torch.zeros(1,51)
        record=dict(width=8,program_options=options,state=net.state_dict())
        value=JointInterventionValue(context,torch.zeros(1,6**6),torch.ones(1,6**6,dtype=torch.bool),5,
            dict(contexts=context,trajectories=[0],returns=[0.]),program_options=options,program_model=record)
        self.assertEqual(value.program_radix,6)
        torch.testing.assert_close(value.program_model(context,torch.tensor([5])),net(context,torch.tensor([5])))

    def test_temporal_programs_decode_the_complete_six_choice_alphabet(self):
        from conditional_fusion_value import SharedProgramPotential,temporal_program_options
        net=SharedProgramPotential(width=8,node_width=8,program_options=temporal_program_options())
        captured=[]
        hook=net.node[0].register_forward_pre_hook(lambda module,args: captured.append(args[0].detach()))
        net(torch.zeros(2,51),torch.tensor([5,6]))
        hook.remove()
        self.assertEqual(captured[0][...,8:].argmax(-1).tolist(),[[5,0,0,0,0,0],[0,1,0,0,0,0]])

    def test_phase_order_and_shared_value_survive_vessel_relabeling(self):
        from conditional_fusion_value import SharedProgramPotential,temporal_program_options
        torch.manual_seed(9)
        net=SharedProgramPotential(width=12,node_width=15,global_width=4,program_options=temporal_program_options())
        rows=torch.randn(2,6,15);global_state=torch.randn(2,4)
        choices=torch.tensor([[4,5,2,3,0,1],[5,4,1,0,3,2]])
        permutation=torch.tensor([3,0,5,1,4,2]);powers=6**torch.arange(6)
        before=net(torch.cat((rows.flatten(1),global_state),-1),(choices*powers).sum(-1))
        after=net(torch.cat((rows[:,permutation].flatten(1),global_state),-1),(choices[:,permutation]*powers).sum(-1))
        torch.testing.assert_close(before,after)
        restored=SharedProgramPotential(width=12,node_width=15,global_width=4,
            program_options=temporal_program_options(),state=net.state_dict())
        torch.testing.assert_close(before,restored(torch.cat((rows.flatten(1),global_state),-1),(choices*powers).sum(-1)))


if __name__ == '__main__':
    unittest.main()
