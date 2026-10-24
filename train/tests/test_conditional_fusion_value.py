import unittest

import torch

from conditional_fusion_value import (
    component_relation_features, component_response_features, conditional_auxiliary, conditional_edges, paired_targets, paired_covariance_targets,
)


class ConditionalFusionValueTests(unittest.TestCase):
    def test_relative_intervention_follows_feedback_keeps_peers_and_null_identity(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import conditional_fusion_value as conditional
        source = torch.tensor([[[.8, -.4, .6], [.2, .7, .4], [-.5, .3, .5]]])
        node = torch.zeros(1, 3, 16)
        node[..., 11:13] = torch.tensor([[[-.2, .1], [.6, -.3], [.4, .8]]])
        coefficients = torch.tensor([[[.2, .8], [.4, .6], [.7, .3]]])
        held = torch.full((1, 3, 2), -1.)
        held[0, 1] = torch.tensor([.8, .2])
        packet = dict(node=node, _committed_actuator_preferences=held,
                      _isolate_factual_actuator_start=torch.tensor([True]))
        actor = SimpleNamespace(generic_fusion_adapter=None)
        def factual(*args, return_coefficients=False, **kwargs):
            command = coefficients*source[..., :2]+(1-coefficients)*node[..., 11:13]
            result = torch.cat((command, torch.ones_like(source[..., :1])), -1)
            return (result, coefficients) if return_coefficients else result
        offsets = torch.full_like(held, float('nan'))
        with patch.object(conditional, '_refine_actuator_interactions', side_effect=factual):
            isolated = conditional.refine_actuator_interactions(actor, packet, source, source)
            first, offsets = conditional.refine_feedback_relative_intervention(actor, packet, source, offsets)
            torch.testing.assert_close(first, isolated, rtol=0., atol=0.)
            torch.testing.assert_close(offsets[0, 1], torch.tensor([.4, -.4]))
            self.assertTrue(torch.isnan(offsets[0, [0, 2]]).all())
            coefficients[0, 1] = torch.tensor([.9, .1])
            following, retained = conditional.refine_feedback_relative_intervention(actor, packet, source, offsets)
            torch.testing.assert_close(following[0, 1, :2], torch.tensor([.2, -.3]), rtol=0., atol=1e-7)
            torch.testing.assert_close(following[:, [0, 2]], factual()[:, [0, 2]], rtol=0., atol=0.)
            torch.testing.assert_close(retained, offsets, equal_nan=True, rtol=0., atol=0.)
            held[0, 1] = coefficients[0, 1]
            unchanged, zero = conditional.refine_feedback_relative_intervention(
                actor, packet, source, torch.full_like(held, float('nan')))
            torch.testing.assert_close(unchanged, factual(), rtol=0., atol=0.)
            self.assertEqual(float(zero[0, 1].abs().sum()), 0.)
            held[0, 0] = .5
            with self.assertRaisesRegex(ValueError, 'one vessel'):
                conditional.refine_feedback_relative_intervention(actor, packet, source, zero)

    def test_relative_training_and_merge_preserve_execution_semantics(self):
        import copy
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import (
            fit_joint_potential, merge_paired_data, LocalDifferenceDecision, PlainJointReturnDecision)
        controls = np.tile(np.array([.8, .3, -.4, .6, .1, -.2], np.float32), (2, 1))
        root = dict(context=np.zeros(21, np.float32), effect_controls=controls,
            scene=(19, 2, 2.), prefix_future=None, root_steps=5, futures=[101, 102],
            feedback_relative_hold=True, options=[(0, 0, 0), (1, 0, 5)],
            outcomes={(q, f): dict(success=float(q == f)) for q in range(2) for f in range(2)})
        data = dict(model={}, roots=[root], predecessor_sha256='relative-test', simulation_steps=20,
                    feedback_relative_hold=True, isolate_factual_start=True)
        absolute = dict(data, feedback_relative_hold=False)
        with self.assertRaisesRegex(ValueError, 'different physical experiments'):
            merge_paired_data([data, absolute])
        merged = merge_paired_data([data, copy.deepcopy(data)])
        self.assertTrue(merged['feedback_relative_hold'])
        self.assertEqual(len(merged['roots'][0]['outcomes']), 4)
        for ordinary in (False, True):
            with patch('local_intervention_value.torch.save'):
                model = fit_joint_potential(data, 'unused-relative-test.pth', ordinary=ordinary,
                    ordinary_architecture='interaction', updates=2, ensemble_size=1, width=4, device='cpu')
            recipe = dict(model['successor_task_value']); recipe.pop('kind')
            decision = PlainJointReturnDecision(**recipe) if ordinary else LocalDifferenceDecision(**recipe)
            self.assertTrue(decision.feedback_relative_hold)
            self.assertTrue(model['successor_task_fit']['feedback_relative_hold'])
        invalid = copy.deepcopy(data); invalid['roots'][0]['feedback_relative_hold'] = False
        with self.assertRaisesRegex(ValueError, 'measured relative branches'):
            fit_joint_potential(invalid, 'unused-relative-invalid.pth', updates=2, ensemble_size=1,
                                width=4, device='cpu')

    def test_conservative_catalog_rejects_an_unmeasured_optimistic_choice(self):
        from local_intervention_value import conservative_catalog_loss
        values = torch.nn.Parameter(torch.tensor([.2, .8]))
        optimizer = torch.optim.SGD([values], lr=.02)
        for _ in range(400):
            candidates = torch.cat((values.new_zeros(1), values))[None]
            observed = candidates[:, :2].mean(-1)
            loss = 20.*(values[0]-.2).square()+.2*conservative_catalog_loss(candidates, observed).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        self.assertGreater(float(values[0].detach()), .15)
        self.assertLess(float(values[1].detach()), float(values[0].detach()))
        self.assertEqual(int(torch.cat((values.new_zeros(1), values)).argmax()), 1)
        values = torch.tensor([[.1, .3, -.2], [.4, -.1, .2]], requires_grad=True)
        observed = values[:, :2].mean(-1)
        base = conservative_catalog_loss(values, observed)
        shift = torch.tensor([.6, -.2])
        shifted = conservative_catalog_loss(values+shift[:, None], observed+shift)
        torch.testing.assert_close(base, shifted, rtol=1e-5, atol=1e-7)
        with self.assertRaises(ValueError):
            conservative_catalog_loss(values, observed, temperature=0.)

    def test_conservative_matched_fits_keep_real_targets_and_reload(self):
        import io
        import pickle
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import fit_joint_potential, LocalDifferenceDecision, PlainJointReturnDecision
        roots = []
        controls = np.tile(np.array([.8, .3, -.4, .6, .1, -.2], np.float32), (2, 1))
        for index, options in enumerate(([(0, 0, 0), (1, 0, 5), (10, 3, 20)], [(0, 0, 0), (9, 0, 5)])):
            context = np.full(21, .02*index, np.float32); context[[4, 12]] = 1.
            roots.append(dict(context=context, effect_controls=controls, futures=[1, 2], root_steps=5,
                options=options, outcomes={(q, f):dict(success=float(q == f), duration=.2*(40+3*q+f),
                    following=context+.01*q, response_steps=20-f, response_sampling='physical_tick')
                    for q in range(len(options)) for f in range(2)}))
        data = dict(model={}, roots=roots, predecessor_sha256='conservative-value-test', simulation_steps=80)
        before = pickle.dumps(data); models = []
        for ordinary in (False, True):
            pair = []
            for weight in (0., .2):
                with patch('local_intervention_value.torch.save'):
                    model = fit_joint_potential(data, 'unused-conservative-test.pth', updates=3,
                        ensemble_size=1, width=4, device='cpu', ordinary=ordinary, ordinary_architecture='interaction',
                        conditional_strata=not ordinary, paired_response_encoding=not ordinary,
                        aligned_response_contrast=not ordinary, paired_event_difference=not ordinary,
                        finite_motion_basis=True, response_supervision_weight=4.,
                        response_supervision_architecture='linear', response_conditioned_value=True,
                        physics_residual_response=True, response_value_weight=1., value_target='discounted_return',
                        additional_candidate_options=[(2, 0, 5), (20, 3, 20), (9, 5, 40)],
                        conservative_catalog_weight=weight, conservative_catalog_samples=5,
                        conservative_catalog_roots=2)
                pair.append(model)
                self.assertEqual(model['successor_task_fit']['conservative_catalog_weight'], weight)
                buffer = io.BytesIO(); torch.save(model, buffer); buffer.seek(0)
                restored = torch.load(buffer, weights_only=False)
                recipe = dict(restored['successor_task_value']); recipe.pop('kind')
                learner = PlainJointReturnDecision(**recipe) if ordinary else LocalDifferenceDecision(**recipe)
                prediction = learner.models[0].forward_catalog(torch.from_numpy(roots[0]['context'][None]),
                    torch.from_numpy(controls[None]), learner.programs, learner.options[:, 1:])
                self.assertTrue(torch.isfinite(prediction).all())
            self.assertTrue(any(not torch.equal(value, pair[1]['successor_task_value']['states'][0][name])
                for name, value in pair[0]['successor_task_value']['states'][0].items()))
            self.assertEqual(pair[0]['successor_task_fit']['training_parameters_per_member'],
                             pair[1]['successor_task_fit']['training_parameters_per_member'])
            models.extend(pair)
        self.assertEqual(before, pickle.dumps(data))
        for key in ('training_observation_sha256', 'training_target_sha256', 'response_label_sha256', 'updates'):
            self.assertTrue(all(model['successor_task_fit'][key] == models[0]['successor_task_fit'][key] for model in models))

    def test_conservative_sampling_keeps_held_out_labels_out_of_gradients(self):
        import copy
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import fit_joint_potential
        controls = np.tile(np.array([.8, .3, -.4, .6, .1, -.2], np.float32), (2, 1))
        roots = []
        for index in range(6):
            context = np.full(21, .02*index, np.float32); context[[4, 12]] = 1.
            roots.append(dict(context=context, effect_controls=controls, futures=[1, 2], scene=(100+index, 2, 2.),
                options=[(0, 0, 0), (1, 0, 5), (10, 0, 5)],
                outcomes={(q, f):dict(success=float((index+q+f) % 3 != 0))
                    for q in range(3) for f in range(2)}))
        data = dict(model={}, roots=roots, predecessor_sha256='conservative-crossfit', simulation_steps=60)
        settings = dict(updates=6, ensemble_size=3, width=4, device='cpu', conditional_strata=True,
            factual_crossfit=True, paired_response_encoding=True, aligned_response_contrast=True,
            paired_event_difference=True, finite_motion_basis=True,
            conservative_catalog_weight=.2, conservative_catalog_samples=4, conservative_catalog_roots=2,
            additional_candidate_options=[(9, 5, 20), (90, 5, 20)])
        with patch('local_intervention_value.torch.save'):
            first = fit_joint_potential(data, 'unused-conservative-crossfit.pth', **settings)
        changed = copy.deepcopy(data)
        for index, fold in enumerate(first['successor_task_fit']['factual_crossfit_root_folds']):
            if fold == 0:
                for outcome in changed['roots'][index]['outcomes'].values():
                    outcome['success'] = 1.
        with patch('local_intervention_value.torch.save'):
            second = fit_joint_potential(changed, 'unused-conservative-crossfit.pth', **settings)
        for key, value in first['successor_task_value']['states'][0].items():
            torch.testing.assert_close(value, second['successor_task_value']['states'][0][key], rtol=0., atol=0.)
        self.assertNotEqual(first['successor_task_fit']['training_target_sha256'],
                            second['successor_task_fit']['training_target_sha256'])

    def test_target_scenarios_use_their_own_fleet_feedback_paths(self):
        from conditional_fusion_value import task_envelope_fusion_factors, task_stopped_fleet_cost
        paths=torch.full((1,2,2,2,1,2),20.)
        paths[...,1]=0.
        paths[0,0,0,0,0,0]=0.
        paths[0,1,1,1,0,0]=10.
        targets=torch.tensor([[[[0.,0.]],[[10.,0.]]]])
        factors=task_envelope_fusion_factors(paths,targets,torch.zeros(1,2,2),torch.zeros_like,
            weights=[.5,.5],interception_weight=1.,control_stride=1,capture_radius=1.)
        plans=torch.tensor([[[0,0],[0,1],[1,0],[1,1]]])
        costs=task_stopped_fleet_cost(factors,plans,risk_weight=0.)
        torch.testing.assert_close(costs,torch.tensor([[4.5,0.,14.,9.5]]))

    def test_feedback_forecast_refreshes_from_the_supplied_public_target_future(self):
        from types import SimpleNamespace
        from conditional_fusion_value import coupled_actuator_risk
        packet=dict(motion=torch.zeros(1,2,6),water=torch.zeros(1,2,3),node=torch.zeros(1,2,16))
        observed=[]
        def feedback(packet,state,time,joint=False,target_override=None):
            observed.append((time,target_override))
            return torch.zeros(1,1,2,4),torch.zeros(1,1,2,2)
        actor=SimpleNamespace(prediction_step=.05,risk_horizon=.6,risk_feedback=True,
            feedback_observation=feedback,proposal_distribution=lambda obs:(torch.zeros(1,2,2),None))
        future=torch.stack((torch.arange(1,13)*.05,torch.arange(1,13)*.1),-1)[None]
        coupled_actuator_risk(actor,packet,torch.zeros(1,2,2),torch.ones(1,1,2,2),'paths',
            synchronized=True,target_paths=future)
        self.assertEqual(len(observed),2)
        for time,(position,velocity) in observed:
            torch.testing.assert_close(position,torch.tensor([[time,2*time]]))
            torch.testing.assert_close(velocity,torch.tensor([[1.,2.]]))
        with self.assertRaisesRegex(ValueError,'align'):
            coupled_actuator_risk(actor,packet,torch.zeros(1,2,2),torch.ones(1,1,2,2),'paths',
                synchronized=True,target_paths=future[:,:-1])

    def test_feedback_library_exposes_every_reference_change_without_joint_selection(self):
        from conditional_fusion_value import feedback_fleet_library, select_feedback_fleet_plan
        unary=torch.ones(2,2,9);unary[...,0]=0.
        library,start=feedback_fleet_library(unary)
        self.assertEqual(start,81)
        self.assertEqual(len(torch.unique(library[0,:start],dim=0)),81)
        for vessel in range(2):
            local=library[:,start+9*vessel:start+9*(vessel+1)]
            torch.testing.assert_close(local[:,:,vessel],torch.arange(9)[None].expand(2,-1))
            self.assertTrue((local[:,:,1-vessel]==0).all())
        costs=torch.full(library.shape[:2],10.)
        costs[(library==0).all(-1)]=4.
        costs[((library==0).sum(-1)==1)&((library==1).sum(-1)==1)]=1.
        costs[(library==1).all(-1)]=5.
        independent=select_feedback_fleet_plan(costs,library,start,'single_reference')
        joint=select_feedback_fleet_plan(costs,library,start,'joint')
        self.assertTrue((independent==1).all())
        self.assertTrue(((joint==0).sum(-1)==1).all())
        self.assertTrue(((joint==1).sum(-1)==1).all())
        with self.assertRaises(ValueError):
            select_feedback_fleet_plan(costs,library,start,'independent')
        unary=torch.ones(1,3,9);unary[...,0]=0.
        library,start=feedback_fleet_library(unary)
        costs=torch.full(library.shape[:2],10.)
        for vessel,choice in enumerate((1,2,3)):
            costs[:,start+9*vessel+choice]=1.
        independent=select_feedback_fleet_plan(costs,library,start,'single_reference')
        self.assertFalse(((library-independent[:,None]).abs().sum(-1)==0).any())
        completed=torch.cat((library,independent[:,None]),1)
        completed_costs=torch.cat((costs,torch.full((1,1),.5)),1)
        torch.testing.assert_close(select_feedback_fleet_plan(completed_costs,completed,start,'single_reference'),independent)
        torch.testing.assert_close(select_feedback_fleet_plan(completed_costs,completed,start,'joint'),independent)

    def test_complete_feedback_paths_preserve_the_existing_collision_forecast(self):
        from types import SimpleNamespace
        from conditional_fusion_value import coupled_actuator_risk
        packet=dict(motion=torch.zeros(1,2,6),water=torch.zeros(1,2,3),
            node=torch.zeros(1,2,16),task=torch.zeros(1,8))
        packet['motion'][0,1,1]=2.
        packet['node'][...,11:13]=-1./3.
        actor=SimpleNamespace(prediction_step=.05,risk_horizon=.4,risk_feedback=False,
            distance_risk=lambda distance:(7.-distance).clamp_min(0.).square())
        source=torch.ones(1,2,2)
        plans=torch.stack((torch.zeros(1,2,2),torch.ones(1,2,2)),1)
        paths=coupled_actuator_risk(actor,packet,source,plans,'paths',synchronized=True)
        cost,clearance=coupled_actuator_risk(actor,packet,source,plans,'both',synchronized=True)
        separation=(paths[:,:,0]-paths[:,:,1]).norm(dim=-1).amin(-1)
        torch.testing.assert_close(clearance,separation,rtol=0.,atol=0.)
        torch.testing.assert_close(cost,actor.distance_risk(separation),rtol=0.,atol=0.)
        packet['task'].fill_(1000.)
        torch.testing.assert_close(paths,coupled_actuator_risk(actor,packet,source,plans,'paths',
            synchronized=True),rtol=0.,atol=0.)

    def test_target_envelope_cost_matches_exact_independent_expectation(self):
        import itertools
        from conditional_fusion_value import (task_envelope_fusion_factors, task_stopped_fleet_cost,
            task_stopped_independent_cost, select_task_stopped_fusion)
        rng = torch.Generator().manual_seed(161)
        paths = 8.*torch.rand(2, 3, 2, 4, 2, generator=rng, dtype=torch.float64)+3.
        targets = torch.zeros(2, 3, 4, 2, dtype=torch.float64)
        targets[:, :, :, 0] = torch.tensor([8., 10., 12.])[None, :, None]
        paths[0, 0, 0, 1] = torch.tensor([10., 0.])
        unary = torch.rand(2, 3, 2, generator=rng, dtype=torch.float64)
        probability = torch.rand(2, 3, 2, generator=rng, dtype=torch.float64).softmax(-1)
        factors = task_envelope_fusion_factors(paths, targets, unary,
            lambda distance: (7.-distance).clamp_min(0.).square(),
            weights=[.2, .5, .3], interception_weight=1.5, control_stride=2)
        plans = torch.tensor(list(itertools.product(range(2), repeat=3)))[None].expand(2, -1, -1)
        measured = task_stopped_fleet_cost(factors, plans, risk_weight=2., escape_weight=3.)
        expected = task_stopped_independent_cost(factors, probability, risk_weight=2., escape_weight=3.)
        for subject in range(3):
            for choice in range(2):
                mass = torch.ones(2, len(plans[0]), dtype=torch.float64)
                for peer in range(3):
                    if peer != subject:
                        mass *= probability[:, peer].gather(1, plans[:, :, peer])
                mass *= plans[:, :, subject] == choice
                torch.testing.assert_close(expected[:, subject, choice], (mass*measured).sum(-1),
                    rtol=1e-10, atol=1e-10)
        selected = select_task_stopped_fusion(factors, selection='joint', beam_width=8,
            risk_weight=2., escape_weight=3.)
        torch.testing.assert_close(task_stopped_fleet_cost(factors, selected[:, None], 2., 3.).squeeze(1),
            measured.amin(-1), rtol=1e-10, atol=1e-10)

    def test_shared_target_futures_require_complementary_fleet_choices(self):
        from conditional_fusion_value import (target_turning_envelope, task_envelope_fusion_factors,
            task_stopped_fleet_cost, select_task_stopped_fusion)
        paths = torch.tensor([[[[[0., 0.]], [[10., 0.]]],
                               [[[0., 0.]], [[10., 0.]]]]])
        targets = torch.tensor([[[[0., 0.]], [[10., 0.]]]])
        unary = torch.tensor([[[0., .1], [0., .1]]])
        factors = task_envelope_fusion_factors(paths, targets, unary, torch.zeros_like,
            weights=[.5, .5], interception_weight=1., control_stride=1, capture_radius=1.)
        joint = select_task_stopped_fusion(factors, selection='joint', beam_width=4,
            risk_weight=0., escape_weight=0.)
        independent = select_task_stopped_fusion(factors, selection='independent', beam_width=4,
            risk_weight=0., escape_weight=0.)
        self.assertEqual(joint.sum().item(), 1)
        self.assertEqual(independent.sum().item(), 2)
        self.assertLess(task_stopped_fleet_cost(factors, joint[:, None], 0.).item(),
                        task_stopped_fleet_cost(factors, independent[:, None], 0.).item())
        position = torch.tensor([[7., 4.]])
        center = position[:, None]+torch.tensor([[[1., 0.], [2., 0.], [3., 0.], [4., 0.]]])
        envelope, weights = target_turning_envelope(position, center, 30.)
        torch.testing.assert_close(envelope[:, 2], center)
        torch.testing.assert_close((envelope-position[:, None, None]).norm(dim=-1),
            (center-position[:, None]).norm(dim=-1)[:, None].expand(-1, 5, -1))
        torch.testing.assert_close(weights.sum(), torch.tensor(1.))
        static = position[:, None].expand(-1, 4, -1)
        unchanged, _ = target_turning_envelope(position, static, 30.)
        torch.testing.assert_close(unchanged, static[:, None].expand(-1, 5, -1, -1))
        with self.assertRaisesRegex(ValueError, 'ninety'):
            target_turning_envelope(position, center, 91.)

    def test_batched_delayed_value_program_preserves_native_prefix_and_activation(self):
        from types import SimpleNamespace
        import numpy as np
        from run_remote_evidence import batched_primitive_program
        actor = SimpleNamespace(successor_task_value=SimpleNamespace(prior=None))
        action = np.zeros((3, 14), np.float32)
        action[:, 3] = -1.; action[:, 4] = 8.; action[:, 5] = -2.
        action[:, 8] = 5.; action[:, 9:11] = -1.; action[:, 11:13] = .5
        action[:, 13] = 1.
        program, activation = batched_primitive_program(actor, action)
        self.assertEqual(activation, 5)
        np.testing.assert_array_equal(program, action[:, :13])
        self.assertEqual(program[0, 4]-activation, 3)
        actor.successor_task_value = None; action[:, 13] = 0.
        same, activation = batched_primitive_program(actor, action)
        np.testing.assert_array_equal(same, program)
        self.assertEqual(activation, 5)

    def test_batched_delayed_value_program_rejects_unresolved_predecessor_chain(self):
        from types import SimpleNamespace
        import numpy as np
        from run_remote_evidence import batched_primitive_program
        action = np.zeros((3, 14), np.float32); action[:, 8] = 5.; action[:, 13] = 1.
        actor = SimpleNamespace(successor_task_value=SimpleNamespace(prior=SimpleNamespace(prior=None)))
        with self.assertRaisesRegex(ValueError, 'one value stage'):
            batched_primitive_program(actor, action)
        actor.successor_task_value.prior = None; action[:, 13] = 2.
        with self.assertRaisesRegex(ValueError, 'value-stage chain'):
            batched_primitive_program(actor, action)

    def test_task_stopped_independent_matches_exhaustive_fleet_expectations(self):
        import itertools
        from conditional_fusion_value import (task_stopped_fusion_factors, task_stopped_fleet_cost,
            task_stopped_independent_cost, select_task_stopped_fusion)
        rng = torch.Generator().manual_seed(715)
        paths = 8.*torch.rand(2, 3, 2, 4, 2, generator=rng, dtype=torch.float64)+4.
        target = torch.zeros(2, 4, 2, dtype=torch.float64); target[..., 0] = 10.
        paths[0, 0, 0, 1] = torch.tensor([10., 0.])
        unary = torch.rand(2, 3, 2, generator=rng, dtype=torch.float64)
        probability = torch.rand(2, 3, 2, generator=rng, dtype=torch.float64).softmax(-1)
        factors = task_stopped_fusion_factors(paths, target, unary,
            lambda distance: (7.-distance).clamp_min(0.).square(), control_stride=2)
        plans = torch.tensor(list(itertools.product(range(2), repeat=3)))[None].expand(2, -1, -1)
        measured = task_stopped_fleet_cost(factors, plans, risk_weight=2., escape_weight=3.)
        expected = task_stopped_independent_cost(factors, probability, risk_weight=2., escape_weight=3.)
        for subject in range(3):
            for choice in range(2):
                mass = torch.ones(2, len(plans[0]), dtype=torch.float64)
                for peer in range(3):
                    if peer != subject:
                        mass *= probability[:, peer].gather(1, plans[:, :, peer])
                mass *= plans[:, :, subject] == choice
                torch.testing.assert_close(expected[:, subject, choice], (mass*measured).sum(-1),
                    rtol=1e-10, atol=1e-10)
        selected = select_task_stopped_fusion(factors, selection='joint', beam_width=8,
            risk_weight=2., escape_weight=3.)
        torch.testing.assert_close(task_stopped_fleet_cost(factors, selected[:, None], 2., 3.).squeeze(1),
            measured.amin(-1), rtol=1e-10, atol=1e-10)

    def test_task_stopping_keeps_same_tick_risk_and_removes_post_capture_risk(self):
        from conditional_fusion_value import task_stopped_fusion_factors, task_stopped_fleet_cost
        paths = torch.tensor([[[[[3., 0.], [4., 0.], [4., 0.]]],
                               [[[12., 0.], [8., 0.], [4.1, 0.]]]]])
        target = torch.zeros(1, 3, 2); unary = torch.zeros(1, 2, 1)
        risk = lambda distance: (7.-distance).clamp_min(0.).square()
        plans = torch.zeros(1, 1, 2, dtype=torch.long)
        stopped = task_stopped_fusion_factors(paths, target, unary, risk, control_stride=1)
        self.assertEqual(task_stopped_fleet_cost(stopped, plans).item(), 0.)
        continuing = task_stopped_fusion_factors(paths, target+100., unary, risk, control_stride=1)
        self.assertGreater(task_stopped_fleet_cost(continuing, plans).item(), 0.)
        paths[0, 1, 0, 0, 0] = 3.1
        simultaneous = task_stopped_fusion_factors(paths, target, unary, risk, control_stride=1)
        self.assertGreater(task_stopped_fleet_cost(simultaneous, plans).item(), 0.)
        # Both boats outside the attacker still incur source-failure cost at
        # the capture tick, which precedes capture in source termination.
        target[..., 0] = 2.
        self.assertGreater(task_stopped_fleet_cost(
            task_stopped_fusion_factors(paths, target, unary, risk, control_stride=1),
            plans, risk_weight=0., escape_weight=1.).item(), 0.)

    def test_breadth_design_pairs_every_root_without_using_outcome_labels(self):
        import copy
        import numpy as np
        from local_intervention_value import breadth_local_design, local_edges
        roots = [dict(scene=(100+i, 3, 2.), prefix_future=None, root_steps=0,
                      options=[(0, 0, 0)], futures=[None],
                      outcomes={(0, 0): dict(success=float(i % 2), duration=5+i)})
                 for i in range(8)]
        catalog = [(0, 0, 0)]+[(code, delay, span)
            for code in (1, 3, 10, 30, 11, 13, 31, 33, 111, 113, 131, 133)
            for delay, span in ((0, 5), (5, 10))]
        first = breadth_local_design(dict(roots=roots), catalog, pairs_per_root=3,
                                     repetitions=2, seed=73)
        changed = copy.deepcopy(roots)
        for root in changed:
            root['outcomes'] = {(0, 0): dict(success=999., duration=999.)}
        second = breadth_local_design(dict(roots=changed), catalog, pairs_per_root=3,
                                      repetitions=2, seed=73)
        self.assertEqual(first['acquisition_design'], second['acquisition_design'])
        self.assertEqual(len(first['roots']), len(roots))
        for a, b, original in zip(first['roots'], second['roots'], roots):
            self.assertEqual(a['options'], b['options'])
            self.assertEqual(a['futures'], b['futures'])
            self.assertEqual(a['outcomes'], b['outcomes'])
            self.assertEqual(a['scene'], original['scene'])
            self.assertGreater(len(a['options']), 1)
            self.assertLessEqual(len(a['outcomes']), 14)
            self.assertTrue(set(a['options']) <= set(catalog))
            self.assertTrue(local_edges(a['options'], 3))
            self.assertTrue(all('success' not in row for row in a['outcomes'].values()))
            np.testing.assert_array_equal(a['future_weights'], np.ones(2))
            for q in range(len(a['options'])):
                for future in range(2):
                    self.assertIn((q, future), a['outcomes'])
        self.assertEqual(roots[0]['outcomes'][(0, 0)]['success'], 0.)
        third = breadth_local_design(dict(roots=roots), catalog, seed=74)
        self.assertNotEqual(first['acquisition_design']['design_sha256'],
                            third['acquisition_design']['design_sha256'])

    def test_confirmation_uses_fresh_futures_and_preserves_all_original_outcomes(self):
        import copy
        import numpy as np
        from local_intervention_value import confirmation_local_design
        options = [(0, 0, 0), (1, 0, 5), (2, 0, 5), (10, 5, 10), (20, 5, 10)]
        outcomes = {(q, f): dict(success=float(value), duration=10.)
                    for q, pair in enumerate(((0, 1), (1, 1), (1, 0), (0, 0), (1, 1)))
                    for f, value in enumerate(pair)}
        root = dict(scene=(100, 3, 2.), prefix_future=9, root_steps=5, options=options,
                    futures=[17, 29], outcomes=outcomes, root_snapshot=dict(root_steps=5))
        adverse = copy.deepcopy(root); adverse['scene'] = (101, 3, 2.)
        adverse['outcomes'] = {(q, f): dict(success=float(q == 0), duration=10.)
                               for q, f in outcomes}
        source = dict(completed=True, roots=[root, adverse]); before = copy.deepcopy(source)
        configuration = dict(source_data_sha256='ab'*32, candidates=2, repetitions=5,
                             seed=73, value_target='success')
        first = confirmation_local_design(source, **configuration)
        second = confirmation_local_design(source, **configuration)
        self.assertEqual(source, before)
        self.assertEqual(first['acquisition_design'], second['acquisition_design'])
        self.assertEqual(first['roots'][0]['options'], [options[q] for q in (0, 1, 4)])
        self.assertEqual(first['roots'][1]['options'], options[:3])
        self.assertFalse(first['acquisition_design']['root_outcome_filtering'])
        self.assertEqual(first['acquisition_design']['planned_observations'], 30)
        self.assertTrue(first['acquisition_design']['same_observations_for_ordinary'])
        for scheduled, original in zip(first['roots'], source['roots']):
            self.assertEqual(len(scheduled['futures']), 5)
            self.assertFalse(set(scheduled['futures']) & set(original['futures']))
            self.assertIs(scheduled['root_snapshot'], original['root_snapshot'])
            self.assertTrue(all('success' not in row for row in scheduled['outcomes'].values()))
            np.testing.assert_array_equal(scheduled['future_weights'], np.ones(5))
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import local_intervention_value as local
        def measured(task):
            rows = []
            for index, scheduled in task[1]:
                observed = copy.deepcopy(scheduled)
                observed['design_root'] = index
                observed['outcomes'] = {key: dict(success=float(key[0] == 0), duration=7.)
                                        for key in scheduled['outcomes']}
                rows.append((observed, 10+index))
            return rows
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory)/'policy.pth'; output = Path(directory)/'observations.pth'
            torch.save(dict(successor_task_value=None), policy)
            settings = dict(workers=1, root_batch_size=2, inference_batch=32,
                            physical_response_horizon=20)
            with patch.object(local, '_recollect_snapshot_root_batch', side_effect=measured) as worker:
                observed = local.recollect_breadth_design(policy, first, output, **settings)
                self.assertTrue(observed['completed'])
                self.assertEqual(observed['measured_terminal_observations'], 30)
                self.assertEqual(observed['simulation_steps'], 21)
                self.assertTrue(all(row['success'] == float(q == 0)
                    for root in observed['roots'] for (q, _), row in root['outcomes'].items()))
                local.recollect_breadth_design(policy, first, output, **settings)
                self.assertEqual(worker.call_count, 1)
            self.assertEqual(source, before)
        incomplete = copy.deepcopy(source)
        del incomplete['roots'][0]['outcomes'][0, 1]
        with self.assertRaisesRegex(ValueError, 'at least two actual paired futures'):
            confirmation_local_design(incomplete, **configuration)
        joint = copy.deepcopy(source); joint['roots'][0]['options'][1] = (11, 0, 5)
        with self.assertRaisesRegex(ValueError, 'single-vessel alternatives'):
            confirmation_local_design(joint, **configuration)

    def test_breadth_replay_resumes_exact_roots_and_preserves_negative_labels(self):
        import copy
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import local_intervention_value as local
        design = local.breadth_local_design(dict(roots=[dict(scene=(i, 3, 2.),
            prefix_future=None, root_steps=0) for i in range(3)]),
            [(0, 0, 0), (1, 0, 5), (11, 0, 5)], pairs_per_root=1, repetitions=2, seed=4)
        def measured(task):
            _, index, root, _, _ = task
            row = copy.deepcopy(root)
            row['design_root'] = index
            row['outcomes'] = {key: dict(success=float(index == 1), duration=7.)
                               for key in root['outcomes']}
            return row, 10+index
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory)/'policy.pth'; output = Path(directory)/'observations.pth'
            torch.save(dict(successor_task_value=None), policy)
            with patch.object(local, '_recollect_breadth_root', side_effect=measured) as worker:
                complete = local.recollect_breadth_design(policy, design, output, workers=1)
                self.assertEqual(worker.call_count, 3)
                self.assertEqual(complete['simulation_steps'], 33)
                self.assertTrue(complete['all_design_roots_measured'])
                self.assertEqual(sum(item['success'] for item in complete['roots'][0]['outcomes'].values()), 0.)
                local.recollect_breadth_design(policy, design, output, workers=1)
                self.assertEqual(worker.call_count, 3)
                partial = dict(complete, completed=False, roots=complete['roots'][:1],
                               next_design_root=1, simulation_steps=10)
                torch.save(partial, output)
                resumed = local.recollect_breadth_design(policy, design, output, workers=1)
                self.assertEqual(worker.call_count, 5)
                self.assertEqual(resumed['simulation_steps'], 33)
                self.assertEqual([root['design_root'] for root in resumed['roots']], [0, 1, 2])
                changed = copy.deepcopy(design)
                changed['acquisition_design']['design_sha256'] = 'changed'
                with self.assertRaisesRegex(ValueError, 'different predecessor, design'):
                    local.recollect_breadth_design(policy, changed, output, workers=1)

    def test_breadth_batch_resize_preserves_completed_paired_roots(self):
        import copy
        import hashlib
        import pickle
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        import local_intervention_value as local
        design = local.breadth_local_design(dict(roots=[dict(scene=(i, 3, 2.),
            prefix_future=None, root_steps=0, root_snapshot=dict(root_steps=0)) for i in range(4)]),
            [(0, 0, 0), (1, 0, 5), (2, 0, 5)], pairs_per_root=1, repetitions=2, seed=9)
        def measured(task):
            rows = []
            for index, root in task[1]:
                row = copy.deepcopy(root)
                row['design_root'] = index
                row['outcomes'] = {key: dict(success=float(index % 2), duration=7.)
                                   for key in root['outcomes']}
                rows.append((row, 10+index))
            return rows
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory)/'policy.pth'; output = Path(directory)/'observations.pth'
            torch.save(dict(successor_task_value=None), policy)
            settings = dict(workers=1, root_batch_size=2, inference_batch=32,
                            physical_response_horizon=20)
            with patch.object(local, '_recollect_snapshot_root_batch', side_effect=measured):
                complete = local.recollect_breadth_design(policy, design, output, **settings)
            partial = dict(complete, completed=False, roots=complete['roots'][:2],
                           next_design_root=2, simulation_steps=21)
            self.assertEqual(complete['physical_response_capture_runs'], [
                dict(first_design_root=0, kind='fixed_horizon_and_separate_window_end')])
            partial.pop('physical_response_capture_runs')
            torch.save(partial, output)
            preserved = pickle.dumps(partial['roots'])
            native_digest = hashlib.sha256(output.read_bytes()).hexdigest()
            settings['inference_batch'] = 128
            with patch.object(local, '_recollect_snapshot_root_batch', side_effect=measured) as worker:
                with self.assertRaisesRegex(ValueError, 'different predecessor, design'):
                    local.recollect_breadth_design(policy, design, output, **settings)
                self.assertEqual(worker.call_count, 0)
                self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), native_digest)
                resumed = local.recollect_breadth_design(policy, design, output,
                    resume_inference_batch=True, **settings)
                self.assertEqual(worker.call_count, 1)
                self.assertEqual(worker.call_args.args[0][2], 128)
                self.assertEqual([index for index, _ in worker.call_args.args[0][1]], [2, 3])
            self.assertEqual(pickle.dumps(resumed['roots'][:2]), preserved)
            self.assertEqual(resumed['simulation_steps'], 46)
            self.assertEqual(resumed['inference_batch_runs'], [
                dict(first_design_root=0, inference_batch=32, inference_device='cpu'),
                dict(first_design_root=2, inference_batch=128, inference_device='cpu')])
            self.assertEqual(resumed['physical_response_capture_runs'], [
                dict(first_design_root=0, kind='maximum_of_window_end_and_horizon'),
                dict(first_design_root=2, kind='fixed_horizon_and_separate_window_end')])
            self.assertTrue(resumed['completed'])

    def test_batched_physical_response_keeps_window_end_and_all_actual_outcomes(self):
        import sys
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import patch
        import run_remote_evidence as remote
        class Environment:
            def __init__(self, n, protocol):
                self.n = n
            def reset(self, agility, noisy_agility):
                self.limit = int(agility); self.steps = 0; self.Current_T = 0.
                self.state = float(np.random.normal())
                return np.array([self.state]), {}
        class Actor:
            gate_period = 1; intent_strength = 0.; training_gamma = .97
            training_step_cost = .01; environment_protocol = 'test'
            successor_task_value = None; cooperative_role_credit = None
            response_feedback_value = None
            def to(self, device):
                return self
            def cpu(self):
                return self
            def __call__(self, packets, **kwargs):
                return packets['state'][:, None, :].expand(-1, 2, 3), None
        def packet(env, observation):
            return dict(state=np.array([env.state], np.float32),
                        clock=np.array([env.steps], np.float32))
        def step(env, controls, gamma, cost):
            env.state += float(controls.mean())*.01+float(np.random.normal())
            env.steps += 1; env.Current_T = env.steps*.05
            return np.array([env.state]), env.state, env.steps == env.limit, env.state*.5
        def outcome(env, done):
            return dict(success=int(env.state > 0.), outcome=float(env.state),
                        duration=env.Current_T)
        tasks = [((17, 2, 55.), 100), ((18, 2, 7.), 101)]
        with patch.dict(sys.modules, {'envs.TADgame': SimpleNamespace(TADEnv=Environment)}), \
                patch.object(remote, 'initialize'), patch.object(remote.d, 'joint_packet', side_effect=packet), \
                patch.object(remote.d, 'joint_task_step', side_effect=step), \
                patch.object(remote.d, 'outcome', side_effect=outcome):
            original = remote.batched_credit_rows('unused', tasks, device='cpu',
                actor_override=Actor(), response_tick=40, keep_steps=True)
            original_rng = np.random.get_state()
            measured = remote.batched_credit_rows('unused', tasks, device='cpu',
                actor_override=Actor(), response_tick=40, physical_response_tick=20, keep_steps=True)
            measured_rng = np.random.get_state()
            uniform = remote.batched_credit_rows('unused', tasks, device='cpu',
                actor_override=Actor(), root_sample_seed=19, keep_steps=True)
            uniform_rng = np.random.get_state()
            seen = []
            def selector(public, factual_actor):
                seen.extend(public['clock'][:, 0].tolist())
                return public['clock'][:, 0] >= 10
            directed = remote.batched_credit_rows('unused', tasks, device='cpu',
                actor_override=Actor(), root_sample_seed=19, root_selector=selector, keep_steps=True)
            directed_rng = np.random.get_state()
        np.testing.assert_equal(original_rng, measured_rng)
        for before, after, expected in zip(original, measured, (20, 7)):
            physical = after['row'].pop('physical_response')
            self.assertEqual(physical['steps'], expected)
            self.assertEqual(physical['packet']['clock'].item(), expected)
            self.assertEqual(physical['terminal'], int(expected == 7))
            np.testing.assert_equal(before, after)
        self.assertEqual(measured[0]['row']['response']['steps'], 40)
        self.assertEqual(measured[1]['row']['response']['steps'], 7)
        np.testing.assert_equal(uniform_rng, directed_rng)
        self.assertEqual(seen, [0., 0., 5., 5., 10.])
        for index, (before, after) in enumerate(zip(uniform, directed)):
            sampled = before.pop('root_snapshot'); chosen = after.pop('root_snapshot')
            np.testing.assert_equal(before, after)
            self.assertEqual(chosen['eligible_states'], sampled['eligible_states'])
            if index == 0:
                self.assertEqual(chosen['root_steps'], 10)
                self.assertEqual(chosen['selection_reason'], 'first_accepted_proposal')
                self.assertEqual(chosen['environment'].steps, 10)
            else:
                self.assertEqual(chosen['selection_reason'], 'uniform_eligible_factual_state')
                self.assertEqual(chosen['root_steps'], sampled['root_steps'])
                np.testing.assert_equal(chosen['random_state'], sampled['random_state'])

    def test_generic_candidate_paths_share_task_coordinates(self):
        from mechanism_controls import generic_shared_task_features
        features=torch.zeros(1,2,1,84)
        features[0,:,0,:2]=torch.tensor([[1.,0.],[0.,1.]])
        features[0,:,0,8:10]=torch.tensor([[-1.,0.],[-1.,0.]])
        # Both paths end at task position (2, 3), with displacements in
        # the original world coordinates and the original 15-m scale.
        features[0,0,0,20:]=torch.tensor([2.,6.]).repeat(32)
        features[0,1,0,20:]=torch.tensor([4.,4.]).repeat(32)
        transformed=generic_shared_task_features(features)
        torch.testing.assert_close(transformed[...,:20],features[...,:20])
        torch.testing.assert_close(transformed[...,20:].reshape(1,2,1,32,2),
            torch.tensor([2.,3.]).expand(1,2,1,32,2))
        # Rotate the original world frame by 90 degrees. The shared
        # task paths remain unchanged when bearing and paths rotate together.
        rotated=features.clone();rotated[...,8:10]=torch.tensor([0.,-1.])
        vectors=features[...,20:].reshape(1,2,1,32,2)
        rotated[...,20:]=torch.stack((-vectors[...,1],vectors[...,0]),-1).flatten(-2)
        torch.testing.assert_close(generic_shared_task_features(rotated)[...,20:],transformed[...,20:])

    def test_distillation_targets_retain_equivalent_controls_and_reject_mismatch(self):
        from mechanism_controls import equivalent_fusion_choices
        candidates=torch.tensor([[[[0.,0.],[.5,.5],[1.,1.]],[[0.,0.],[.5,.5],[1.,1.]]]])
        source=torch.tensor([[[1.,-1.],[.2,.3]]]);rule=torch.tensor([[[-1.,1.],[.2,.3]]])
        command=torch.tensor([[[0.,0.],[.2,.3]]])
        self.assertEqual(equivalent_fusion_choices(candidates,source,rule,command).tolist(),
            [[[False,True,False],[True,True,True]]])
        command[0,0]=torch.tensor([.7,.7])
        with self.assertRaisesRegex(ValueError,'common candidate library'):
            equivalent_fusion_choices(candidates,source,rule,command)

    def test_joint_loader_rejects_descriptive_ablation_metadata(self):
        from unittest.mock import patch
        import develop_source_fusion as d
        d.joint_load.cache_clear()
        with patch.object(d.torch,'load',return_value={'mechanism_ablation':{'component':'no-factual-anchor'}}), \
             patch.object(d,'JointLearningActor') as constructor:
            with self.assertRaisesRegex(ValueError,'Boolean removal flags'):
                d.joint_load('invalid-component-metadata.pth',0)
            constructor.assert_not_called()

    def test_fixed_opponent_binding_restores_apf_and_respects_explicit_actions(self):
        import tempfile
        import numpy as np
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch
        import run_remote_evidence as r
        class Environment:
            def step(self, rl_action, controller='Boids', att_action=None):
                return controller, att_action
        original = Environment.step
        opponent = lambda observation: np.array([.3, -.2])
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(r.d, 'numerical_source', return_value=SimpleNamespace(TADEnv=Environment)), \
             patch.object(r, 'FrozenSourceAttacker', return_value=opponent), \
             patch.object(r, 'public_attacker_observation', return_value=np.zeros(8)):
            path = Path(directory)/'attacker.pth'
            torch.save(dict(completed=True), path)
            try:
                self.assertEqual(len(r.configure_source_attacker(path)), 64)
                env = Environment()
                controller, action = env.step(None)
                self.assertEqual(controller, 'Boids')
                np.testing.assert_array_equal(action, [.3, -.2])
                explicit = np.array([-.9, .1])
                self.assertIs(env.step(None, controller='AdaRes', att_action=explicit)[1], explicit)
            finally:
                r.configure_source_attacker(None)
            self.assertIs(Environment.step, original)
            self.assertEqual(Environment().step(None), ('Boids', None))

    def test_opponent_mixture_retains_distinct_physical_experiments(self):
        import copy
        import numpy as np
        from local_intervention_value import merge_paired_data
        root = dict(scene=(42, 2, 2.), prefix_future=None, root_steps=0,
            context=np.zeros(21), effect_controls=np.zeros((2, 6)), options=[(0, 0, 0)],
            futures=[None], outcomes={(0, 0):dict(success=1.)})
        original = dict(model={}, predecessor_sha256='same', completed=True, roots=[root])
        changed = copy.deepcopy(original)
        changed['opponent_sha256'] = 'learned-opponent'
        changed['roots'][0]['outcomes'][0, 0]['success'] = 0.
        with self.assertRaisesRegex(ValueError, 'opponent populations'):
            merge_paired_data([original, changed])
        mixture = merge_paired_data([original, changed], allow_opponent_mixture=True)
        self.assertEqual(len(mixture['roots']), 2)
        self.assertEqual([r['outcomes'][0, 0]['success'] for r in mixture['roots']], [1., 0.])
        again = merge_paired_data([mixture], allow_opponent_mixture=True)
        self.assertEqual(len(again['roots']), 2)

    def test_attacker_observation_uses_three_nearest_sensed_boats(self):
        import numpy as np
        from types import SimpleNamespace
        from run_remote_evidence import public_attacker_observation, source_attacker_reward
        env = SimpleNamespace(attacker=SimpleNamespace(pos=np.array([10., 0.]), theta=0.),
            defender_list=[SimpleNamespace(pos=np.array(point, float)) for point in
                           [(30., 0.), (13., 0.), (10., 2.), (6., 0.), (10., 20.), (10., -10.)]],
            Att_Sensing_R=15., LearningSide='Def')
        observation = public_attacker_observation(env)
        np.testing.assert_allclose(observation[[0, 2, 4, 6]], [10., 2., 3., 4.])
        np.testing.assert_allclose(observation[[1, 3, 5, 7]], [-np.pi, np.pi/2, 0., np.pi])
        env._get_rewards = lambda outcome: 7. if env.LearningSide == 'Att' else -3.
        self.assertEqual(source_attacker_reward(env, 1), 7.)
        self.assertEqual(env.LearningSide, 'Def')

    def test_joint_task_step_preserves_default_and_explicit_attacker_execution(self):
        import numpy as np
        from types import SimpleNamespace
        from develop_source_fusion import joint_task_step
        calls = []
        env = SimpleNamespace(def_att_dists=np.array([12., 15.]))
        def step(action, controller, **kwargs):
            calls.append((action, controller, kwargs))
            return np.ones((2, 18)), np.array([2., 4.]), 3, np.zeros(8)
        env.step = step
        action, attacker = np.zeros((2, 3)), np.array([.1, -.2])
        ordinary = joint_task_step(env, action)
        explicit = joint_task_step(env, action, attacker_action=attacker)
        self.assertEqual(calls[0][2], {})
        self.assertIs(calls[1][2]['att_action'], attacker)
        self.assertEqual(ordinary[1:], explicit[1:])

    def test_shared_conditional_fleet_catalog_preserves_roles_and_single_boat_choices(self):
        from types import SimpleNamespace
        from local_intervention_value import LocalDifferenceDecision
        options = torch.tensor([[0, 0, 0], [900000, 0, 10], [777777, 0, 20], [30001, 5, 10]])
        decision = SimpleNamespace(options=options, programs=(options[:, :1]//(10**torch.arange(6))) % 10,
                                   _fleet_catalogs={})
        for defenders in (3, 8):
            programs, timing = LocalDifferenceDecision.cross_fleet_catalog(decision, defenders)
            self.assertEqual(programs.shape[1], defenders)
            self.assertTrue((programs[0] == 0).all())
            self.assertEqual(int((programs == 0).all(-1).sum()), 1)
            for vessel in range(defenders):
                single = torch.zeros(defenders, dtype=torch.long); single[vessel] = 9
                self.assertTrue(bool(((programs == single).all(-1) & (timing == torch.tensor([0, 10])).all(-1)).any()))
            self.assertTrue(bool(((programs == 7).all(-1) & (timing == torch.tensor([0, 20])).all(-1)).any()))
            self.assertEqual(len(torch.unique(torch.cat((programs, timing), -1), dim=0)), len(programs))

    def test_cross_fleet_decisions_use_shared_values_without_six_boat_empirical_labels(self):
        from unittest.mock import patch
        from local_intervention_value import JointTaskPotential, LocalDifferenceDecision
        net = JointTaskPotential(width=4)
        learner = LocalDifferenceDecision(width=4, states=[net.state_dict()],
            contexts=torch.zeros((1, 53)), controls=torch.zeros((1, 6, 6)),
            options=[(0, 0, 0), (9, 0, 10), (999999, 0, 10)],
            corrections=torch.zeros((1, 3)), uncertainty=torch.zeros((1, 3)),
            observed=torch.zeros((1, 3), dtype=torch.bool), fleet_transfer='shared_neural',
            confidence=.5, global_penalty=.2, margin=.02, intervention_budget=1)
        class KnownValues(torch.nn.Module):
            motion_basis = False
            coefficients = net.coefficients
            def forward(self, context, controls, programs, timing, responses=None):
                return context.new_full((len(context),), .1).logit()
            def forward_catalog(self, context, controls, programs, timing, responses=None):
                return torch.where((programs == 9).all(-1), .9, .1).logit()[None].expand(len(context), -1)
        learner.models = torch.nn.ModuleList([KnownValues()])
        packet = dict(node=torch.zeros((1, 3, 16)), _local_difference_controls=torch.zeros((1, 3, 6)),
                      _conditional_count=torch.zeros((1, 1)))
        action = torch.rand((1, 3, 3))
        with patch('conditional_fusion_value.intervention_task_context',
                   return_value=(torch.zeros((1, 29)), torch.tensor([[2, 0, 1]]))), \
             patch.object(learner, 'continuation_statistics', side_effect=AssertionError('six-boat factual labels')), \
             patch.object(learner, 'context_relation', side_effect=AssertionError('six-boat local labels')):
            selected = learner(packet, action)
            torch.testing.assert_close(selected[..., :3], action)
            self.assertTrue((selected[..., 9:13] == 1.).all())
            self.assertTrue((selected[..., 4] == 10.).all())
            packet['_conditional_count'].fill_(1)
            torch.testing.assert_close(learner(packet, action), action)

    def test_cross_fleet_strata_mix_repair_and_damage_at_calibrated_factual_risk(self):
        from unittest.mock import patch
        from local_intervention_value import JointTaskPotential, LocalDifferenceDecision
        net = JointTaskPotential(width=4, output_width=3)
        learner = LocalDifferenceDecision(width=4, states=[net.state_dict()], conditional_strata=True,
            contexts=torch.zeros((1, 53)), controls=torch.zeros((1, 6, 6)),
            options=[(0, 0, 0), (9, 0, 10), (999999, 0, 10)],
            corrections=torch.zeros((1, 3)), uncertainty=torch.zeros((1, 3)),
            observed=torch.zeros((1, 3), dtype=torch.bool), fleet_transfer='shared_neural',
            neural_factual_calibration=[[.1, .7]], confidence=.5, global_penalty=.2, margin=.08)
        class KnownStrata(torch.nn.Module):
            motion_basis = False
            coefficients = net.coefficients
            def forward(self, context, controls, programs, timing, responses=None):
                return context.new_tensor([.99, .5, .5]).logit()[None].expand(len(context), -1)
            def forward_catalog(self, context, controls, programs, timing, responses=None):
                good = (programs == 9).all(-1)
                probabilities = torch.stack((torch.full_like(good, .99, dtype=context.dtype),
                    torch.where(good, .9, .1), torch.where(good, .05, .9)), -1)
                return probabilities.logit()[None].expand(len(context), -1, -1)
        learner.models = torch.nn.ModuleList([KnownStrata()])
        for defenders in (3, 4, 5, 7, 8):
            context = torch.zeros((1, 8*defenders+5))
            controls = torch.zeros((1, defenders, 6))
            programs, timing = learner.cross_fleet_catalog(defenders)
            self.assertEqual(net.forward_catalog(context, controls, programs, timing).shape,
                             (1, len(programs), 3))
            packet = dict(node=torch.zeros((1, defenders, 16)), _local_difference_controls=controls,
                          _conditional_count=torch.zeros((1, 1)))
            action = torch.rand((1, defenders, 3))
            with patch('conditional_fusion_value.intervention_task_context',
                       return_value=(context, torch.arange(defenders)[None])), \
                 patch.object(learner, 'continuation_statistics', side_effect=AssertionError('six-boat labels')), \
                 patch.object(learner, 'context_relation', side_effect=AssertionError('six-boat local labels')):
                learner.neural_factual_calibration = torch.tensor([[.1, .7]])
                selected = learner(packet, action)
                torch.testing.assert_close(selected[..., :3], action)
                self.assertTrue((selected[..., 9:13] == 1.).all())
                self.assertTrue((selected[..., 4] == 10.).all())
                # The calibrated marginal gain is .306*.9-.694*.05=.2407.
                # The uncalibrated gain is .01*.9-.99*.05=-.0405.
                learner.neural_factual_calibration = None
                torch.testing.assert_close(learner(packet, action), action)

    def test_independent_interception_cost_matches_complete_fleet_expectation(self):
        from itertools import product
        from conditional_fusion_value import independent_fleet_minimum_cost
        values = torch.tensor([[[0., 2., 2.], [1., 3., 5.], [0., 4., 6.]]])
        probabilities = torch.tensor([[[.2, .3, .5], [.5, 0., .5], [.1, .7, .2]]])
        expected = torch.zeros_like(values)
        for vessel in range(3):
            others = [i for i in range(3) if i != vessel]
            for candidate in range(3):
                for chosen in product(range(3), repeat=2):
                    mass = probabilities[0, others[0], chosen[0]]*probabilities[0, others[1], chosen[1]]
                    outcome = min(values[0, vessel, candidate],
                        *(values[0, other, index] for other, index in zip(others, chosen)))
                    expected[0, vessel, candidate] += mass*outcome
        torch.testing.assert_close(independent_fleet_minimum_cost(values, probabilities), expected)
        permutation = torch.tensor([2, 0, 1])
        torch.testing.assert_close(independent_fleet_minimum_cost(values[:, permutation],
            probabilities[:, permutation]), expected[:, permutation])

    def test_measured_response_supervision_keeps_identity_and_missingness(self):
        import numpy as np
        from local_intervention_value import learning_records, measured_response_records
        context = np.arange(21, dtype=np.float32)/10.
        root = dict(context=context, options=[(0, 0, 0), (1, 0, 1)], outcomes={
            (0, 0): dict(success=0., following=context+.1, response_steps=20, response_sampling='physical_tick'),
            (1, 0): dict(success=1., following=context+.2, response_steps=7, response_sampling='physical_tick'),
            (0, 1): dict(success=1.), (1, 1): dict(success=0.)})
        rows = learning_records(dict(roots=[root]))[0]
        indices, targets = measured_response_records([root], rows)
        np.testing.assert_array_equal(indices, [0, 2])
        np.testing.assert_allclose(targets[:, :-1], np.stack((context+.1-context, context+.2-context)))
        np.testing.assert_allclose(targets[:, -1], [1., .35])
        root['outcomes'][1, 0]['following'] = np.full(21, np.nan, np.float32)
        with self.assertRaisesRegex(ValueError, 'valid aligned'):
            measured_response_records([root], rows)

    def test_response_auxiliary_training_does_not_require_future_state_at_inference(self):
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import fit_joint_potential, LocalDifferenceDecision
        context = np.zeros(21, np.float32)
        context[[4, 12]] = 1.
        controls = np.tile(np.array([.8, .3, -.4, .6, .1, -.2], np.float32), (2, 1))
        root = dict(context=context, effect_controls=controls, futures=[1, 2],
            options=[(0, 0, 0), (1, 0, 1), (10, 3, 2)],
            outcomes={(q, f): dict(success=float(q == f), following=context+.01*q,
                response_steps=20-f, response_sampling='physical_tick') for q in range(3) for f in range(2)})
        data = dict(model={}, roots=[root], predecessor_sha256='response-test', simulation_steps=40)
        with patch('local_intervention_value.torch.save'):
            model = fit_joint_potential(data, 'unused-response-test.pth', updates=2, ensemble_size=1,
                width=4, device='cpu', response_supervision_weight=4.)
        self.assertEqual(model['successor_task_fit']['measured_response_labels'], 6)
        self.assertFalse(model['successor_task_fit']['response_auxiliary_heads_at_inference'])
        recipe = dict(model['successor_task_value']); recipe.pop('kind')
        learner = LocalDifferenceDecision(**recipe)
        self.assertTrue(torch.isfinite(learner.values(torch.from_numpy(context[None]),
                                                     torch.from_numpy(controls[None]))).all())

    def test_physical_contrasts_cancel_only_common_future_motion(self):
        import numpy as np
        from local_intervention_value import learning_records, measured_response_records, measured_response_contrasts
        context = np.zeros(21, np.float32)
        root = dict(context=context, options=[(0, 0, 0), (1, 0, 5)], outcomes={
            (q, f): dict(success=float(q == f), following=context+10*f+.2*q,
                response_steps=20-3*q, response_sampling='physical_tick') for q in range(2) for f in range(2)})
        rows, _, anchors, _ = learning_records(dict(roots=[root]))
        ids, targets = measured_response_records([root], rows)
        pairs, differences = measured_response_contrasts(rows, anchors, ids, targets)
        np.testing.assert_array_equal(pairs, [[0, 2], [1, 3]])
        np.testing.assert_allclose(differences[:, :-1], .2, atol=1e-6)
        np.testing.assert_allclose(differences[:, -1], -.15, atol=1e-6)
        remaining, _ = measured_response_contrasts(rows, anchors, ids[[0, 2, 3]], targets[[0, 2, 3]])
        np.testing.assert_array_equal(remaining, [[0, 2]])
        with self.assertRaisesRegex(ValueError, 'common future'):
            measured_response_contrasts(rows, [[0, 3]], ids, targets)
        with self.assertRaisesRegex(ValueError, 'paired measured'):
            measured_response_contrasts(rows, anchors, ids[:2], targets[:2])

    def test_factual_response_targets_align_physical_vessels_and_observed_clocks(self):
        import copy
        import pickle
        import numpy as np
        from local_intervention_value import measured_factual_response_records
        from conditional_fusion_value import append_vessel_identity
        order = np.array([[1, 0], [0, 1], [1, 0]])
        nodes = np.arange(16, dtype=np.float32).reshape(2, 8)
        contexts = np.stack([np.concatenate(((nodes+index*.2)[permutation].flatten(),
            np.full(5, .05*index, np.float32))) for index, permutation in enumerate(order)])
        record = dict(contexts=contexts, controls=np.zeros((3, 2, 6), np.float32),
            orders=order, remaining=np.array([45., 25., 20.]))
        bank = dict(records=[record, dict(record, contexts=contexts[:2], controls=record['controls'][:2],
            orders=order[:2], remaining=np.array([25., 5.]))])
        original = pickle.dumps(bank)
        indices, targets = measured_factual_response_records(bank)
        np.testing.assert_array_equal(indices, [0, 3])
        expected = np.r_[np.full(16, .2), np.full(5, .05), 1.]
        np.testing.assert_allclose(targets, np.stack((expected, expected)), atol=1e-6)
        identified = copy.deepcopy(bank)
        for item in identified['records']:
            item['contexts'] = append_vessel_identity(torch.from_numpy(item['contexts']),
                torch.from_numpy(item['orders'])).numpy()
        identity_ids, identity_targets = measured_factual_response_records(identified)
        np.testing.assert_array_equal(indices, identity_ids)
        np.testing.assert_array_equal(targets, identity_targets)
        self.assertEqual(original, pickle.dumps(bank))
        identified['records'][0]['contexts'][0, 16] = 0.
        with self.assertRaisesRegex(ValueError, 'identities'):
            measured_factual_response_records(identified)
        with self.assertRaisesRegex(ValueError, 'exactly 20'):
            measured_factual_response_records(dict(records=[dict(record, remaining=np.array([45., 26., 20.]))]))
        with self.assertRaisesRegex(ValueError, 'ordered states'):
            measured_factual_response_records(dict(records=[dict(record, orders=np.zeros_like(order))]))

    def test_factual_response_learning_matches_data_and_excludes_held_out_trajectories(self):
        import copy
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import fit_joint_potential, LocalDifferenceDecision, PlainJointReturnDecision
        controls = np.tile(np.array([.8, .3, -.4, .6, .1, -.2], np.float32), (2, 1))
        roots, records = [], []
        for index in range(6):
            context = np.full(21, .02*index, np.float32); context[[4, 12]] = 1.
            scene = (800+index, 2, 2.)
            roots.append(dict(context=context, effect_controls=controls, futures=[1, 2], scene=scene,
                options=[(0, 0, 0), (1, 0, 5)], outcomes={(q, f): dict(success=float((index+q+f) % 3 != 0),
                    duration=.2*(40+index+q+f), following=context+.1*f+.03*q,
                    response_steps=20, response_sampling='physical_tick') for q in range(2) for f in range(2)}))
            records.append(dict(scene=scene, contexts=np.stack((context, context+.05)),
                controls=np.stack((controls, controls)), orders=np.array([[0, 1], [0, 1]]),
                remaining=np.array([40., 20.]), values=np.full(2, float(index % 2))))
        data = dict(model={}, roots=roots, predecessor_sha256='trajectory-response', simulation_steps=60)
        bank = dict(records=records, predecessor_sha256='trajectory-response', trajectories=6, simulation_steps=240)
        shared = dict(updates=6, ensemble_size=3, width=4, device='cpu', root_sampling_fraction=.5,
            finite_motion_basis=True, response_supervision_weight=4., response_supervision_architecture='linear',
            response_conditioned_value=True, response_value_weight=1., value_target='discounted_return',
            factual_response_supervision=True, physics_residual_response=True)
        conditional = dict(conditional_strata=True, factual_crossfit=True, paired_response_encoding=True,
            aligned_response_contrast=True, paired_event_difference=True, response_contrast_weight=1.)
        with patch('local_intervention_value.torch.save'):
            paired = fit_joint_potential(data, 'unused-trajectory-response.pth', factual_bank=bank,
                **shared, **conditional)
            ordinary = fit_joint_potential(data, 'unused-trajectory-response.pth', factual_bank=bank,
                **shared, ordinary=True, ordinary_architecture='interaction')
        for key in ('training_observation_sha256', 'training_target_sha256', 'response_label_sha256',
                    'factual_response_label_sha256', 'measured_factual_response_labels', 'updates'):
            self.assertEqual(paired['successor_task_fit'][key], ordinary['successor_task_fit'][key])
        self.assertEqual(paired['successor_task_fit']['measured_factual_response_labels'], 6)
        for model, cls in ((paired, LocalDifferenceDecision), (ordinary, PlainJointReturnDecision)):
            recipe = dict(model['successor_task_value']); recipe.pop('kind')
            learner = cls(**recipe)
            self.assertEqual(model['successor_task_fit']['training_parameters_per_member'],
                sum(parameter.numel() for parameter in learner.models[0].parameters()))
            self.assertTrue(torch.isfinite(learner.models[0].forward_catalog(
                torch.from_numpy(context[None]), torch.from_numpy(controls[None]),
                learner.programs, learner.options[:, 1:])).all())
        changed = copy.deepcopy(bank)
        for index, fold in enumerate(paired['successor_task_fit']['factual_crossfit_trajectory_folds']):
            if fold == 0: changed['records'][index]['contexts'][1, [0, 1, 8, 9]] += .2
        with patch('local_intervention_value.torch.save'):
            refit = fit_joint_potential(data, 'unused-trajectory-response.pth', factual_bank=changed,
                **shared, **conditional)
        self.assertNotEqual(paired['successor_task_fit']['factual_response_label_sha256'],
            refit['successor_task_fit']['factual_response_label_sha256'])
        for name, weight in paired['successor_task_value']['states'][0].items():
            torch.testing.assert_close(weight, refit['successor_task_value']['states'][0][name], rtol=0., atol=0.)
        with self.assertRaisesRegex(ValueError, 'measured trajectories'):
            fit_joint_potential(data, 'unused-trajectory-response.pth', **shared, **conditional)
        scalar_models = []
        for is_ordinary, cls in ((False, LocalDifferenceDecision), (True, PlainJointReturnDecision)):
            with patch('local_intervention_value.torch.save'):
                scalar = fit_joint_potential(data, 'unused-shared-successor.pth', factual_bank=bank,
                    **shared, ordinary=is_ordinary, ordinary_architecture='interaction',
                    shared_successor_value=True, shared_factual_response=True)
            scalar_models.append(scalar)
            recipe = dict(scalar['successor_task_value']); recipe.pop('kind')
            learner = cls(**recipe)
            self.assertTrue(all(net.shared_successor_value for net in learner.models))
            self.assertTrue(torch.isfinite(learner.models[0].forward_catalog(
                torch.from_numpy(context[None]), torch.from_numpy(controls[None]),
                learner.programs, learner.options[:, 1:])).all())
            self.assertEqual(scalar['successor_task_fit']['response_value_difference_weight'],
                0. if is_ordinary else 2.)
            self.assertFalse(scalar['success_objective'])
        for key in ('training_observation_sha256', 'training_target_sha256', 'response_label_sha256',
                    'factual_response_label_sha256', 'training_parameters_per_member', 'updates'):
            self.assertEqual(scalar_models[0]['successor_task_fit'][key], scalar_models[1]['successor_task_fit'][key])

    def test_public_response_prior_reconstructs_the_successor_world_frame(self):
        from local_intervention_value import public_physical_response_prior
        dtype = torch.float64
        bearing = torch.tensor([.6, .8], dtype=dtype)
        inward = -bearing
        radius = 60.
        relative = torch.tensor([[65., 3.], [70., -4.], [75., 8.]], dtype=dtype)
        velocity = torch.tensor([[.8, -.1], [-.3, .2], [.2, .5]], dtype=dtype)
        yaw = torch.tensor([0., .4, -.6], dtype=dtype)
        angular = torch.tensor([.02, -.01, .03], dtype=dtype)
        target_velocity = torch.tensor([1.5, .4], dtype=dtype)

        def world(vector):
            return torch.stack((inward[0]*vector[..., 0]-inward[1]*vector[..., 1],
                                inward[1]*vector[..., 0]+inward[0]*vector[..., 1]), -1)

        target = radius*bearing
        position_world = target+world(relative)
        before = torch.cat((relative/30., velocity/3., torch.stack((yaw.cos(), yaw.sin()), -1),
                            angular[:, None], position_world.norm(dim=-1, keepdim=True)/30.), -1)
        global_before = torch.cat((bearing, target_velocity/5., torch.tensor([.1], dtype=dtype)))
        context = torch.cat((before.flatten(), global_before))[None]
        following_yaw = yaw+4.*angular
        states = torch.cat((relative+4.*velocity, following_yaw[:, None], velocity, angular[:, None]), -1)
        paths = states[None, :, None, None].expand(1, 3, 10, 3, 6)
        node, global_response = public_physical_response_prior(context, paths)
        target_after = target+4.*world(target_velocity)
        new_inward = -target_after/target_after.norm()

        def observed(vector):
            return torch.stack(((vector*new_inward).sum(-1),
                                new_inward[0]*vector[..., 1]-new_inward[1]*vector[..., 0]), -1)

        position_after = position_world+4.*world(velocity)
        following = torch.cat((observed(position_after-target_after)/30., observed(world(velocity))/3.,
            observed(world(torch.stack((following_yaw.cos(), following_yaw.sin()), -1))),
            angular[:, None], position_after.norm(dim=-1, keepdim=True)/30.), -1)
        expected_global = torch.cat((target_after/target_after.norm(), observed(world(target_velocity))/5.,
                                     torch.tensor([.15], dtype=dtype)))
        torch.testing.assert_close(node[0, :, 0], following-before, rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(global_response[0, :5], expected_global-global_before, rtol=1e-10, atol=1e-10)
        self.assertEqual(float(global_response[0, -1]), 1.)
        self.assertTrue(torch.equal(node, node[:, :, :1].expand_as(node)))

    def test_predicted_consequences_preserve_catalog_clocks_and_vessel_permutations(self):
        from local_intervention_value import JointTaskPotential
        torch.manual_seed(199)
        context = .05*torch.randn(2, 29)
        controls = .2*torch.randn(2, 3, 6)
        programs = torch.tensor([[0, 0, 0], [9, 0, 0], [9, 0, 0], [0, 7, 0], [0, 7, 0], [1, 0, 0]])
        timing = torch.tensor([[0, 0], [0, 20], [0, 40], [95, 40], [120, 5], [5, 4]])
        permutation = torch.tensor([2, 0, 1])
        reordered = torch.cat((context[:, :24].reshape(2, 3, 8)[:, permutation].flatten(1), context[:, -5:]), -1)
        for dual in (False, True):
            for paired, physics, shared, scalar in ((False, False, False, False), (True, False, False, False),
                    (False, True, False, False), (True, True, False, False),
                    (False, True, True, False), (True, True, True, False), (False, True, True, True)):
                net = JointTaskPotential(width=8, output_width=3 if paired else 1, context_conditioned=True,
                    motion_basis=True, finite_motion_basis=True, dual_motion_basis=dual,
                    paired_response_encoding=paired, aligned_response_contrast=paired,
                    paired_event_difference=paired, response_conditioned_value=True, physics_residual_response=physics,
                    shared_factual_response=shared, shared_successor_value=scalar)
                actual = net.forward_catalog(context, controls, programs, timing)
                expected = torch.stack([net(context, controls, p[None].expand(2, -1), t[None].expand(2, -1))
                    for p, t in zip(programs, timing)], 1)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(actual, net.forward_catalog(reordered, controls[:, permutation],
                    programs[:, permutation], timing), rtol=2e-5, atol=2e-6)
                if paired:
                    torch.testing.assert_close(actual[..., 0], actual[:, :1, 0].expand(-1, len(programs)))
                    self.assertLess(float(actual[:, 0, 1:].detach().sigmoid().max()), 2e-7)
                actual.sum().backward()
                for head in (net.response_node, net.response_global):
                    self.assertTrue(torch.isfinite(head.weight.grad).all())
                    self.assertGreater(float(head.weight.grad.abs().sum()), 0.)
                responses = []
                for q in (1, 2):
                    program = programs[q:q+1].expand(2, -1); clock = timing[q:q+1].expand(2, -1)
                    encoded, representation = net.encode(context, controls, program, clock)
                    prior = net.physical_prior(context, controls, program, clock)
                    response = net.response_prediction(encoded, representation, program, clock, prior,
                        net.factual_response_residual(context, controls) if shared else None)
                    if physics:
                        for actual_prior, expected_prior in zip(response, prior):
                            torch.testing.assert_close(actual_prior, expected_prior, rtol=0., atol=0.)
                    responses.append(response)
                for left, right in zip(*responses):
                    torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)

    def test_shared_successor_value_uses_the_same_physical_pool_for_facts_and_interventions(self):
        from local_intervention_value import JointTaskPotential
        torch.manual_seed(214)
        net = JointTaskPotential(width=8, context_conditioned=True, motion_basis=True,
            finite_motion_basis=True, response_conditioned_value=True, physics_residual_response=True,
            shared_factual_response=True, shared_successor_value=True)
        context = .05*torch.randn(2, 29)
        controls = .2*torch.randn(2, 3, 6)
        observed = .1*torch.randn(2, 30)
        clock = torch.zeros(2, 2, dtype=torch.long)
        factual = torch.zeros(2, 3, dtype=torch.long)
        program = torch.tensor([[9, 0, 0], [0, 7, 0]])
        before = net.observed_response_logits(context, controls, factual, clock, observed)
        after = net.observed_response_logits(context, controls, program, clock, observed)
        torch.testing.assert_close(before, after, rtol=0., atol=0.)
        changed = observed.clone(); changed[:, [0, 8, 16]] += .3
        moved = net.observed_response_logits(context, controls, program, clock, changed)
        self.assertFalse(torch.allclose(before, moved))
        (moved.sigmoid()-before.sigmoid()).sum().backward()
        self.assertGreater(float(net.head[0].weight.grad.abs().sum()), 0.)
        with self.assertRaisesRegex(ValueError, 'scalar potential'):
            JointTaskPotential(shared_successor_value=True)

    def test_shared_factual_motion_preserves_factual_value_and_cancels_common_prediction_errors(self):
        from local_intervention_value import JointTaskPotential, response_basis
        torch.manual_seed(213)
        context = .05*torch.randn(2, 29)
        controls = .2*torch.randn(2, 3, 6)
        programs = torch.tensor([[9, 0, 0], [0, 7, 0]])
        timing = torch.tensor([[0, 20], [5, 40]])
        zero = torch.zeros_like(programs); zero_clock = torch.zeros_like(timing)
        basis = response_basis(context, controls, timing)
        for paired in (False, True):
            net = JointTaskPotential(width=8, output_width=3 if paired else 1, context_conditioned=True,
                motion_basis=True, finite_motion_basis=True, paired_response_encoding=paired,
                aligned_response_contrast=paired, paired_event_difference=paired,
                response_conditioned_value=True, physics_residual_response=True)
            with torch.no_grad():
                for head in (net.response_node, net.response_global):
                    head.weight.normal_(0., .1); head.bias.normal_(0., .1)
            parameter_count = sum(parameter.numel() for parameter in net.parameters())
            factual_value = net(context, controls, zero, zero_clock, basis)
            net.shared_factual_response = True
            torch.testing.assert_close(net(context, controls, zero, zero_clock, basis), factual_value)
            residual = net.factual_response_residual(context, controls, basis)
            predicted = []; priors = []
            for choices, clock in ((zero, zero_clock), (programs, timing)):
                encoded, representation = net.encode(context, controls, choices, clock, basis)
                prior = net.physical_prior(context, controls, choices, clock, basis)
                prediction = net.response_prediction(encoded, representation, choices, clock, prior, residual)
                predicted.append(torch.cat((prediction[0].flatten(1), prediction[1]), -1))
                priors.append(torch.cat((prior[0].flatten(1), prior[1]), -1))
            torch.testing.assert_close(predicted[1]-predicted[0], priors[1]-priors[0])
            expected = net.observed_response_logits(context, controls, programs, timing,
                predicted[1], predicted[0])
            torch.testing.assert_close(net(context, controls, programs, timing, basis), expected)
            self.assertEqual(sum(parameter.numel() for parameter in net.parameters()), parameter_count)
        with self.assertRaisesRegex(ValueError, 'public physical response prior'):
            JointTaskPotential(shared_factual_response=True)

    def test_measured_consequence_teacher_shares_value_head_without_future_decoder_gradients(self):
        from local_intervention_value import JointTaskPotential, response_basis
        torch.manual_seed(208)
        context = .05*torch.randn(2, 29)
        controls = .2*torch.randn(2, 3, 6)
        programs = torch.tensor([[0, 7, 0], [9, 0, 0]])
        timing = torch.tensor([[5, 10], [0, 20]])
        basis = response_basis(context, controls, timing)
        for paired in (False, True):
            net = JointTaskPotential(width=8, output_width=3 if paired else 1, context_conditioned=True,
                motion_basis=True, finite_motion_basis=True, paired_response_encoding=paired,
                aligned_response_contrast=paired, paired_event_difference=paired,
                response_conditioned_value=True, physics_residual_response=True)
            targets = []
            with torch.no_grad():
                for choices, clock in ((programs, timing), (torch.zeros_like(programs), torch.zeros_like(timing))):
                    encoded, representation = net.encode(context, controls, choices, clock, basis)
                    node, global_response = net.response_prediction(encoded, representation, choices, clock,
                        net.physical_prior(context, controls, choices, clock, basis))
                    targets.append(torch.cat((node.flatten(1), global_response), -1))
            expected = net(context, controls, programs, timing, basis)
            teacher = net.observed_response_logits(context, controls, programs, timing, targets[0],
                factual=targets[1])
            torch.testing.assert_close(teacher, expected, rtol=0., atol=0.)
            teacher.sum().backward()
            self.assertIsNone(net.response_node.weight.grad)
            self.assertIsNone(net.response_global.weight.grad)
            self.assertIsNone(net.node[0].weight.grad)
            self.assertGreater(float(net.future_node[0].weight.grad.abs().sum()), 0.)
            self.assertGreater(float(net.head[-1].weight.grad.abs().sum()), 0.)
            if paired:
                self.assertGreater(float(net.paired_head[-1].weight.grad.abs().sum()), 0.)
                with self.assertRaisesRegex(ValueError, 'same-future factual'):
                    net.observed_response_logits(context, controls, programs, timing, targets[0])
            with self.assertRaisesRegex(ValueError, 'aligned physical labels'):
                net.observed_response_logits(context, controls, programs, timing, targets[0][:, :-1])

    def test_held_out_physical_effects_do_not_change_training_scales_or_weights(self):
        import copy
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import fit_joint_potential
        controls = np.tile(np.array([.8, .3, -.4, .6, .1, -.2], np.float32), (2, 1))
        roots = []
        for index in range(6):
            context = np.full(21, .02*index, np.float32); context[[4, 12]] = 1.
            roots.append(dict(context=context, effect_controls=controls, futures=[1, 2], scene=(100+index, 2, 2.),
                options=[(0, 0, 0), (1, 0, 5)], outcomes={(q, f): dict(success=float((index+q+f) % 3 != 0),
                    duration=.2*(40+index+q+f), following=context+.1*f+.03*q,
                    response_steps=20, response_sampling='physical_tick')
                    for q in range(2) for f in range(2)}))
        data = dict(model={}, roots=roots, predecessor_sha256='physical-crossfit', simulation_steps=60)
        settings = dict(updates=6, ensemble_size=3, width=4, device='cpu', conditional_strata=True,
            factual_crossfit=True, root_sampling_fraction=.5, paired_response_encoding=True,
            aligned_response_contrast=True, paired_event_difference=True, finite_motion_basis=True,
            response_supervision_weight=4., response_supervision_architecture='linear',
            response_conditioned_value=True, response_contrast_weight=1., response_value_weight=1.,
            value_target='discounted_return')
        with patch('local_intervention_value.torch.save'):
            model = fit_joint_potential(data, 'unused-physical-crossfit-test.pth', **settings)
        changed = copy.deepcopy(data)
        for index, fold in enumerate(model['successor_task_fit']['factual_crossfit_root_folds']):
            if fold == 0:
                for (q, future), outcome in changed['roots'][index]['outcomes'].items():
                    if q: outcome['following'] += 100.
                    outcome['duration'] += .2*(q+1)
        with patch('local_intervention_value.torch.save'):
            refit = fit_joint_potential(changed, 'unused-physical-crossfit-test.pth', **settings)
        np.testing.assert_array_equal(model['successor_task_fit']['response_contrast_scales'][0],
                                      refit['successor_task_fit']['response_contrast_scales'][0])
        self.assertNotEqual(model['successor_task_fit']['response_contrast_label_sha256'],
                            refit['successor_task_fit']['response_contrast_label_sha256'])
        self.assertNotEqual(model['successor_task_fit']['training_target_sha256'],
                            refit['successor_task_fit']['training_target_sha256'])
        for name, weight in model['successor_task_value']['states'][0].items():
            torch.testing.assert_close(weight, refit['successor_task_value']['states'][0][name], rtol=0., atol=0.)

    def test_response_conditioned_matched_fits_reload_without_future_inputs(self):
        import io
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import fit_joint_potential, LocalDifferenceDecision, PlainJointReturnDecision
        context = np.zeros(21, np.float32); context[[4, 12]] = 1.
        controls = np.tile(np.array([.8, .3, -.4, .6, .1, -.2], np.float32), (2, 1))
        root = dict(context=context, effect_controls=controls, futures=[1, 2],
            options=[(0, 0, 0), (1, 0, 5), (10, 3, 20)],
            outcomes={(q, f): dict(success=float(q == f), following=context+.01*q,
                response_steps=20-f, response_sampling='physical_tick') for q in range(3) for f in range(2)})
        data = dict(model={}, roots=[root], predecessor_sha256='response-value-test', simulation_steps=40)
        models = []
        for ordinary, contrast_weight, physics, value_weight, shared in (
                (False, 0., False, 0., False), (False, 1., False, 0., False), (True, 0., False, 0., False),
                (False, 1., True, 0., False), (True, 0., True, 0., False), (False, 1., True, 1., False),
                (True, 0., True, 1., False), (False, 0., True, 1., True), (True, 0., True, 1., True)):
            with patch('local_intervention_value.torch.save'):
                model = fit_joint_potential(data, 'unused-response-value-test.pth', updates=2, ensemble_size=1,
                    width=4, device='cpu', ordinary=ordinary, ordinary_architecture='interaction',
                    conditional_strata=not ordinary, paired_response_encoding=not ordinary,
                    aligned_response_contrast=not ordinary, paired_event_difference=not ordinary,
                    finite_motion_basis=True, response_supervision_weight=4.,
                    response_supervision_architecture='linear', response_conditioned_value=True,
                    response_contrast_weight=contrast_weight, physics_residual_response=physics,
                    response_value_weight=value_weight, shared_factual_response=shared)
            buffer = io.BytesIO(); torch.save(model, buffer); buffer.seek(0)
            model = torch.load(buffer, weights_only=False)
            recipe = dict(model['successor_task_value']); recipe.pop('kind')
            learner = PlainJointReturnDecision(**recipe) if ordinary else LocalDifferenceDecision(**recipe)
            self.assertTrue(model['successor_task_fit']['response_heads_used_at_inference'])
            self.assertEqual(model['successor_task_fit']['physics_residual_response'], physics)
            self.assertEqual(model['successor_task_fit']['shared_factual_response'], shared)
            self.assertTrue(all(net.shared_factual_response == shared for net in learner.models))
            self.assertEqual(model['successor_task_fit']['measured_response_contrasts'], 4 if contrast_weight else 0)
            self.assertEqual(len(model['successor_task_fit']['response_contrast_scales']), int(bool(contrast_weight)))
            self.assertEqual(model['successor_task_fit']['response_value_weight'], value_weight)
            self.assertEqual(model['successor_task_fit']['measured_response_value_pairs'], 4 if value_weight else 0)
            self.assertEqual(model['successor_task_fit']['training_parameters_per_member'],
                             sum(parameter.numel() for parameter in learner.models[0].parameters()))
            value = learner.models[0].forward_catalog(torch.from_numpy(context[None]),
                torch.from_numpy(controls[None]), learner.programs, learner.options[:, 1:])
            self.assertTrue(torch.isfinite(value).all())
            models.append(model)
        for key in ('training_observation_sha256', 'response_label_sha256', 'updates', 'measured_response_labels'):
            for other in models[1:]:
                self.assertEqual(models[0]['successor_task_fit'][key], other['successor_task_fit'][key])
        with self.assertRaisesRegex(ValueError, 'supervised linear response'):
            fit_joint_potential(data, 'unused-response-value-test.pth', response_conditioned_value=True)
        with self.assertRaisesRegex(ValueError, 'paired response-conditioned'):
            fit_joint_potential(data, 'unused-response-value-test.pth', response_contrast_weight=1.)
        with self.assertRaisesRegex(ValueError, 'physical response prior'):
            fit_joint_potential(data, 'unused-response-value-test.pth', physics_residual_response=True)
        with self.assertRaisesRegex(ValueError, 'Measured consequence values'):
            fit_joint_potential(data, 'unused-response-value-test.pth', response_value_weight=1.)
        with self.assertRaisesRegex(ValueError, 'paired response-conditioned'):
            fit_joint_potential(data, 'unused-response-value-test.pth', ordinary=True,
                finite_motion_basis=True, response_supervision_weight=4., response_supervision_architecture='linear',
                response_conditioned_value=True, response_contrast_weight=1.)

    def test_discounted_task_targets_preserve_measured_outcomes_and_dense_timing(self):
        import numpy as np
        from local_intervention_value import discounted_task_values, measured_root_values, paired_difference_loss
        success = np.array([0., 1., 1., 0.]); ticks = np.array([1, 1, 40, 400])
        values = discounted_task_values(success, ticks)
        expected = []
        for outcome, count in zip(success, ticks):
            rewards = np.full(count, -.001); rewards[-1] += 2*outcome-1
            expected.append((np.power(.995, np.arange(count))@rewards+1.001)/2)
        np.testing.assert_allclose(values, expected, rtol=0., atol=1e-13)
        np.testing.assert_array_equal(values[:2], [0., 1.])
        self.assertLess(values[2], values[1])
        self.assertGreater(values[3], values[0])
        root = dict(root_steps=5, outcomes={(0, 0): dict(success=1., duration=9.),
            (1, 0): dict(success=1., duration=5.)})
        self.assertEqual(measured_root_values(root), {(0, 0): 1., (1, 0): 1.})
        self.assertGreater(measured_root_values(root, 'discounted_return')[1, 0],
                           measured_root_values(root, 'discounted_return')[0, 0])
        logits = torch.tensor([[.2, -.3, .1], [.4, -.1, .5]], requires_grad=True)
        probabilities = logits.softmax(-1); difference = torch.tensor([.2, -.4])
        target = torch.tensor([[.8, .2, 0.], [.6, 0., .4]])
        loss = paired_difference_loss(probabilities, difference, 'discounted_return')
        torch.testing.assert_close(loss, -(target*probabilities.log()).sum(-1))
        loss.sum().backward(); self.assertTrue(torch.isfinite(logits.grad).all())
        binary = torch.tensor([1., -1.])
        torch.testing.assert_close(paired_difference_loss(probabilities, binary),
            paired_difference_loss(probabilities, binary, 'discounted_return'), rtol=0., atol=0.)
        for remaining in (0, 401, 1.2, float('nan')):
            with self.assertRaises(ValueError): discounted_task_values(1., remaining)

    def test_return_reconciliation_uses_paired_duration_and_effective_futures(self):
        import copy
        import numpy as np
        from local_intervention_value import reconcile_root, measured_root_values
        root = dict(context=np.zeros(21, np.float32), futures=[1, 2], root_steps=5,
            options=[(0, 0, 0), (1, 0, 10), (10, 0, 10)],
            outcomes={(q, f): dict(success=1., duration=duration+.2*f)
                for q, duration in enumerate((21., 5., 41.)) for f in range(2)})
        original = copy.deepcopy(root)
        result = reconcile_root(root, [.7, .7, .7], value_target='discounted_return')
        self.assertGreater(result[0][1], result[0][0])
        self.assertLess(result[0][2], result[0][0])
        self.assertAlmostEqual(result[0][0], np.mean([measured_root_values(root, 'discounted_return')[0, f] for f in range(2)]))
        repeated = reconcile_root(dict(root, future_weights=np.array([100., 100.])),
            [.7, .7, .7], value_target='discounted_return')
        np.testing.assert_allclose(result[0], repeated[0], rtol=0., atol=1e-14)
        np.testing.assert_allclose(result[1], repeated[1], rtol=0., atol=1e-14)
        self.assertEqual(result[3]['paired_effective_count'][1], 2.)
        self.assertGreater(result[1][1], 0.)
        self.assertEqual(root['outcomes'], original['outcomes'])

    def test_discounted_response_fits_share_targets_reload_and_recalibrate(self):
        import copy
        import io
        import pickle
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import fit_joint_potential, LocalDifferenceDecision, PlainJointReturnDecision, refresh_local_calibration
        context = np.zeros(21, np.float32); context[[4, 12]] = 1.
        controls = np.tile(np.array([.8, .3, -.4, .6, .1, -.2], np.float32), (2, 1))
        root = dict(context=context, effect_controls=controls, futures=[1, 2], root_steps=5,
            options=[(0, 0, 0), (1, 0, 5), (10, 3, 20)],
            outcomes={(q, f): dict(success=float(q == f), duration=.2*(40+3*q+f), following=context+.01*q,
                response_steps=20-f, response_sampling='physical_tick') for q in range(3) for f in range(2)})
        data = dict(model={}, roots=[root], predecessor_sha256='return-value-test', simulation_steps=40)
        bank = dict(records=[dict(contexts=np.stack((context, context+.01)), controls=np.stack((controls, controls)),
            values=np.array([1., 0.]), remaining=np.array([35., 20.]))], predecessor_sha256='return-value-test',
            trajectories=1, simulation_steps=40)
        before = pickle.dumps((data, bank)); models = []
        for ordinary in (False, True):
            with patch('local_intervention_value.torch.save'):
                model = fit_joint_potential(data, 'unused-return-value-test.pth', factual_bank=bank,
                    updates=2, ensemble_size=1, width=4, device='cpu', ordinary=ordinary,
                    ordinary_architecture='interaction', conditional_strata=not ordinary,
                    paired_response_encoding=not ordinary, aligned_response_contrast=not ordinary,
                    paired_event_difference=not ordinary, finite_motion_basis=True,
                    response_supervision_weight=4., response_supervision_architecture='linear',
                    response_conditioned_value=True, response_contrast_weight=0. if ordinary else 1.,
                    physics_residual_response=True, response_value_weight=1., value_target='discounted_return')
            buffer = io.BytesIO(); torch.save(model, buffer); buffer.seek(0)
            model = torch.load(buffer, weights_only=False)
            recipe = dict(model['successor_task_value']); recipe.pop('kind')
            learner = PlainJointReturnDecision(**recipe) if ordinary else LocalDifferenceDecision(**recipe)
            prediction = learner.models[0].forward_catalog(torch.from_numpy(context[None]),
                torch.from_numpy(controls[None]), learner.programs, learner.options[:, 1:])
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertEqual(model['successor_task_fit']['value_target'], 'discounted_return')
            self.assertFalse(model['success_objective'])
            if not ordinary:
                self.assertFalse(recipe['temporal_value_bounds'])
                recalibrated = refresh_local_calibration(model, data, device='cpu')
                np.testing.assert_allclose(recalibrated['successor_task_value']['corrections'], recipe['corrections'], atol=1e-6)
                refreshed = refresh_local_calibration(model, data, device='cpu', confidence_only=True)
                np.testing.assert_allclose(refreshed['successor_task_value']['uncertainty'], recipe['uncertainty'], atol=1e-6)
            models.append(model)
        self.assertEqual(before, pickle.dumps((data, bank)))
        for key in ('training_observation_sha256', 'training_target_sha256', 'response_label_sha256', 'updates'):
            self.assertEqual(models[0]['successor_task_fit'][key], models[1]['successor_task_fit'][key])
        self.assertNotIn('discounted_return', str(data['roots'][0]['outcomes']))
        with self.assertRaisesRegex(ValueError, 'paired difference learning'):
            fit_joint_potential(data, 'unused-return-value-test.pth', value_target='discounted_return')

    def test_actuator_resolution_retains_source_hull_and_exact_reference(self):
        from conditional_fusion_value import actuator_candidate_grid
        preferred = torch.tensor([[[0., 1.], [.2, .7], [.5, .5]]])
        coarse = actuator_candidate_grid(preferred)
        values = torch.stack((torch.zeros_like(preferred), preferred, torch.ones_like(preferred)), -2)
        digits = torch.arange(9)
        torch.testing.assert_close(coarse,
            torch.stack((values[:, :, digits//3, 0], values[:, :, digits%3, 1]), -1), rtol=0., atol=0.)
        for subdivisions in (1, 2, 3):
            grid = actuator_candidate_grid(preferred, subdivisions)
            levels = 2*subdivisions+1
            self.assertEqual(grid.shape, (1, 3, levels*levels, 2))
            self.assertTrue(bool(((grid >= 0.) & (grid <= 1.)).all()))
            torch.testing.assert_close(grid[:, :, subdivisions*levels+subdivisions], preferred, rtol=0., atol=0.)
            retained = (grid[:, :, :, None] == coarse[:, :, None]).all(-1).any(2)
            self.assertTrue(bool(retained.all()))
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                actuator_candidate_grid(preferred, invalid)
        union = actuator_candidate_grid(preferred, include_uniform_midpoint=True)
        self.assertEqual(union.shape, (1, 3, 16, 2))
        self.assertTrue(bool((union[:, :, :, None] == coarse[:, :, None]).all(-1).any(2).all()))
        fixed = actuator_candidate_grid(torch.full_like(preferred, .5))
        self.assertTrue(bool((union[:, :, :, None] == fixed[:, :, None]).all(-1).any(2).all()))
        torch.testing.assert_close(union[:, :, 5], preferred, rtol=0., atol=0.)
        self.assertTrue(bool(((union >= 0.) & (union <= 1.)).all()))

    def test_dual_motion_preserves_both_physical_response_bases(self):
        from local_intervention_value import response_basis, response_pair_geometry
        torch.manual_seed(83)
        context = .05*torch.randn(2, 21)
        controls = .2*torch.randn(2, 2, 6)
        timing = torch.tensor([[0, 1], [4, 3]])
        finite = response_basis(context, controls, timing)
        persistent = response_basis(context, controls)
        dual = response_basis(context, controls, timing, dual=True)
        for index, width in ((0, 42), (1, 6)):
            torch.testing.assert_close(dual[index][..., :width], finite[index], rtol=0., atol=0.)
            torch.testing.assert_close(dual[index][..., width:], persistent[index], rtol=0., atol=0.)
        geometry = response_pair_geometry(dual[1][:, :1], dual[1][:, 1:], dual=True)
        for index, basis in enumerate((finite, persistent)):
            torch.testing.assert_close(geometry[..., 18*index:18*(index+1)],
                response_pair_geometry(basis[1][:, :1], basis[1][:, 1:]), rtol=0., atol=0.)
        self.assertGreater(float((dual[1][..., :6]-dual[1][..., 6:]).abs().max()), 0.)
        with self.assertRaisesRegex(ValueError, 'requires an intervention window'):
            response_basis(context, controls, dual=True)

    def test_finite_motion_features_respect_delay_release_and_factual_commands(self):
        from local_intervention_value import response_basis
        context = torch.zeros(2, 21)
        context[:, [4, 12]] = 1.
        controls = torch.tensor([1., 1., -1., -1., 0., 0.]).repeat(2, 2, 1)
        persistent = response_basis(context, controls)
        full = response_basis(context, controls, torch.tensor([[0, 20], [0, 200]]))
        for actual, expected in zip(full[:2], persistent):
            torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        for clock in ((0, 0), (20, 10), (120, 40)):
            features, path, _ = response_basis(context, controls, torch.tensor([clock, clock]))
            torch.testing.assert_close(path, persistent[1][:, :, :1].expand_as(path), rtol=0., atol=0.)
            self.assertEqual(float(features.reshape(2, 2, 10, 3, 14)[..., 7:].abs().max()), 0.)
        delayed = response_basis(context, controls, torch.tensor([[10, 1], [10, 20]]))[1]
        torch.testing.assert_close(delayed[..., :2, :],
            persistent[1][:, :, :1, :2].expand_as(delayed[..., :2, :]), rtol=0., atol=0.)
        pulse = response_basis(context, controls, torch.tensor([[0, 1], [0, 1]]))[1]
        self.assertLess(float((pulse[:, :, 9, -1, :2]-pulse[:, :, 0, -1, :2]).norm()),
                        float((persistent[1][:, :, 9, -1, :2]-persistent[1][:, :, 0, -1, :2]).norm()))

    def test_finite_motion_matches_environment_control_intervals(self):
        import numpy as np
        from envs.modules import WAMV
        from local_intervention_value import response_basis
        context = torch.zeros(1, 21)
        rows = context[:, :16].reshape(1, 2, 8)
        rows[0, 1, :2] = torch.tensor([3., -2.])/30.
        rows[..., 2:4] = torch.tensor([.2, -.1])/3.
        rows[..., 4:6] = torch.tensor([np.cos(.3), np.sin(.3)])
        rows[..., 6] = .05
        controls = torch.tensor([1., .25, -.2, .6, .1, -.3]).repeat(1, 2, 1)
        coefficients = torch.tensor([[-1., -1.]]+[[a, z] for a in (0., .5, 1.) for z in (0., .5, 1.)])
        for delay, span in ((0, 1), (2, 3), (15, 8)):
            path = response_basis(context, controls, torch.tensor([[delay, span]]))[1]
            for vessel in range(2):
                source, rule, factual = controls[0, vessel].split(2)
                for digit in (0, 5, 9):
                    boat = WAMV()
                    self.assertAlmostEqual(boat.N*boat.dt, .2)
                    boat.init_velocity_r = np.array([.2, -.1, .05])
                    boat.reset(np.array([0., 0.]) if vessel == 0 else np.array([3., -2.]), .3)
                    command = factual if digit == 0 else coefficients[digit]*source+(1-coefficients[digit])*rule
                    for step in range(20):
                        applied = command if delay <= step < delay+span else factual
                        boat.step((750.*applied+250.).numpy())
                        if step in (4, 9, 19):
                            snapshot = (4, 9, 19).index(step)
                            predicted = path[0, vessel, digit, snapshot]
                            torch.testing.assert_close(predicted[[0, 1, 3, 4, 5]],
                                torch.tensor([boat.x, boat.y, *boat.velocity_r], dtype=predicted.dtype),
                                rtol=2e-5, atol=2e-5)
                            torch.testing.assert_close(torch.stack((predicted[2].cos(), predicted[2].sin())),
                                torch.tensor([np.cos(boat.theta), np.sin(boat.theta)], dtype=predicted.dtype),
                                rtol=2e-5, atol=2e-5)

    def test_paired_response_catalog_preserves_factual_reference_and_zero_exposure(self):
        from local_intervention_value import JointTaskPotential
        torch.manual_seed(165)
        context=.05*torch.randn(2,29)
        controls=.2*torch.randn(2,3,6)
        programs=torch.tensor([[0,0,0],[9,0,1],[9,0,1],[0,7,0],[1,9,0]])
        timing=torch.tensor([[0,0],[0,0],[0,20],[5,4],[12,5]])
        for aligned in (False,True):
            net=JointTaskPotential(width=8,output_width=3,context_conditioned=True,
                motion_basis=True,paired_response_encoding=True,aligned_response_contrast=aligned)
            actual=net.forward_catalog(context,controls,programs,timing)
            expected=torch.stack([net(context,controls,p[None].expand(2,-1),t[None].expand(2,-1))
                for p,t in zip(programs,timing)],1)
            torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
            torch.testing.assert_close(actual[...,0],actual[:,:1,0].expand(-1,len(programs)))
            self.assertTrue(bool((actual[:,:2,1:].sigmoid()<2e-7).all()))
            self.assertTrue(bool((actual[:,2:,1:].sigmoid()>1e-4).all()))
            permutation=torch.tensor([2,0,1])
            permuted=torch.cat((context[:,:24].reshape(2,3,8)[:,permutation].flatten(1),context[:,-5:]),-1)
            reordered=net.forward_catalog(permuted,controls[:,permutation],programs[:,permutation],timing)
            torch.testing.assert_close(actual,reordered,rtol=2e-5,atol=2e-6)
            with torch.enable_grad():
                result=net(context,controls,programs[2:3].expand(2,-1),timing[2:3].expand(2,-1))
                (result[:,1].square()+result[:,2].square()).mean().backward()
            self.assertTrue(bool(torch.isfinite(net.paired_head[2].weight.grad).all()))
            self.assertGreater(float(net.paired_head[2].weight.grad.abs().sum()),0.)

    def test_aligned_response_contrast_retains_factual_subject_and_signed_peer_effect(self):
        from local_intervention_value import JointTaskPotential
        factual=torch.tensor([[[1.,2.],[3.,4.],[5.,6.]]])
        active=torch.tensor([[True,False,False]])
        reference=torch.zeros(1,14)
        unchanged=JointTaskPotential.aligned_response_features(factual,factual,reference,active,torch.zeros(1,8))
        torch.testing.assert_close(unchanged[:,4:6],factual[:,0])
        torch.testing.assert_close(unchanged[:,14:],torch.zeros(1,14))
        following=factual.clone();following[:,1,0]-=2.
        changed=JointTaskPotential.aligned_response_features(following,factual,reference,active,torch.zeros(1,8))
        torch.testing.assert_close(changed[:,:14],unchanged[:,:14])
        self.assertEqual(float(changed[0,16]),-2.)
        torch.testing.assert_close(changed[:,18:20],torch.zeros(1,2))

    def test_finite_catalog_matches_individual_windows_and_boat_permutations(self):
        from local_intervention_value import JointTaskPotential, response_basis
        torch.manual_seed(24)
        context = .05*torch.randn(2, 29)
        controls = .2*torch.randn(2, 3, 6)
        programs = torch.tensor([[0, 0, 0], [9, 0, 1], [9, 0, 1], [0, 7, 0], [1, 9, 0], [0, 4, 9]])
        timing = torch.tensor([[0, 0], [0, 1], [0, 20], [5, 4], [120, 5], [95, 40]])
        permutation = torch.tensor([2, 0, 1])
        permuted_context = torch.cat((context[:, :24].reshape(2, 3, 8)[:, permutation].flatten(1),
                                      context[:, -5:]), -1)
        cases = [(peer, output, residual, dual) for dual in (False, True)
                 for peer, output, residual in ((True, 1, False), (False, 3, False), (True, 1, True))]
        for peer, output, residual, dual in cases:
            net = JointTaskPotential(width=8, motion_basis=True, context_conditioned=True,
                finite_motion_basis=True, peer_conditioned=peer, output_width=output,
                residual_potential=residual, dual_motion_basis=dual)
            actual = net.forward_catalog(context, controls, programs, timing)
            expected = torch.stack([net(context, controls, p[None].expand(2, -1), t[None].expand(2, -1))
                                    for p, t in zip(programs, timing)], 1)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            reordered = net.forward_catalog(permuted_context, controls[:, permutation],
                                             programs[:, permutation], timing)
            torch.testing.assert_close(actual, reordered, rtol=2e-5, atol=2e-6)
            clock = timing[1:2].expand(2, -1)
            program = programs[1:2].expand(2, -1)
            torch.testing.assert_close(net(context, controls, program, clock),
                net(context, controls, program, clock, response_basis(context, controls, clock, dual)))
            actual.sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))
        with self.assertRaises(ValueError):
            JointTaskPotential(finite_motion_basis=True)
        with self.assertRaises(ValueError):
            JointTaskPotential(finite_motion_basis=True, motion_basis=True, local_contrast_head=True)
        with self.assertRaisesRegex(ValueError, 'requires finite motion features'):
            JointTaskPotential(dual_motion_basis=True, motion_basis=True)

    def test_dual_motion_fitting_and_reload_retain_window_features(self):
        import io
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import fit_joint_potential, LocalDifferenceDecision
        context = np.zeros(21, np.float32)
        context[[4, 12]] = 1.
        controls = np.tile(np.array([.8, .3, -.4, .6, .1, -.2], np.float32), (2, 1))
        root = dict(context=context, effect_controls=controls, futures=[1, 2],
            options=[(0, 0, 0), (1, 0, 1), (10, 3, 2)],
            outcomes={(q, f): dict(success=float(q == f)) for q in range(3) for f in range(2)})
        data = dict(model={}, roots=[root], predecessor_sha256='dual-test', simulation_steps=40)
        bank = dict(predecessor_sha256='dual-test', trajectories=1, simulation_steps=10,
            records=[dict(contexts=context[None], controls=controls[None],
                          values=np.array([1.], np.float32), remaining=np.array([10.], np.float32))])
        with patch('local_intervention_value.torch.save'):
            model = fit_joint_potential(data, 'unused-dual-test.pth', updates=2, ensemble_size=1,
                width=4, device='cpu', factual_bank=bank, finite_motion_basis=True, dual_motion_basis=True)
        stream = io.BytesIO()
        torch.save(model, stream); stream.seek(0)
        loaded = torch.load(stream, map_location='cpu', weights_only=False)
        self.assertTrue(loaded['successor_task_fit']['dual_motion_features'])
        self.assertEqual(loaded['successor_task_fit']['supervised_node_labels'], 7)
        recipe = dict(loaded['successor_task_value']); recipe.pop('kind')
        learner = LocalDifferenceDecision(**recipe)
        self.assertTrue(learner.models[0].dual_motion_basis)
        predicted = learner.values(torch.from_numpy(context[None]), torch.from_numpy(controls[None]))
        self.assertEqual(tuple(predicted.shape), (1, 1, 3))
        self.assertTrue(torch.isfinite(predicted).all())

    def test_componentwise_refinement_rejects_compensating_individual_harm(self):
        from conditional_fusion_value import pareto_fleet_refinement
        unary = torch.tensor([[[1., 2.], [3., 0.]]])
        risk = torch.zeros(1, 2, 2, 2, 2)
        reference = torch.zeros(1, 2, dtype=torch.long)
        plans = torch.tensor([[[1, 1], [0, 1]]])
        torch.testing.assert_close(pareto_fleet_refinement(unary, risk, plans[:, :1], reference, 32.),
                                   torch.tensor([[1, 1]]))
        torch.testing.assert_close(pareto_fleet_refinement(
            unary, risk, plans[:, :1], reference, 32., componentwise=True), reference)
        torch.testing.assert_close(pareto_fleet_refinement(
            unary, risk, plans, reference, 32., componentwise=True), torch.tensor([[0, 1]]))
        unary = torch.zeros(1, 3, 2)
        risk = torch.zeros(1, 3, 3, 2, 2)
        risk[:, 0, 1, 0, 0] = 2.
        risk[:, 0, 2, 1, 1] = 1.
        reference = torch.zeros(1, 3, dtype=torch.long)
        plans = torch.ones(1, 1, 3, dtype=torch.long)
        torch.testing.assert_close(pareto_fleet_refinement(unary, risk, plans, reference, 32.), plans[:, 0])
        torch.testing.assert_close(pareto_fleet_refinement(
            unary, risk, plans, reference, 32., componentwise=True), reference)

    def test_factual_risk_confidence_preserves_rescues_and_protects_promising_factuals(self):
        from unittest.mock import patch
        from local_intervention_value import JointTaskPotential, LocalDifferenceDecision
        net = JointTaskPotential(width=8)
        settings = dict(width=8, states=[net.state_dict()],
            contexts=torch.zeros(1, 21), controls=torch.zeros(1, 2, 6),
            options=[(0, 0, 0), (1, 0, 1)], corrections=torch.zeros(1, 2),
            uncertainty=torch.zeros(1, 2), observed=torch.ones(1, 2, dtype=torch.bool),
            confidence=1., global_penalty=.9, margin=.02, project_task_value=True,
            factual_bound_pruning=True, candidate_bound_pruning=True, factual_risk_confidence=True)
        decision = LocalDifferenceDecision(**settings)
        estimates = torch.tensor([[[.2, .9], [.8, .99]]])
        def values(*args, factual_only=False, candidate_indices=None, **kwargs):
            if factual_only:
                return estimates[:, :, :1]
            return estimates if candidate_indices is None else estimates[:, :, candidate_indices]
        packet = dict(node=torch.zeros(2, 2, 16), _local_difference_controls=torch.zeros(2, 2, 6))
        action = torch.zeros(2, 2, 3)
        with patch('conditional_fusion_value.intervention_task_context',
                   return_value=(torch.zeros(2, 21), torch.tensor([[0, 1], [0, 1]]))), \
             patch.object(decision, 'values', side_effect=values), \
             patch.object(decision, 'context_relation', return_value=(torch.full((2, 1), 100.), None)):
            pruned = decision(packet, action)
            self.assertEqual(pruned[:, 0, -1].tolist(), [1., -1.])
            decision.candidate_bound_pruning = False
            torch.testing.assert_close(decision(packet, action), pruned)
            decision.factual_risk_confidence = False
            torch.testing.assert_close(decision(packet, action), action)
        with self.assertRaises(ValueError):
            LocalDifferenceDecision(**settings, ordinary=True)
        with self.assertRaises(ValueError):
            LocalDifferenceDecision(**settings, outcome_scaled_confidence=True)

    def test_feedback_guard_checks_every_horizon_and_restores_boat_order(self):
        from unittest.mock import patch
        from conditional_fusion_value import feedback_guarded_refinement
        source = torch.zeros(4, 2, 2)
        candidates = torch.tensor([[[[0., 0.], [1., 1.]], [[.2, .2], [.8, .8]]]]).repeat(4, 1, 1, 1)
        order = torch.tensor([[1, 0]]).repeat(4, 1)
        reference = torch.zeros(4, 2, dtype=torch.long)
        proposed = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]])
        packet = dict(node=torch.zeros(4, 2, 16), marker='unchanged')
        calls = []
        def forecast(actor, subset, values, physical, both, synchronized, horizon):
            calls.append(horizon)
            self.assertEqual(subset['node'].shape[0], 3)
            self.assertEqual(subset['marker'], 'unchanged')
            self.assertEqual(both, 'both')
            self.assertTrue(synchronized)
            torch.testing.assert_close(physical[0, 1], torch.tensor([[.2, .2], [1., 1.]]))
            risk = torch.tensor([[1., .9], [1., .9], [1., .9]])
            clearance = torch.tensor([[5., 6.], [5., 6.], [5., 6.]])
            if horizon == 1.6:
                risk[0, 1] = 1.1
            if horizon == 3.2:
                clearance[1, 1] = 4.
            return risk, clearance
        with patch('conditional_fusion_value.coupled_actuator_risk', side_effect=forecast):
            actual = feedback_guarded_refinement(
                object(), packet, source, candidates, order, reference, proposed, (.8, 1.6, 3.2))
        self.assertEqual(calls, [.8, 1.6, 3.2])
        torch.testing.assert_close(actual, torch.tensor([[0, 0], [0, 0], [1, 1], [0, 0]]))
        with patch('conditional_fusion_value.coupled_actuator_risk') as forecast_mock:
            feedback_guarded_refinement(object(), packet, source, candidates, order, reference, reference, (.8,))
            forecast_mock.assert_not_called()

    def test_feedback_task_guard_rejects_lost_capture_or_asset_progress(self):
        from unittest.mock import patch
        from conditional_fusion_value import feedback_guarded_refinement
        source = torch.zeros(3, 2, 2)
        candidates = torch.tensor([[[[0., 0.], [1., 1.]], [[0., 0.], [1., 1.]]]]).repeat(3, 1, 1, 1)
        reference = torch.zeros(3, 2, dtype=torch.long)
        proposed = torch.ones_like(reference)
        order = torch.tensor([[0, 1]]).repeat(3, 1)
        def forecast(actor, packet, values, physical, mode, **settings):
            self.assertEqual(mode, 'task')
            self.assertEqual(settings['tail_horizon'], 1.6)
            risk = torch.tensor([[1., .9]]).repeat(3, 1)
            clearance = torch.tensor([[5., 6.]]).repeat(3, 1)
            progress = torch.tensor([[[10., -1.], [11., -2.]],
                                     [[10., -1.], [9., -.5]],
                                     [[10., -1.], [9., -2.]]])
            return risk, clearance, progress
        with patch('conditional_fusion_value.coupled_actuator_risk', side_effect=forecast):
            result = feedback_guarded_refinement(object(), {}, source, candidates, order,
                reference, proposed, (.8, 1.6, 3.2), tail_horizon=1.6, task_guard=True)
        torch.testing.assert_close(result, torch.tensor([[0, 0], [0, 0], [1, 1]]))
        with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
            feedback_guarded_refinement(object(), {}, source, candidates, order,
                reference, proposed, (.8,), tail_horizon=float('nan'))

    def test_coupled_task_guard_tracks_public_target_without_changing_risk(self):
        from types import SimpleNamespace
        from conditional_fusion_value import coupled_actuator_risk
        packet = dict(motion=torch.zeros(1, 2, 6), water=torch.zeros(1, 2, 3),
                      node=torch.zeros(1, 2, 16), task=torch.zeros(1, 8))
        packet['motion'][0, 1, 1] = 20.
        packet['node'][..., 11:13] = -1./3.
        packet['node'][0, 0, 2] = 8./30.
        packet['node'][0, 0, 4] = 1.
        actor = SimpleNamespace(prediction_step=.05, risk_horizon=.2, risk_feedback=False,
            target_motion=None, interaction_enabled=True,
            distance_risk=lambda distance: torch.exp(-distance),
            public_target_motion=lambda p: (30.*p['node'][:, 0, 2:4], 5.*p['node'][:, 0, 4:6]))
        source = torch.full((1, 2, 2), -1./3.)
        plans = torch.stack((torch.zeros(1, 2, 2), torch.ones(1, 2, 2)), 1)
        baseline = coupled_actuator_risk(actor, packet, source, plans, 'both', synchronized=True)
        actual = coupled_actuator_risk(actor, packet, source, plans, 'task', synchronized=True)
        for value, expected in zip(actual[:2], baseline):
            torch.testing.assert_close(value, expected, rtol=0., atol=0.)
        torch.testing.assert_close(actual[2], torch.tensor([[[9., -8.25], [9., -8.25]]]),
                                   rtol=0., atol=1e-5)
        packet['task'].fill_(1000.)
        unchanged = coupled_actuator_risk(actor, packet, source, plans, 'task', synchronized=True)
        torch.testing.assert_close(unchanged[2], actual[2], rtol=0., atol=0.)

    def test_outcome_scaled_confidence_keeps_pruning_consistent_with_decisions(self):
        from unittest.mock import patch
        from local_intervention_value import JointTaskPotential, LocalDifferenceDecision
        net = JointTaskPotential(width=8)
        decision = LocalDifferenceDecision(width=8, states=[net.state_dict()],
            contexts=torch.zeros(1, 21), controls=torch.zeros(1, 2, 6),
            options=[(0, 0, 0), (1, 0, 1)], corrections=torch.zeros(1, 2),
            uncertainty=torch.zeros(1, 2), observed=torch.ones(1, 2, dtype=torch.bool),
            confidence=1., global_penalty=.2, margin=.02, project_task_value=True,
            factual_bound_pruning=True, candidate_bound_pruning=True, outcome_scaled_confidence=True)
        estimates = torch.tensor([[[.8, .99], [.5, .69]]])
        def values(*args, factual_only=False, candidate_indices=None, **kwargs):
            if factual_only:
                return estimates[:, :, :1]
            return estimates if candidate_indices is None else estimates[:, :, candidate_indices]
        packet = dict(node=torch.zeros(2, 2, 16), _local_difference_controls=torch.zeros(2, 2, 6))
        action = torch.zeros(2, 2, 3)
        with patch('conditional_fusion_value.intervention_task_context',
                   return_value=(torch.zeros(2, 21), torch.tensor([[0, 1], [0, 1]]))), \
             patch.object(decision, 'values', side_effect=values), \
             patch.object(decision, 'context_relation', return_value=(torch.full((2, 1), 100.), None)):
            pruned = decision(packet, action)
            self.assertEqual(pruned[:, 0, -1].tolist(), [1., -1.])
            decision.candidate_bound_pruning = False
            torch.testing.assert_close(decision(packet, action), pruned)
            decision.outcome_scaled_confidence = False
            torch.testing.assert_close(decision(packet, action), action)

    def test_pareto_refinement_rejects_risk_deviation_tradeoffs_and_ties(self):
        from conditional_fusion_value import pareto_fleet_refinement
        unary = torch.tensor([[[2., 1., 3.], [2., 1., 3.]]]).repeat(2, 1, 1)
        risk = torch.zeros(2, 2, 2, 3, 3)
        risk[:, 0, 1] = torch.tensor([[2., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
        reference = torch.tensor([[0, 0], [1, 1]])
        plans = torch.tensor([[[2, 2], [1, 1], [0, 0]],
                              [[2, 2], [1, 1], [0, 0]]])
        # (2, 2) has the lowest weighted score but increases control deviation.
        result = pareto_fleet_refinement(unary, risk, plans, reference, 32.)
        self.assertEqual(result.tolist(), [[1, 1], [1, 1]])

    def setUp(self):
        self.parts = torch.tensor([[0, 0], [1, 0], [2, 0], [1, 2]])
        self.options = torch.tensor([[[[0., 0.], [0., 0.]],
            [[.5, .5], [0., 0.]], [[0., 0.], [.5, -.5]], [[.5, .5], [.5, -.5]]]])

    def test_pair_block_escapes_a_strict_single_vessel_local_minimum(self):
        from conditional_fusion_value import refine_pair_blocks
        unary = torch.zeros((1, 2, 2))
        risk = torch.zeros((1, 2, 2, 2, 2))
        risk[0, 0, 1] = torch.tensor([[2., 5.], [5., 0.]])
        risk[0, 1, 0] = risk[0, 0, 1].T
        plans = torch.tensor([[[0, 0], [1, 1]]])
        self.assertEqual(risk[0, 0, 1, :, 0].argmin().item(), 0)
        self.assertEqual(risk[0, 1, 0, :, 0].argmin().item(), 0)
        refined = refine_pair_blocks(unary, risk, plans, 1., sweeps=1)
        torch.testing.assert_close(refined, torch.ones_like(plans))
        torch.testing.assert_close(plans, torch.tensor([[[0, 0], [1, 1]]]))
        torch.testing.assert_close(
            refine_pair_blocks(unary, torch.zeros_like(risk), plans, 1., sweeps=2), plans)

    def test_joint_feedback_shortlist_preserves_distinct_coordination_plans(self):
        from conditional_fusion_value import distinct_fleet_shortlist
        plans = torch.tensor([[[1, 1], [1, 1], [0, 1], [0, 0]],
                              [[0, 0], [0, 0], [0, 0], [0, 0]]])
        scores = torch.tensor([[.1, .2, .4, .3], [.1, .2, .3, .4]])
        selected = distinct_fleet_shortlist(plans, scores, 3)
        torch.testing.assert_close(selected, torch.tensor(
            [[[1, 1], [0, 0], [0, 1]], [[0, 0], [0, 0], [0, 0]]]))

    def test_plain_return_ignores_development_future_weights(self):
        import copy
        import numpy as np
        from unittest.mock import patch
        from plain_return import fit_plain_return
        # No paired factual label exists for the changed action's future.
        root = dict(context=np.arange(21, dtype=np.float32)/20.,
            effect_controls=np.zeros((2, 6), np.float32), futures=[1, 2],
            options=[(0, 0, 0), (1, 0, 10)], future_weights=np.ones(2, np.float32),
            outcomes={(0, 0): dict(success=0.), (1, 1): dict(success=1.)})
        data = dict(model={}, roots=[root], predecessor_sha256='plain-test',
                    simulation_steps=20)
        bank = dict(predecessor_sha256='plain-test', simulation_steps=10, records=[dict(
            contexts=np.zeros((2, 21), np.float32), controls=np.zeros((2, 2, 6), np.float32),
            values=np.array([0., 1.], np.float32), remaining=np.array([10., 20.], np.float32))])
        weighted = copy.deepcopy(data)
        weighted['roots'][0]['future_weights'] = np.array([1., 128.], np.float32)
        with patch('plain_return.torch.save'):
            first = fit_plain_return(data, 'unused-plain-test.pth', factual_bank=bank,
                updates=3, parameter_budget=1000, device='cpu')
            second = fit_plain_return(weighted, 'unused-plain-test.pth', factual_bank=bank,
                updates=3, parameter_budget=1000, device='cpu')
            resumed = fit_plain_return(data, 'unused-plain-test.pth', factual_bank=bank,
                updates=1, parameter_budget=1000, device='cpu', initial_model=first,
                learning_rate=0., additional_candidate_options=[(10, 0, 10)],
                expected_observation_sha256=first['successor_task_fit']['training_observation_sha256'])
        for key, value in first['successor_task_value']['state'].items():
            torch.testing.assert_close(value, second['successor_task_value']['state'][key], atol=0, rtol=0)
            torch.testing.assert_close(value, resumed['successor_task_value']['state'][key], atol=0, rtol=0)
        fit = first['successor_task_fit']
        self.assertEqual(fit['supervised_node_labels'], 4)
        self.assertEqual(fit['development_future_weight'], 1.)
        self.assertFalse(fit['separate_factual_objective'])
        self.assertEqual(fit['paired_local_observations'], 0)
        self.assertEqual(resumed['successor_task_fit']['initialization_updates'], 3)
        self.assertEqual(resumed['successor_task_fit']['total_parameter_updates'], 4)
        self.assertEqual(resumed['successor_task_fit']['additional_unobserved_candidates'], 1)
        self.assertIn((10, 0, 10), resumed['successor_task_value']['options'])
        self.assertEqual(resumed['successor_task_fit']['paired_local_observations'], 0)

    def test_paired_replay_keeps_distinct_behavior_states_at_the_same_clock(self):
        import copy
        import numpy as np
        from local_intervention_value import merge_paired_data
        root = dict(scene=(123, 2, 2.), prefix_future=17, root_steps=5,
            context=np.zeros(21, np.float32), effect_controls=np.zeros((2, 6), np.float32),
            options=[(0, 0, 0)], futures=[29], outcomes={(0, 0): dict(success=0.)})
        first = dict(model={}, roots=[root], predecessor_sha256='same-tail', completed=True)
        second = copy.deepcopy(first)
        second['roots'][0]['context'][0] = .5
        second['roots'][0]['outcomes'][0, 0]['success'] = 1.
        third = copy.deepcopy(first)
        third['roots'][0]['effect_controls'][0, 0] = .5
        third['roots'][0]['outcomes'][0, 0]['success'] = 1.
        with self.assertRaisesRegex(ValueError, 'changed physical context'):
            merge_paired_data([first, second])
        merged = merge_paired_data([first, second, third, first, second],
                                  allow_distinct_contexts=True)
        self.assertEqual(len(merged['roots']), 3)
        self.assertEqual([r['outcomes'][0, 0]['success'] for r in merged['roots']], [0., 1., 1.])
        self.assertTrue(all(len(r['outcomes']) == 1 for r in merged['roots']))

    def test_plain_return_can_share_the_conditional_execution_budget(self):
        from plain_return import PlainReturnNetwork, PlainReturnDecision
        from local_intervention_value import LocalDifferenceDecision
        from types import SimpleNamespace
        network = PlainReturnNetwork(2, 8)
        plain = PlainReturnDecision(2, 8, network.state_dict(), [(0, 0, 0), (1, 0, 10)],
                                    intervention_budget=1)
        packet = dict(node=torch.zeros((4, 2, 16)),
                      _conditional_count=torch.tensor([[0.], [1.], [0.], [2.]]))
        packet['node'][:, :, 15] = torch.tensor([5., 10., 7., 15.])[:, None]/400.
        conditional = SimpleNamespace(decision_mode='rolling', decision_period=5,
                                      intervention_budget=1, ordinary=False)
        self.assertEqual(plain.eligible(packet).tolist(), [True, False, False, False])
        torch.testing.assert_close(plain.eligible(packet), LocalDifferenceDecision.eligible(conditional, packet))
        plain.intervention_budget = 0
        self.assertEqual(plain.eligible(packet).tolist(), [True, True, False, True])

    def test_plain_return_fixed_margin_preserves_network_and_default_argmax(self):
        from unittest.mock import patch
        from plain_return import PlainReturnNetwork, PlainReturnDecision
        network = PlainReturnNetwork(2, 8)
        options = [(0, 0, 0), (1, 0, 10)]
        plain = PlainReturnDecision(2, 8, network.state_dict(), options, intervention_budget=1)
        conservative = PlainReturnDecision(2, 8, network.state_dict(), options,
                                          intervention_budget=1, gain_margin=.08)
        for key, value in plain.network.state_dict().items():
            torch.testing.assert_close(value, conservative.network.state_dict()[key], rtol=0., atol=0.)
        packet = dict(node=torch.zeros((2, 2, 16)), _plain_return_controls=torch.zeros((2, 2, 6)),
                      _conditional_count=torch.zeros((2, 1)))
        action = torch.rand((2, 2, 3))
        logits = torch.tensor([[.9, .95], [.5, .9]]).logit()
        with patch('conditional_fusion_value.intervention_task_context',
                   return_value=(torch.zeros((2, 21)), torch.tensor([[0, 1], [0, 1]]))), \
             patch.object(plain.network, 'forward_catalog', return_value=logits), \
             patch.object(conservative.network, 'forward_catalog', return_value=logits):
            plain.record_decisions = conservative.record_decisions = True
            original = plain(packet, action)
            selected = conservative(packet, action)
            torch.testing.assert_close(selected[..., :3], action)
            self.assertEqual(plain.last_decision['accepted'].tolist(), [True, True])
            self.assertEqual(conservative.last_decision['accepted'].tolist(), [False, True])
            self.assertEqual(selected[:, 0, -1].tolist(), [-1., 1.])
            torch.testing.assert_close(selected[1], original[1], rtol=0., atol=0.)
            torch.testing.assert_close(conservative.last_decision['candidate_gain'], torch.tensor([.05, .4]))
            packet['_conditional_count'].fill_(1)
            torch.testing.assert_close(conservative(packet, action), action, rtol=0., atol=0.)

    def test_decision_measurement_preserves_commands_and_clears_stale_values(self):
        from unittest.mock import patch
        from plain_return import PlainReturnNetwork, PlainReturnDecision
        from local_intervention_value import JointTaskPotential, LocalDifferenceDecision
        torch.manual_seed(43)
        options = [(0, 0, 0), (1, 0, 10)]
        context = torch.zeros(1, 21); controls = torch.zeros(1, 2, 6)
        potential = JointTaskPotential(width=4)
        conditional = LocalDifferenceDecision(width=4, states=[potential.state_dict()],
            contexts=context, controls=controls, options=options, corrections=torch.zeros(1, 2),
            uncertainty=torch.zeros(1, 2), observed=torch.ones(1, 2, dtype=torch.bool))
        network = PlainReturnNetwork(2, 8)
        ordinary = PlainReturnDecision(2, 8, network.state_dict(), options)
        packet = dict(node=torch.zeros(1, 2, 16), _local_difference_controls=controls,
                      _plain_return_controls=controls)
        action = torch.zeros(1, 2, 3)
        with patch('conditional_fusion_value.intervention_task_context',
                   return_value=(context, torch.tensor([[0, 1]]))):
            for learner in (conditional, ordinary):
                original = learner(packet, action)
                learner.record_decisions = True
                measured = learner(packet, action)
                torch.testing.assert_close(original, measured, rtol=0., atol=0.)
                self.assertEqual(learner.last_decision['option'].shape, (1, 3))
                self.assertTrue(torch.isfinite(learner.last_decision['predicted_gain']).all())
                learner.record_catalog = True
                learner(packet, action)
                self.assertEqual(learner.last_catalog['predicted_gain'].shape, (1, 2))
                torch.testing.assert_close(learner.last_catalog['predicted_gain'][:, 0], torch.zeros(1))
                option_index = int((learner.options == learner.last_decision['option'][0]).all(-1).nonzero()[0])
                torch.testing.assert_close(learner.last_catalog['predicted_gain'][:, option_index],
                    learner.last_decision['candidate_gain'])
                later = dict(packet, node=packet['node'].clone())
                later['node'][:, :, 15] = 1./400.
                torch.testing.assert_close(learner(later, action), action, rtol=0., atol=0.)
                self.assertIsNone(learner.last_decision)
                self.assertIsNone(learner.last_catalog)
            conditional.record_catalog = False
            conditional.project_task_value = True
            conditional.factual_bound_pruning = True
            conditional.factual_values = torch.ones(1)
            conditional(packet, action)
            self.assertEqual(conditional.last_decision['reason'], 'factual_upper_bound')
            conditional.record_catalog = True
            conditional(packet, action)
            self.assertEqual(conditional.last_catalog['predicted_gain'].shape, (1, 2))
            self.assertTrue(torch.isfinite(conditional.last_catalog['predicted_gain']).all())

    def test_catalog_intervention_execution_uses_physical_identity_and_timing(self):
        import numpy as np
        from mechanism_controls import catalog_intervention_action
        factual = np.zeros((2, 3), np.float32); order = np.array([1, 0])
        action = catalog_intervention_action(factual, (9, 3, 10), order)
        self.assertEqual(action.shape, (2, 14))
        np.testing.assert_array_equal(action[:, 4], [13, 13])
        np.testing.assert_array_equal(action[:, 8:11], [[3, -1, -1], [3, -1, -1]])
        np.testing.assert_array_equal(action[:, 11:13], [[-1, -1], [1, 1]])
        np.testing.assert_array_equal(action[:, 13], [0, 0])
        immediate = catalog_intervention_action(factual, (9, 0, 10), order)
        np.testing.assert_array_equal(immediate[:, 9:11], immediate[:, 11:13])
        np.testing.assert_array_equal(catalog_intervention_action(factual, (0, 0, 0), order), factual)
        with self.assertRaisesRegex(ValueError, 'actual vessel permutation'):
            catalog_intervention_action(factual, (9, 0, 10), [0, 0])

    def test_measured_catalog_acquisition_keeps_unselected_and_rejected_branches_distinct(self):
        import copy
        import numpy as np
        from local_intervention_value import measured_choice_options
        record = dict(decisions={
            'conditional': dict(accepted=[False], option=[[9, 3, 10]]),
            'ordinary': dict(accepted=[True], option=[[9, 3, 10]])},
            ranking=dict(labels=['factual', 'catalog:1', 'catalog:2'],
                options=np.array([[0, 0, 0], [9, 3, 10], [1, 0, 20]])),
            outcomes={label: [] for label in ['factual', 'conditional', 'ordinary', 'catalog:1', 'catalog:2']})
        options = measured_choice_options(record)
        self.assertEqual(options['conditional'], (0, 0, 0))
        self.assertEqual(options['ordinary'], options['catalog:1'])
        self.assertEqual(options['catalog:2'], (1, 0, 20))
        missing = copy.deepcopy(record); missing['outcomes']['unidentified'] = []
        with self.assertRaisesRegex(ValueError, 'lacks its executed physical program'):
            measured_choice_options(missing)
        duplicate = copy.deepcopy(record); duplicate['ranking']['labels'][2] = 'catalog:1'
        with self.assertRaisesRegex(ValueError, 'uniquely identify'):
            measured_choice_options(duplicate)
        conflict = copy.deepcopy(record); conflict['ranking']['options'][0] = [1, 0, 20]
        with self.assertRaisesRegex(ValueError, 'two different physical programs'):
            measured_choice_options(conflict)

    def test_paired_replay_aligns_future_weights_and_retains_measured_responses(self):
        import copy
        import numpy as np
        from local_intervention_value import merge_paired_data
        root = dict(scene=(123, 2, 2.), prefix_future=17, root_steps=5,
            context=np.zeros(21, np.float32), effect_controls=np.zeros((2, 6), np.float32),
            options=[(0, 0, 0)], futures=[29], future_weights=np.array([3.], np.float32),
            outcomes={(0, 0): dict(success=0., following=np.ones(21, np.float32),
                                  response_steps=20, response_sampling='physical_tick')})
        first = dict(model={}, roots=[root], predecessor_sha256='same-tail', completed=True)
        second = copy.deepcopy(first)
        second['roots'][0].update(futures=[31, 29], future_weights=np.array([2., 3.], np.float32),
            outcomes={(0, 0): dict(success=1.), (0, 1): dict(success=0.)})
        merged = merge_paired_data([first, second])['roots'][0]
        self.assertEqual(merged['futures'], [29, 31])
        np.testing.assert_array_equal(merged['future_weights'], [3., 2.])
        np.testing.assert_array_equal(merged['outcomes'][0, 0]['following'], np.ones(21))
        self.assertEqual(merged['outcomes'][0, 1]['success'], 1.)
        second['roots'][0]['future_weights'][1] = 4.
        with self.assertRaisesRegex(ValueError, 'conflicting population weights'):
            merge_paired_data([first, second])

    def test_ordinary_return_rejects_conditional_parameter_initialization(self):
        from local_intervention_value import fit_joint_potential
        from plain_return import fit_plain_return
        with self.assertRaisesRegex(ValueError, 'cannot inherit conditional parameters'):
            fit_joint_potential({}, 'unused-plain-test.pth', ordinary=True, initial_states=[{}])
        with self.assertRaisesRegex(ValueError, 'cannot inherit learned decision modules'):
            fit_plain_return(dict(model={}), 'unused-plain-test.pth',
                initial_model=dict(successor_task_value=dict(kind='local_difference_joint')))

    def test_fresh_calibration_preserves_parameters_and_does_not_invent_observations(self):
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import JointTaskPotential, fit_joint_potential, refresh_local_calibration
        net = JointTaskPotential(width=4)
        prior = {key: value.clone() for key, value in net.state_dict().items()}
        root = dict(context=np.zeros(21, np.float32),
            effect_controls=np.zeros((2, 6), np.float32), futures=[1, 2],
            options=[(0, 0, 0), (1, 0, 10)],
            outcomes={(0, 0): dict(success=0.), (0, 1): dict(success=0.),
                      (1, 0): dict(success=1.), (1, 1): dict(success=1.)})
        data = dict(model={}, roots=[root], predecessor_sha256='current-controller',
                    simulation_steps=40)
        with patch('local_intervention_value.torch.save'):
            model = fit_joint_potential(data, 'unused-calibration.pth', updates=0,
                ensemble_size=1, width=4, initial_states=[prior], device='cpu',
                context_conditioned=False, motion_basis=False,
                additional_candidate_options=[(10, 0, 10)], factual_difference_transport=True)
        value = model['successor_task_value']
        for key, tensor in prior.items():
            torch.testing.assert_close(tensor, value['states'][0][key], atol=0, rtol=0)
        extra = value['options'].index((10, 0, 10))
        self.assertFalse(value['observed'][0, extra])
        self.assertTrue(value['observed'][0, value['options'].index((1, 0, 10))])
        self.assertTrue(value['factual_difference_transport'])
        self.assertEqual(model['successor_task_fit']['updates'], 0)
        self.assertEqual(model['successor_task_fit']['supervised_node_labels'], 4)
        self.assertEqual(model['successor_task_fit']['last_loss'], [None])
        refreshed = refresh_local_calibration(model, data, device='cpu',
                                             effective_future_confidence=True, confidence_only=True)
        np.testing.assert_array_equal(refreshed['successor_task_value']['member_corrections'],
                                      value['member_corrections'])
        np.testing.assert_allclose(refreshed['successor_task_value']['uncertainty'],
                                   value['uncertainty'], rtol=1e-6, atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'requires a fitted parameter prior'):
            fit_joint_potential(data, 'unused-calibration.pth', updates=0)

    def test_confidence_counts_futures_instead_of_rescaled_objective_mass(self):
        import numpy as np
        from local_intervention_value import reconcile_root
        root = dict(context=np.zeros(21, np.float32), futures=[1, 2],
            options=[(0, 0, 0), (1, 0, 10)],
            outcomes={(0, 0): dict(success=0.), (0, 1): dict(success=0.),
                      (1, 0): dict(success=1.), (1, 1): dict(success=1.)})
        reference = reconcile_root(root, [.2, .7], effective_future_confidence=True)
        weighted = dict(root, future_weights=np.array([100., 100.]))
        legacy = reconcile_root(weighted, [.2, .7])
        corrected = reconcile_root(weighted, [.2, .7], effective_future_confidence=True)
        np.testing.assert_array_equal(legacy[0], corrected[0])
        self.assertEqual(corrected[3]['confidence_failure_count'][1], 2.)
        self.assertEqual(reference[3]['repair_lower'][1], corrected[3]['repair_lower'][1])
        self.assertGreater(corrected[1][1], legacy[1][1])
        concentrated = reconcile_root(dict(root, future_weights=np.array([100., 1.])),
                                     [.2, .7], effective_future_confidence=True)
        self.assertAlmostEqual(concentrated[3]['confidence_failure_count'][1], 101.**2/10001.)
        self.assertLess(concentrated[3]['repair_lower'][1], corrected[3]['repair_lower'][1])

    def test_conditional_gains_obey_factual_outcome_boundaries(self):
        from types import SimpleNamespace
        from local_intervention_value import LocalDifferenceDecision
        estimator = SimpleNamespace(stratum_value_bounds=True, stratum_localized_bounds=False)
        gains = torch.tensor([[[.8, -.8], [.8, -.8]]])
        baseline = torch.tensor([[[0.], [1.]]])
        locality = torch.ones((2, 2, 2))
        residual = torch.zeros((2, 1, 2, 2))
        result = LocalDifferenceDecision.combine_stratum_gains(
            estimator, gains, baseline, locality, residual)
        self.assertTrue((result[:, 0] >= 0).all())
        self.assertTrue((result[:, 1] <= 0).all())
        self.assertTrue(((baseline+result >= 0) & (baseline+result <= 1)).all())

    def test_conditional_bounds_preserve_unsupported_neural_gains(self):
        from types import SimpleNamespace
        from local_intervention_value import LocalDifferenceDecision
        estimator = SimpleNamespace(stratum_value_bounds=True, stratum_localized_bounds=True)
        gains = torch.tensor([[[.4, -.3]]])
        baseline = torch.tensor([[[.6]]])
        residual = torch.tensor([[[[.8, -.9]]], [[[.2, -.8]]]])
        combine = LocalDifferenceDecision.combine_stratum_gains
        torch.testing.assert_close(
            combine(estimator, gains, baseline, torch.zeros((2, 1, 2)), residual), gains)
        full = combine(estimator, gains, baseline, torch.ones((2, 1, 2)), residual)
        self.assertTrue(((baseline+full >= 0) & (baseline+full <= 1)).all())
        half = combine(estimator, gains, baseline, torch.full((2, 1, 2), .5), residual)
        torch.testing.assert_close(half, (gains+full)/2)

    def test_factual_calibration_can_preserve_the_learned_paired_effect(self):
        from types import SimpleNamespace
        from local_intervention_value import LocalDifferenceDecision
        class Potential:
            motion_basis = False
            def forward_catalog(self, context, controls, programs, timing, **kwargs):
                return torch.tensor([[.2, .4, .1]]).logit().expand(len(context), -1)
        estimator = SimpleNamespace(models=[Potential()], local_contrast_head=False,
            programs=torch.tensor([[0], [1], [2]]), options=torch.tensor([[0, 0, 0], [1, 0, 10], [2, 0, 10]]),
            ordinary=False, factual_values=torch.tensor([.6]), bandwidth=.1, interpolation_power=2.,
            conditional_strata=False, factual_difference_transport=True,
            continuation_statistics=lambda context, controls: None)
        context=torch.zeros((1, 13)); controls=torch.zeros((1, 1, 6))
        relation=(torch.zeros((1, 1)), None)
        values = LocalDifferenceDecision.values(estimator, context, controls, relation=relation)
        torch.testing.assert_close(values, torch.tensor([[[.6, .8, .5]]]))
        estimator.factual_values = torch.tensor([.9])
        bounded = LocalDifferenceDecision.values(estimator, context, controls, relation=relation)
        torch.testing.assert_close(bounded, torch.tensor([[[.9, 1., .8]]]))
        estimator.factual_difference_transport = False
        original = LocalDifferenceDecision.values(estimator, context, controls, relation=relation)
        torch.testing.assert_close(original, torch.tensor([[[.9, .4, .1]]]))
        # A changed predecessor must not be anchored to the old factual label.
        estimator.continuation_factual_only = True
        estimator.factual_difference_transport = True
        continuation = (torch.tensor([.3]), torch.ones(1), None, None, torch.ones(1))
        transferred = LocalDifferenceDecision.values(estimator, context, controls,
            continuation=continuation, relation=relation)
        torch.testing.assert_close(transferred, torch.tensor([[[.3, .5, .2]]]))

    def test_ordinary_return_executes_its_argmax_without_a_stage_or_support_lookup(self):
        from unittest.mock import patch
        from conditional_fusion_value import OrdinaryJointReturnDecision
        options=torch.tensor([[-1.,-1.]]+[[a,b] for a in (0.,.5,1.) for b in (0.,.5,1.)])
        decision=OrdinaryJointReturnDecision(dict(width=8,paired=False,node_width=8,
            global_width=6,program_options=options),[0,7],[8,10],6)
        class Returns(torch.nn.Module):
            def forward(self,context,codes):
                return torch.where(context[:,0]<0.,(codes==7).float(),(codes==0).float())
        value=Returns();value.program_options=options;decision.program_model=value
        context=torch.full((2,53),1000.);context[0,0]=-1000.
        order=torch.tensor([[3,1,0,2,4,5],[5,4,3,2,1,0]])
        action=torch.rand(2,6,3)
        with patch('conditional_fusion_value.intervention_task_context',return_value=(context,order)):
            selected=decision({},action)
        torch.testing.assert_close(selected[...,:3],action)
        self.assertEqual(selected[:,0,4].tolist(),[10.,8.])
        torch.testing.assert_close(selected[0,3,9:11],options[7])
        self.assertTrue((selected[0,[0,1,2,4,5],9:11]==-1.).all())
        self.assertTrue((selected[1,:,9:13]==-1.).all())
        self.assertFalse(any(hasattr(decision,key) for key in
            ('prior','contexts','candidate_contexts','candidate_support','explicit_wait','time_bandwidth')))

    def test_ordinary_return_observations_need_no_pair_labels_or_successor_states(self):
        import numpy as np
        from unittest.mock import patch
        from conditional_fusion_value import ordinary_task_return_examples
        options=np.array([[-1.,-1.]]+[[a,b] for a in (0.,.5,1.) for b in (0.,.5,1.)])
        factual=np.zeros((6,13));factual[:,4]=10;factual[:,9:13]=-1.
        changed=factual.copy();changed[0,9:11]=options[7];changed[0,11:13]=options[7]
        branch=dict(packet={},left=factual,right=changed,before={'success':0},after={'success':1},
            continuation_seed=9,edges='unused',candidate_support='unused')
        data=dict(completed=True,span=10,program_options=options,
            rollouts=[dict(branches=[branch,branch])])
        with patch('conditional_fusion_value.intervention_task_context',
                   return_value=(torch.zeros(1,53),torch.arange(6)[None])):
            examples,actions,_,fleet,fit=ordinary_task_return_examples([data])
        self.assertEqual(actions,[(0,10),(7,10)])
        self.assertEqual(fleet,6);self.assertEqual(fit['unique_outcomes'],2)
        self.assertEqual(examples[0]['targets'].tolist(),[0.,1.])
        self.assertNotIn('edges',examples[0])

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


class PhysicalVesselIdentityTests(unittest.TestCase):
    def test_proposal_roots_use_both_native_decisions_and_preserve_factual_actions(self):
        import copy
        import numpy as np
        from types import SimpleNamespace
        from unittest.mock import patch
        from local_intervention_value import (JointTaskPotential, PlainJointReturnDecision,
            LocalDifferenceDecision, FactualProposalRootSelector)
        class Actor:
            successor_task_value = None
            def __call__(self, packet, **kwargs):
                self.last_kwargs = kwargs
                return action.clone(), None
        net = JointTaskPotential(4, context_conditioned=True, motion_basis=False)
        context = torch.zeros(3, 21); controls = torch.zeros(3, 2, 6)
        order = torch.tensor([[1, 0]]).expand(3, -1); action = torch.zeros(3, 2, 3)
        common = dict(width=4, states=[net.state_dict()], options=[(0, 0, 0), (1, 0, 1)],
            motion_basis=False, intervention_budget=1, decision_period=5, isolate_factual_start=True,
            context_conditioned=True)
        ordinary = PlainJointReturnDecision(**common, defenders=2, gain_margin=.02)
        paired = LocalDifferenceDecision(**common, contexts=context[:1], controls=controls[:1],
            corrections=np.zeros((1, 2), np.float32), uncertainty=np.zeros((1, 2), np.float32),
            observed=np.zeros((1, 2), bool), margin=.02, confidence=0., global_penalty=0.)
        selector = FactualProposalRootSelector([paired, ordinary], ['a'*64, 'b'*64])
        actor = Actor(); packet = dict(node=torch.zeros(3, 2, 16))
        original = copy.deepcopy(net.state_dict())
        pvalues = torch.tensor([[[.5, .5], [.5, .55], [.5, .51]]])
        ovalues = torch.tensor([[.5, .55], [.5, .5], [.5, .51]])
        with patch('conditional_fusion_value.intervention_task_context', return_value=(context, order)), \
                patch('conditional_fusion_value.paired_effect_controls', return_value=controls), \
                patch.object(paired, 'values', return_value=pvalues), \
                patch.object(ordinary.models[0], 'forward_catalog', return_value=ovalues.logit()):
            self.assertEqual(selector(packet, actor).tolist(), [True, True, False])
            self.assertEqual(selector.last_observations['options'].tolist(),
                [[[0, 0, 0], [1, 0, 1]], [[1, 0, 1], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
            np.testing.assert_array_equal(selector.last_observations['order'], order.numpy())
            changed = dict(packet, unobserved_terminal_outcome=torch.tensor([999., -999., 999.]))
            self.assertEqual(selector(changed, actor).tolist(), [True, True, False])
            for index, expected in [(0, [False, True, False]), (1, [True, False, False])]:
                own = FactualProposalRootSelector([paired, ordinary], ['a'*64, 'b'*64],
                    trigger_estimators=[index])
                self.assertEqual(own(packet, actor).tolist(), expected)
                self.assertEqual(own(changed, actor).tolist(), expected)
                self.assertEqual(own.specification['trigger_estimators'], [index])
                np.testing.assert_array_equal(own.last_observations['options'],
                    selector.last_observations['options'])
        self.assertEqual(actor.last_kwargs, dict(deterministic=True, gate_only=True))
        torch.testing.assert_close(action, torch.zeros_like(action), rtol=0., atol=0.)
        for learner in (paired, ordinary):
            for key, value in learner.models[0].state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0., atol=0.)
        self.assertEqual(selector.specification['proposal_model_sha256'], ['a'*64, 'b'*64])
        with self.assertRaisesRegex(ValueError, 'unchanged factual'):
            selector(packet, SimpleNamespace(successor_task_value=ordinary))
        with self.assertRaisesRegex(ValueError, 'actual frozen estimator hashes'):
            FactualProposalRootSelector([ordinary], [])
        for invalid in ([], [2], [0, 0], [True]):
            with self.assertRaisesRegex(ValueError, 'nonempty subset'):
                FactualProposalRootSelector([paired, ordinary], ['a'*64, 'b'*64],
                    trigger_estimators=invalid)

    def test_directed_root_resume_requires_the_same_proposal_estimators(self):
        import copy
        import hashlib
        import tempfile
        import numpy as np
        from pathlib import Path
        from unittest.mock import patch
        import local_intervention_value as local
        net = local.JointTaskPotential(4, context_conditioned=True, motion_basis=False)
        learner = local.PlainJointReturnDecision(width=4, states=[net.state_dict()], defenders=2,
            options=[(0, 0, 0), (1, 0, 1)], motion_basis=False, intervention_budget=1,
            decision_period=5, isolate_factual_start=True, context_conditioned=True)
        selector = local.FactualProposalRootSelector([learner, learner], ['a'*64, 'b'*64])
        with tempfile.TemporaryDirectory() as directory:
            policy = Path(directory)/'predecessor.pth'; policy.write_bytes(b'factual-source')
            output = Path(directory)/'states.pth'
            record = dict(scene=(71, 2, 1.), future=19, physical_steps=12,
                contexts=np.zeros((3, 21), np.float32), controls=np.zeros((3, 2, 6), np.float32),
                orders=np.tile(np.arange(2), (3, 1)), values=np.ones(3), remaining=np.array([12, 7, 2]),
                root_snapshot=dict(predecessor_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(),
                    root_steps=5, selection_reason='first_accepted_proposal'))
            settings = dict(root_sample_seed=13, inference_batch=1, inference_device='cpu',
                root_selector=selector)
            with patch.object(local, 'factual_trajectory_batch', return_value=[record]) as measured:
                saved = local.collect_onpolicy_factual_roots(policy, [(record['scene'], record['future'])],
                    output, **settings)
                self.assertIs(measured.call_args.kwargs['root_selector'], selector)
                self.assertTrue(saved['completed'])
                self.assertEqual(saved['root_selection'], selector.specification)
                self.assertEqual(saved['simulation_steps'], 12)
                local.collect_onpolicy_factual_roots(policy, [(record['scene'], record['future'])], output, **settings)
                self.assertEqual(measured.call_count, 1)
                old_digest = hashlib.sha256(output.read_bytes()).hexdigest()
                changed = copy.copy(selector); changed.specification = dict(selector.specification,
                    proposal_model_sha256=['b'*64])
                settings['root_selector'] = changed
                with self.assertRaisesRegex(ValueError, 'different predecessor, tasks, sampler'):
                    local.collect_onpolicy_factual_roots(policy, [(record['scene'], record['future'])],
                        output, **settings)
                self.assertEqual(measured.call_count, 1)
                self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), old_digest)
                settings['root_selector'] = local.FactualProposalRootSelector(
                    [learner, learner], ['a'*64, 'b'*64], trigger_estimators=[0])
                with self.assertRaisesRegex(ValueError, 'different predecessor, tasks, sampler'):
                    local.collect_onpolicy_factual_roots(policy, [(record['scene'], record['future'])],
                        output, **settings)
                self.assertEqual(measured.call_count, 1)
                self.assertEqual(hashlib.sha256(output.read_bytes()).hexdigest(), old_digest)

    def test_factual_wait_discounts_elapsed_ticks_and_respects_terminal_boundaries(self):
        from local_intervention_value import factual_wait_targets
        immediate = torch.tensor([.08, .08, .08])
        wait = torch.tensor([0., .2, .9])
        elapsed = torch.tensor([5., 10., 1.])
        live = torch.tensor([True, True, False])
        expected = factual_wait_targets(immediate, wait, elapsed, live)
        torch.testing.assert_close(expected, torch.tensor([.08*.995**5, .2*.995**10, 0.]))
        self.assertLess(float(expected[0]), float(immediate[0]))
        with self.assertRaises(ValueError):
            factual_wait_targets(immediate, wait, elapsed, live, 1.01)

    def test_waiting_uses_same_current_state_rule_for_both_estimators(self):
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import JointTaskPotential, PlainJointReturnDecision, LocalDifferenceDecision
        class FixedWait(torch.nn.Module):
            def __init__(self, value, observed_returns=False):
                super().__init__(); self.value = value
                self.margin_adjusted = not observed_returns
            def forward(self, context, controls):
                return context.new_full((len(context),), self.value)
        net = JointTaskPotential(4, context_conditioned=True, motion_basis=False)
        context = torch.zeros(1, 21); controls = torch.zeros(1, 2, 6)
        order = torch.tensor([[1, 0]]); action = torch.zeros(1, 2, 3)
        common = dict(width=4, states=[net.state_dict()], options=[(0, 0, 0), (1, 0, 1)],
            motion_basis=False, intervention_budget=1, context_conditioned=True)
        ordinary = PlainJointReturnDecision(**common, defenders=2, gain_margin=.02)
        paired = LocalDifferenceDecision(**common, contexts=context, controls=controls,
            corrections=np.zeros((1, 2), np.float32), uncertainty=np.zeros((1, 2), np.float32),
            observed=np.zeros((1, 2), bool), margin=.02, confidence=0., global_penalty=0.)
        for learner, control_key in [(ordinary, '_ordinary_return_controls'), (paired, '_local_difference_controls')]:
            learner.record_catalog = True
            packet = dict(node=torch.zeros(1, 2, 16), **{control_key: controls})
            values = torch.tensor([[[.5, .55]]])
            prediction = (patch.object(ordinary.models[0], 'forward_catalog', return_value=values[0].logit())
                if learner is ordinary else patch.object(paired, 'values', return_value=values))
            with prediction, patch('conditional_fusion_value.intervention_task_context', return_value=(context, order)):
                for wait, observed, accepted in [(.04, False, False), (.02, False, True),
                                                 (.04, True, True), (.06, True, False)]:
                    learner.wait_value = FixedWait(wait, observed)
                    result = learner(packet, action)
                    direct = learner.forward_public(context, controls, order, action, torch.tensor([True]))
                    torch.testing.assert_close(result, direct, rtol=0., atol=0.)
                    self.assertEqual(result.shape[-1], 14 if accepted else 3)
                    self.assertAlmostEqual(float(learner.last_catalog['wait_value'][0]), wait, places=6)

    def test_factual_wait_fit_preserves_teacher_and_ignores_future_outcome_labels(self):
        import copy
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import (JointTaskPotential, PlainJointReturnDecision,
            factual_opportunity_transitions, fit_factual_wait_opportunity)
        net = JointTaskPotential(4, context_conditioned=True, motion_basis=False)
        recipe = dict(kind='plain_joint_return', width=4, states=[net.state_dict()], defenders=2,
            options=[(0, 0, 0), (1, 0, 1)], motion_basis=False, intervention_budget=1,
            gain_margin=.02, context_conditioned=True)
        model = dict(successor_task_value=recipe)
        x = np.zeros((2, 21), np.float32); x[1, -1] = 5/400
        bank = dict(completed=True, records=[dict(contexts=x.copy(), controls=np.zeros((2, 2, 6), np.float32),
            orders=np.tile(np.arange(2), (2, 1)), values=[0., 1.], remaining=[10, 5]) for _ in range(2)])
        def prediction(self, context, controls, programs, timing, responses=None):
            return torch.tensor([[.5, .6]], device=context.device).expand(len(context), -1).logit()
        settings = dict(recipe); settings.pop('kind')
        learner = PlainJointReturnDecision(**settings)
        with patch.object(JointTaskPotential, 'forward_catalog', prediction):
            original = factual_opportunity_transitions(learner, bank, 3)
            changed = copy.deepcopy(bank)
            for record in changed['records']:
                record['values'] = [999., -999.]; record['remaining'] = [-8, -9]
            altered = factual_opportunity_transitions(learner, changed, 3)
            self.assertEqual(original['following'].tolist(), [1, -1, 3, -1])
            self.assertEqual(original['observation_sha256'], altered['observation_sha256'])
            torch.testing.assert_close(original['opportunity'], altered['opportunity'], rtol=0., atol=0.)
            fitted = fit_factual_wait_opportunity(model, bank, None, updates=3, width=4, ensemble_size=2,
                score_batch_size=3)
        self.assertNotIn('wait_value', model['successor_task_value'])
        for key, value in net.state_dict().items():
            torch.testing.assert_close(value, fitted['successor_task_value']['states'][0][key], rtol=0., atol=0.)
        restored = dict(fitted['successor_task_value']); restored.pop('kind')
        one = PlainJointReturnDecision(**restored); two = PlainJointReturnDecision(**copy.deepcopy(restored))
        context = torch.from_numpy(x); control = torch.zeros(2, 2, 6)
        torch.testing.assert_close(one.wait_value(context, control), two.wait_value(context, control), rtol=0., atol=0.)
        self.assertEqual(fitted['factual_wait_fit']['updates'], 6)
        self.assertFalse(fitted['factual_wait_fit']['future_outcomes_used'])
        malformed = copy.deepcopy(bank); malformed['records'][0]['contexts'][1, -1] = 10/400
        with self.assertRaises(ValueError):
            factual_opportunity_transitions(learner, malformed)

    def test_graph_return_margin_retains_a_small_predicted_gain(self):
        from unittest.mock import patch
        from local_intervention_value import JointTaskPotential, PlainJointReturnDecision
        net = JointTaskPotential(4, context_conditioned=True, motion_basis=False)
        settings = dict(width=4, states=[net.state_dict()], defenders=2,
                        options=[(0, 0, 0), (1, 0, 1)], motion_basis=False)
        packet = dict(node=torch.zeros(1, 2, 16), _ordinary_return_controls=torch.zeros(1, 2, 6))
        action = torch.zeros(1, 2, 3)
        context = torch.zeros(1, 21); order = torch.tensor([[0, 1]])
        logits = torch.logit(torch.tensor([[.5, .55]]))
        for margin, accepted in [(0., True), (.02, True), (.08, False)]:
            model = PlainJointReturnDecision(**settings, gain_margin=margin)
            with patch('conditional_fusion_value.intervention_task_context', return_value=(context, order)), \
                    patch.object(model.models[0], 'forward_catalog', return_value=logits):
                result = model(packet, action)
            if accepted:
                self.assertEqual(result.shape[-1], 14)
                self.assertTrue((result[..., -1] == 1).all())
            else:
                torch.testing.assert_close(result, action, rtol=0., atol=0.)
        with self.assertRaises(ValueError):
            PlainJointReturnDecision(**settings, gain_margin=-.1)
        for budget, count, eligible in [(1, 0, True), (1, 1, False), (2, 1, True), (0, 9, True)]:
            model = PlainJointReturnDecision(**settings, intervention_budget=budget)
            counted = dict(packet, _conditional_count=torch.tensor([[count]]))
            self.assertEqual(bool(model.eligible(counted).item()), eligible)
            if not eligible:
                torch.testing.assert_close(model(counted, action), action, rtol=0., atol=0.)
        with self.assertRaises(ValueError):
            PlainJointReturnDecision(**settings, intervention_budget=-1)

    def test_graph_return_shares_unobserved_catalog_without_inventing_labels(self):
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import fit_joint_potential
        context = np.zeros(21, np.float32); context[[4, 12]] = 1.
        root = dict(context=context, effect_controls=np.zeros((2, 6), np.float32),
                    futures=[1, 2], options=[(0, 0, 0), (1, 0, 1)],
                    outcomes={(q, f): dict(success=float(q == f)) for q in range(2) for f in range(2)})
        data = dict(model={}, roots=[root], predecessor_sha256='graph-return-test', simulation_steps=8)
        settings = dict(updates=2, ensemble_size=1, width=4, device='cpu', motion_basis=False)
        with patch('local_intervention_value.torch.save'):
            reference = fit_joint_potential(data, 'unused-graph-test.pth', **settings)
            ordinary = fit_joint_potential(data, 'unused-graph-test.pth', ordinary=True,
                ordinary_architecture='interaction', additional_candidate_options=[(10, 2, 3)], **settings)
        self.assertIn((10, 2, 3), ordinary['successor_task_value']['options'])
        self.assertEqual(reference['successor_task_fit']['training_observation_sha256'],
                         ordinary['successor_task_fit']['training_observation_sha256'])
        self.assertEqual(ordinary['successor_task_fit']['objective'], 'absolute return')
        self.assertFalse(ordinary['successor_task_fit']['local_reconciliation'])
        self.assertEqual(sum(ordinary['successor_task_fit']['member_updates']), 2)

    def test_randomized_branch_screen_retains_groups_and_changes_queried_boats(self):
        import numpy as np
        from local_intervention_value import candidate_screen_order
        codes = [1, 2, 10, 20, 100, 200, 11, 22, 111, 222]
        options = [(0, 0, 0)]+[(code, 0, span) for code in codes for span in (1, 5, 20)]
        def group(index):
            code, _, span = options[index]
            return int(np.count_nonzero(code//(10**np.arange(3)) % 10)), span
        legacy = candidate_screen_order(options, 3, strategy='interleaved')
        self.assertEqual(legacy[:3], [options.index((111, 0, span)) for span in (1, 5, 20)])
        first = candidate_screen_order(options, 3, strategy='root_randomized', root_seed=[42, 123])
        self.assertEqual(first, candidate_screen_order(options, 3,
            strategy='root_randomized', root_seed=[42, 123]))
        self.assertEqual(sorted(first), list(range(1, len(options))))
        self.assertEqual(len({group(index) for index in first[:9]}), 9)
        self.assertNotEqual(first, candidate_screen_order(options, 3,
            strategy='root_randomized', root_seed=[42, 124]))
        boats = set()
        for root in range(20):
            screen = candidate_screen_order(options, 3, strategy='root_randomized', root_seed=[42, root])[:9]
            for index in screen:
                digits = options[index][0]//(10**np.arange(3)) % 10
                if np.count_nonzero(digits) == 1:
                    boats.add(int(np.flatnonzero(digits)[0]))
        self.assertEqual(boats, {0, 1, 2})
        with self.assertRaisesRegex(ValueError, 'root seed'):
            candidate_screen_order(options, 3, strategy='root_randomized')

    def test_root_sampling_balances_states_without_discarding_observations(self):
        import numpy as np
        from plain_return import root_sampling_probabilities
        roots = np.array([10]*10000+[73])
        self.assertIsNone(root_sampling_probabilities(roots))
        for fraction in (.5, 1.):
            probability = root_sampling_probabilities(roots, fraction)
            self.assertTrue((probability > 0).all())
            self.assertAlmostEqual(float(probability.sum()), 1.)
            self.assertAlmostEqual(float(probability[roots == 73].sum()),
                                   (1-fraction)/len(roots)+fraction/2.)
        for fraction in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                root_sampling_probabilities(roots, fraction)

    def test_identity_requires_measured_permutations_and_preserves_labels(self):
        import copy
        import numpy as np
        from conditional_fusion_value import append_vessel_identity
        from local_intervention_value import retain_vessel_identity
        context = np.arange(21, dtype=np.float32)
        root = dict(context=context, order=np.array([1, 0]), outcomes={(0, 0): dict(success=1.)})
        data = dict(roots=[root], predecessor_sha256='identity-test')
        bank = dict(predecessor_sha256='identity-test', records=[dict(contexts=context[None],
            orders=np.array([[1, 0]]), values=np.array([1.], np.float32))])
        transformed, factual = retain_vessel_identity(data, bank)
        np.testing.assert_array_equal(transformed['roots'][0]['context'][16:18], [1., 0.])
        np.testing.assert_array_equal(transformed['roots'][0]['context'][-5:], context[-5:])
        np.testing.assert_array_equal(factual['records'][0]['contexts'][0], transformed['roots'][0]['context'])
        self.assertEqual(transformed['roots'][0]['outcomes'], root['outcomes'])
        self.assertEqual(len(data['roots'][0]['context']), 21)
        with self.assertRaisesRegex(ValueError, 'permutation'):
            append_vessel_identity(torch.from_numpy(context), torch.tensor([0, 0]))
        missing = copy.deepcopy(bank); missing['records'][0].pop('orders')
        with self.assertRaisesRegex(ValueError, 'measured vessel orders'):
            retain_vessel_identity(data, missing)
        missing = copy.deepcopy(data); missing['roots'][0].pop('order')
        with self.assertRaisesRegex(ValueError, 'measured physical vessel order'):
            retain_vessel_identity(missing)

    def test_post_intervention_bank_uses_its_measured_vessel_order(self):
        import copy
        import numpy as np
        from local_intervention_value import paired_continuation_bank, retain_vessel_identity
        context = np.zeros(21, np.float32)
        value = dict(success=1., continuation_context=context.copy(),
                     continuation_controls=np.zeros((2, 6), np.float32),
                     continuation_order=np.array([1, 0]), continuation_remaining=10)
        root = dict(context=context.copy(), order=np.array([0, 1]),
                    scene=(123, 2, 2.), futures=[456], options=[(0, 0, 0)],
                    outcomes={(0, 0): value})
        data = dict(roots=[root], predecessor_sha256='continuation-identity', completed=True)
        bank = paired_continuation_bank(data)
        np.testing.assert_array_equal(bank['records'][0]['orders'], [[1, 0]])
        _, exposed = retain_vessel_identity(data, bank)
        np.testing.assert_array_equal(exposed['records'][0]['contexts'][0, -7:-5], [1, 0])
        np.testing.assert_array_equal(exposed['records'][0]['values'], [1.])
        self.assertEqual(exposed['records'][0]['future'], 456)
        missing = copy.deepcopy(data)
        missing['roots'][0]['outcomes'][0, 0].pop('continuation_order')
        with self.assertRaises(ValueError):
            retain_vessel_identity(missing, paired_continuation_bank(missing))

    def test_catalog_respects_identity_and_equivariant_spatial_reordering(self):
        from conditional_fusion_value import append_vessel_identity
        from local_intervention_value import JointTaskPotential
        from plain_return import PlainReturnNetwork
        torch.manual_seed(123)
        raw = .05*torch.randn(2, 29)
        order = torch.tensor([[2, 0, 1], [1, 2, 0]])
        context = append_vessel_identity(raw, order)
        controls = .2*torch.randn(2, 3, 6)
        programs = torch.tensor([[0, 0, 0], [9, 0, 1], [0, 7, 0]])
        timing = torch.tensor([[0, 0], [0, 20], [5, 4]])
        permutation = torch.tensor([2, 0, 1])
        permuted = append_vessel_identity(torch.cat((raw[:, :24].reshape(2, 3, 8)
            [:, permutation].flatten(1), raw[:, -5:]), -1), order[:, permutation])
        for peer in (False, True):
            net = JointTaskPotential(width=8, context_conditioned=True,
                motion_basis=True, peer_conditioned=peer, identity_conditioned=True)
            actual = net.forward_catalog(context, controls, programs, timing)
            expected = torch.stack([net(context, controls, p[None].expand(2, -1),
                t[None].expand(2, -1)) for p, t in zip(programs, timing)], 1)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(actual, net.forward_catalog(permuted, controls[:, permutation],
                programs[:, permutation], timing), rtol=2e-5, atol=2e-6)
            changed = append_vessel_identity(raw, order[:, permutation])
            self.assertGreater(float((actual-net.forward_catalog(changed, controls, programs, timing))
                .detach().abs().max()), 1e-7)
        plain = PlainReturnNetwork(3, 8, identity_conditioned=True)
        expected = torch.stack([plain(context, controls, p[None].expand(2, -1),
            t[None].expand(2, -1)) for p, t in zip(programs, timing)], 1)
        torch.testing.assert_close(plain.forward_catalog(context, controls, programs, timing), expected)

    def test_matched_fitting_and_reload_share_recorded_identity_observations(self):
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import retain_vessel_identity, fit_joint_potential, LocalDifferenceDecision
        from plain_return import fit_plain_return, PlainReturnDecision
        context = np.zeros(21, np.float32); context[[4, 12]] = 1.
        controls = np.tile(np.array([.8, .3, -.4, .6, .1, -.2], np.float32), (2, 1))
        root = dict(context=context, order=np.array([1, 0]), effect_controls=controls, futures=[1, 2],
            options=[(0, 0, 0), (1, 0, 1), (10, 3, 2)],
            outcomes={(q, f): dict(success=float(q == f)) for q in range(3) for f in range(2)})
        data = dict(model={}, roots=[root], predecessor_sha256='identity-fit', simulation_steps=40)
        bank = dict(predecessor_sha256='identity-fit', trajectories=1, simulation_steps=10,
            records=[dict(contexts=context[None], orders=np.array([[1, 0]]), controls=controls[None],
                values=np.array([1.], np.float32), remaining=np.array([10.], np.float32))])
        data, bank = retain_vessel_identity(data, bank)
        with patch('local_intervention_value.torch.save'):
            model = fit_joint_potential(data, 'unused-identity-test.pth', updates=2, ensemble_size=1,
                width=4, device='cpu', factual_bank=bank, identity_conditioned=True, snapshot_updates=(1, 2))
        recipe = dict(model['successor_task_value']); recipe.pop('kind')
        conditional = LocalDifferenceDecision(**recipe)
        parameters = sum(p.numel() for p in conditional.models.parameters())
        with patch('plain_return.torch.save'):
            ordinary = fit_plain_return(data, 'unused-identity-test.pth', updates=2,
                parameter_budget=parameters, device='cpu', factual_bank=bank, identity_conditioned=True,
                snapshot_updates=(1, 2),
                expected_observation_sha256=model['successor_task_fit']['training_observation_sha256'])
        recipe = dict(ordinary['successor_task_value']); recipe.pop('kind')
        plain = PlainReturnDecision(**recipe)
        self.assertEqual(model['successor_task_value']['options'], ordinary['successor_task_value']['options'])
        self.assertTrue(conditional.models[0].identity_conditioned)
        self.assertTrue(plain.network.identity_conditioned)
        current = torch.from_numpy(data['roots'][0]['context'][None])
        predicted = conditional.values(current, torch.from_numpy(controls[None]))
        self.assertTrue(torch.isfinite(predicted).all())
        self.assertTrue(torch.isfinite(plain.network.forward_catalog(current, torch.from_numpy(controls[None]),
            plain.programs, plain.options[:, 1:])).all())
        from mechanism_controls import value_learning_checkpoint
        for trained, key in ((model, 'states'), (ordinary, 'state')):
            snapshots = trained['learning_snapshots']['states']
            first = snapshots[1][0] if key == 'states' else snapshots[1]
            final = snapshots[2][0] if key == 'states' else snapshots[2]
            expected = trained['successor_task_value'][key]
            expected = expected[0] if key == 'states' else expected
            self.assertTrue(any(not torch.equal(first[name], final[name]) for name in final))
            for name in final:
                torch.testing.assert_close(final[name], expected[name], rtol=0., atol=0.)
            checkpoint = value_learning_checkpoint(trained, 1, data, device='cpu')
            self.assertEqual(checkpoint['successor_task_fit']['updates'], 1)
            self.assertEqual(checkpoint['successor_task_fit']['total_parameter_updates'], 1)
            self.assertNotIn('learning_snapshots', checkpoint)
        # Recording CPU snapshots must clone their storage and leave the
        # optimization path itself unchanged.
        with patch('local_intervention_value.torch.save'):
            unrecorded = fit_joint_potential(data, 'unused-identity-test.pth', updates=2, ensemble_size=1,
                width=4, device='cpu', factual_bank=bank, identity_conditioned=True)
        for name, value in model['successor_task_value']['states'][0].items():
            torch.testing.assert_close(value, unrecorded['successor_task_value']['states'][0][name], rtol=0., atol=0.)
        # Uneven root populations exercise the same state-sampling change in
        # the conditional learner and the plain absolute-return learner.
        import copy
        smaller = copy.deepcopy(data['roots'][0])
        smaller['context'] = smaller['context']+.01
        smaller['outcomes'] = {key:value for key,value in smaller['outcomes'].items() if key[0] < 2}
        unequal = dict(data, roots=[data['roots'][0], smaller])
        with patch('local_intervention_value.torch.save'):
            balanced = fit_joint_potential(unequal, 'unused-identity-test.pth', updates=2, ensemble_size=1,
                width=4, device='cpu', factual_bank=bank, identity_conditioned=True, root_sampling_fraction=.5)
        with patch('plain_return.torch.save'):
            balanced_plain = fit_plain_return(unequal, 'unused-identity-test.pth', updates=2,
                parameter_budget=parameters, device='cpu', factual_bank=bank, identity_conditioned=True,
                root_sampling_fraction=.5,
                expected_observation_sha256=balanced['successor_task_fit']['training_observation_sha256'])
        self.assertEqual(balanced['successor_task_fit']['root_sampling_fraction'], .5)
        self.assertEqual(balanced_plain['successor_task_fit']['root_sampling_fraction'], .5)
        self.assertTrue(np.isfinite(balanced_plain['successor_task_fit']['last_loss']))


class FactualCrossfitTests(unittest.TestCase):
    def test_continued_fits_preserve_layout_separation_and_own_parameter_history(self):
        import copy
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import fit_joint_potential, joint_potential_initialization
        controls = np.tile(np.array([.8, .3, -.4, .6, .1, -.2], np.float32), (2, 1))
        roots = []
        for index in range(6):
            context = np.full(21, .02*index, np.float32); context[[4, 12]] = 1.
            roots.append(dict(context=context, effect_controls=controls, futures=[1, 2],
                scene=(100+index, 2, 2.), options=[(0, 0, 0), (1, 0, 5), (10, 0, 5)],
                outcomes={(q, f):dict(success=float((index+q+f) % 3 != 0))
                          for q in range(3) for f in range(2)}))
        data = dict(model={}, roots=roots, predecessor_sha256='continued-crossfit', simulation_steps=60)
        expanded = copy.deepcopy(data)
        expanded['roots'][0]['futures'].append(3)
        for q in range(3): expanded['roots'][0]['outcomes'][q, 2] = dict(success=float(q == 1))
        shared = dict(ensemble_size=3, width=4, seed=7, device='cpu', root_sampling_fraction=.5)
        fits = {}
        for ordinary in (False, True):
            settings = (dict(ordinary=True, ordinary_architecture='interaction') if ordinary else
                        dict(conditional_strata=True, factual_crossfit=True))
            with patch('local_intervention_value.torch.save'):
                original = fit_joint_potential(data, 'unused-continued-test.pth', updates=6, **shared, **settings)
            before = copy.deepcopy(original['successor_task_value']['states'])
            initialization = joint_potential_initialization(original, data)
            with patch('local_intervention_value.torch.save'):
                continued = fit_joint_potential(expanded, 'unused-continued-test.pth', updates=3,
                    **shared, **settings, **initialization)
            self.assertEqual(continued['successor_task_fit']['updates'], 3)
            self.assertEqual(continued['successor_task_fit']['cumulative_optimizer_steps'], 9)
            self.assertEqual(continued['successor_task_fit']['initialization_parameter_sha256'],
                             initialization['initial_training_context']['source_parameter_sha256'])
            self.assertNotEqual(continued['successor_task_fit']['training_observation_sha256'],
                                original['successor_task_fit']['training_observation_sha256'])
            self.assertTrue(any(not torch.equal(tensor, continued['successor_task_value']['states'][0][key])
                                for key, tensor in before[0].items()))
            for prior, after in zip(before, original['successor_task_value']['states']):
                for key, tensor in prior.items(): torch.testing.assert_close(tensor, after[key], atol=0., rtol=0.)
            with self.assertRaisesRegex(ValueError, 'observations do not match'):
                joint_potential_initialization(original, expanded)
            tampered = copy.deepcopy(initialization)
            first_key = next(key for key, value in tampered['initial_states'][0].items() if value.is_floating_point())
            tampered['initial_states'][0][first_key].add_(1.)
            with self.assertRaisesRegex(ValueError, 'parameters do not match'):
                fit_joint_potential(expanded, 'unused-continued-test.pth', updates=3,
                                    **shared, **settings, **tampered)
            if not ordinary:
                changed = copy.deepcopy(expanded)
                for index, fold in enumerate(original['successor_task_fit']['factual_crossfit_root_folds']):
                    if fold == 0:
                        for outcome in changed['roots'][index]['outcomes'].values(): outcome['success'] = 1.
                with patch('local_intervention_value.torch.save'):
                    altered = fit_joint_potential(changed, 'unused-continued-test.pth', updates=3,
                                                 **shared, **settings, **initialization)
                for key, tensor in continued['successor_task_value']['states'][0].items():
                    torch.testing.assert_close(tensor, altered['successor_task_value']['states'][0][key], atol=0., rtol=0.)
                invalid = copy.deepcopy(initialization)
                scene = next(iter(invalid['initial_training_context']['layout_folds']))
                invalid['initial_training_context']['layout_folds'][scene] = (
                    invalid['initial_training_context']['layout_folds'][scene]+1) % 3
                with self.assertRaisesRegex(ValueError, 'held-out fold'):
                    fit_joint_potential(expanded, 'unused-continued-test.pth', updates=3,
                                        **shared, **settings, **invalid)
                with self.assertRaisesRegex(ValueError, 'preserve its estimator'):
                    fit_joint_potential(expanded, 'unused-continued-test.pth', updates=3,
                        ordinary=True, ordinary_architecture='interaction', **shared, **initialization)
            fits[ordinary] = continued['successor_task_fit']
        self.assertEqual(fits[False]['training_observation_sha256'], fits[True]['training_observation_sha256'])
        self.assertEqual(fits[False]['training_target_sha256'], fits[True]['training_target_sha256'])

    def test_factual_calibration_ablation_retains_temporal_statistics(self):
        from local_intervention_value import JointTaskPotential, LocalDifferenceDecision
        net = JointTaskPotential(width=4, output_width=3)
        with torch.no_grad():
            for parameter in net.parameters():
                parameter.zero_()
            net.head[-1].bias.copy_(torch.tensor([.2, .7, .3]).logit())
        context = torch.zeros((1, 21)); controls = torch.zeros((1, 2, 6))
        learner = LocalDifferenceDecision(width=4, states=[net.state_dict()], conditional_strata=True,
            contexts=context, controls=controls, options=[(0, 0, 0), (1, 5, 3)],
            corrections=torch.zeros((1, 2)), uncertainty=torch.zeros((1, 2)),
            observed=torch.ones((1, 2), dtype=torch.bool), factual_values=torch.tensor([.6]),
            neural_factual_calibration=[[.3, .9]], joint_factual_calibration=True,
            continuation_contexts=context.repeat(2, 1), continuation_controls=controls.repeat(2, 1, 1),
            continuation_values=torch.tensor([0., 1.]), continuation_remaining=torch.tensor([2., 10.]),
            temporal_value_bounds=True)
        before = learner.continuation_statistics(context, controls)
        calibrated = learner.values(context, controls)
        torch.testing.assert_close(calibrated[:, :, 0], torch.tensor([[1.6/3.]]))
        self.assertEqual(float(before[2][0, 1]), 0.)
        self.assertEqual(float(before[3][0, 1]), 1.)
        learner.factual_probability_calibration = False
        uncalibrated = learner.values(context, controls)
        torch.testing.assert_close(uncalibrated, torch.tensor([[[.2, .7]]]))
        torch.testing.assert_close(learner.calibrate_factual_probability(torch.tensor([.2]), 0),
                                   torch.tensor([.2]))
        for first, second in zip(before, learner.continuation_statistics(context, controls)):
            torch.testing.assert_close(first, second, rtol=0., atol=0.)

    def test_logit_map_recovers_monotone_risk_without_a_probability_floor(self):
        import numpy as np
        from local_intervention_value import fit_factual_logit_calibration
        probability = np.linspace(.02, .995, 100)
        logits = np.log(probability)-np.log1p(-probability)
        target = 1./(1.+np.exp(-(.4*logits-.7)))
        parameters, stats = fit_factual_logit_calibration(probability, target, ridge=1e-6)
        np.testing.assert_allclose(parameters, [.4, -.7], atol=2e-4)
        self.assertLess(stats['calibrated_brier'], stats['raw_brier']/1000.)
        self.assertLess(stats['calibrated_log_loss'], stats['raw_log_loss'])
        reversed_parameters, _ = fit_factual_logit_calibration(probability, 1.-probability)
        self.assertEqual(float(reversed_parameters[0]), 0.)
        with self.assertRaises(ValueError):
            fit_factual_logit_calibration([.5], [1.], [0.])

    def test_probability_map_is_bounded_and_corrects_overconfidence(self):
        import numpy as np
        from local_intervention_value import fit_factual_probability_calibration, factual_crossfit_groups
        p = np.linspace(.05, .995, 80)
        y = .75+.1*p
        endpoints, stats = fit_factual_probability_calibration(p, y)
        self.assertGreaterEqual(endpoints[0], 0.)
        self.assertLessEqual(endpoints[1], 1.)
        self.assertLessEqual(endpoints[0], endpoints[1])
        self.assertLess(stats['calibrated_brier'], stats['raw_brier']/20.)
        self.assertLess(endpoints[1], .9)
        with self.assertRaises(ValueError):
            fit_factual_probability_calibration([.5], [1.], [0.])
        roots = [dict(scene=(seed, 2, 2.)) for seed in (1, 1, 2, 3, 4, 5, 6)]
        bank = dict(records=[dict(scene=(seed, 2, 2.)) for seed in (1, 4, 7)])
        root_folds, bank_folds, groups = factual_crossfit_groups(roots, bank, 3, 42)
        self.assertEqual(root_folds[0], root_folds[1])
        self.assertEqual(root_folds[0], bank_folds[0])
        self.assertEqual(set(root_folds), {0, 1, 2})
        self.assertEqual(len(groups), 7)

    def test_held_out_labels_do_not_train_their_member_and_snapshots_keep_calibration(self):
        import copy
        import numpy as np
        from unittest.mock import patch
        from local_intervention_value import fit_joint_potential, LocalDifferenceDecision
        from mechanism_controls import value_learning_checkpoint
        from plain_return import fit_plain_return
        roots = []; records = []
        controls = np.tile(np.array([.8, .3, -.4, .6, .1, -.2], np.float32), (2, 1))
        for index in range(6):
            context = np.full(21, .02*index, np.float32); context[[4, 12]] = 1.
            scene = (100+index, 2, 2.)
            roots.append(dict(context=context, effect_controls=controls, futures=[1, 2], scene=scene,
                options=[(0, 0, 0), (1, 0, 5), (10, 0, 5), (11, 0, 5)],
                outcomes={(q, f):dict(success=float((index+q+f) % 3 != 0))
                          for q in range(4) for f in range(2)}))
            records.append(dict(contexts=np.stack((context, context+.001)),
                controls=np.stack((controls, controls)), values=np.array([index % 2]*2, np.float32),
                remaining=np.array([20., 15.], np.float32), scene=scene))
        data = dict(model={}, roots=roots, predecessor_sha256='crossfit', simulation_steps=60)
        bank = dict(records=records, predecessor_sha256='crossfit', trajectories=6, simulation_steps=30)
        settings = dict(updates=12, ensemble_size=3, width=4, device='cpu', conditional_strata=True,
            factual_crossfit=True, root_sampling_fraction=.5, snapshot_updates=(6, 12))
        with patch('local_intervention_value.torch.save'):
            model = fit_joint_potential(data, 'unused-crossfit-test.pth', factual_bank=bank, **settings)
        fit = model['successor_task_fit']; recipe = dict(model['successor_task_value']); recipe.pop('kind')
        self.assertEqual(fit['factual_crossfit_root_folds'], fit['factual_crossfit_trajectory_folds'])
        self.assertEqual(len(fit['factual_calibration_diagnostics']), 3)
        learner = LocalDifferenceDecision(**recipe)
        values = learner.values(torch.tensor(np.array([root['context'] for root in roots])),
                                torch.tensor(np.stack([controls]*6)))
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue(((values >= 0) & (values <= 1)).all())
        checkpoint = value_learning_checkpoint(model, 6, data, device='cpu')
        np.testing.assert_array_equal(checkpoint['successor_task_value']['neural_factual_calibration'],
                                      model['learning_snapshots']['neural_factual_calibration'][6])
        parameters = sum(p.numel() for p in learner.models.parameters())+fit['factual_calibration_parameters']
        with patch('plain_return.torch.save'):
            ordinary = fit_plain_return(data, 'unused-crossfit-test.pth', factual_bank=bank, updates=12,
                parameter_budget=parameters, device='cpu', root_sampling_fraction=.5,
                expected_observation_sha256=fit['training_observation_sha256'])
        self.assertEqual(ordinary['successor_task_fit']['training_observation_sha256'], fit['training_observation_sha256'])
        changed = copy.deepcopy(data); changed_bank = copy.deepcopy(bank)
        for index, fold in enumerate(fit['factual_crossfit_root_folds']):
            if fold == 0:
                for outcome in changed['roots'][index]['outcomes'].values():outcome['success'] = 1.
                changed_bank['records'][index]['values'][:] = 1.
        with patch('local_intervention_value.torch.save'):
            refit = fit_joint_potential(changed, 'unused-crossfit-test.pth', factual_bank=changed_bank, **settings)
        for key, tensor in model['successor_task_value']['states'][0].items():
            torch.testing.assert_close(tensor, refit['successor_task_value']['states'][0][key], atol=0., rtol=0.)
        self.assertGreater(float(np.abs(model['successor_task_value']['neural_factual_calibration'][0]-
            refit['successor_task_value']['neural_factual_calibration'][0]).max()), .01)
        with patch('local_intervention_value.torch.save'):
            logit_model = fit_joint_potential(data, 'unused-crossfit-test.pth', factual_bank=bank,
                factual_calibration_kind='logit', **settings)
        for member, state in enumerate(model['successor_task_value']['states']):
            for key, tensor in state.items():
                torch.testing.assert_close(tensor, logit_model['successor_task_value']['states'][member][key],
                                           atol=0., rtol=0.)
        self.assertEqual(logit_model['successor_task_fit']['training_observation_sha256'],
                         fit['training_observation_sha256'])
        logit_checkpoint = value_learning_checkpoint(logit_model, 6, data, device='cpu')
        self.assertEqual(logit_checkpoint['successor_task_value']['neural_factual_calibration_kind'], 'logit')
        logit_recipe = dict(logit_model['successor_task_value']); logit_recipe.pop('kind')
        logit_learner = LocalDifferenceDecision(**logit_recipe)
        query = torch.tensor([[0., .1, .9, 1.]]).expand(3, -1)
        calibrated = logit_learner.calibrate_factual_probability(query)
        self.assertTrue(torch.isfinite(calibrated).all())
        self.assertTrue((calibrated[:, 1:] >= calibrated[:, :-1]).all())
        for member in range(3):
            torch.testing.assert_close(calibrated[member],
                logit_learner.calibrate_factual_probability(query[member], member))
        logit_recipe['neural_factual_calibration'] = [[-1., 0.]]*3
        with self.assertRaises(ValueError):
            LocalDifferenceDecision(**logit_recipe)


if __name__ == '__main__':
    unittest.main()
