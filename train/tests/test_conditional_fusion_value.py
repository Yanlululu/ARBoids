import unittest

import torch

from conditional_fusion_value import (
    component_relation_features, component_response_features, conditional_auxiliary, conditional_edges, paired_targets, paired_covariance_targets,
)


class ConditionalFusionValueTests(unittest.TestCase):
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


if __name__ == '__main__':
    unittest.main()
