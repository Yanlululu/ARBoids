"""Regression checks for the retained first mechanism."""
import unittest
import torch


class ActuatorFusionTests(unittest.TestCase):
    def test_terminal_fleet_credit_preserves_late_success_and_failure(self):
        from fusion_training import fleet_return_targets, fleet_anchor_kl
        def episode(success, count):
            return dict(row=dict(success=success), fleet_steps=[dict(value=0., discount=.5,
                reward=float(success) if step==count-1 else 0., done=step==count-1)
                for step in range(count)])
        observations=[episode(True,1),episode(True,4),episode(False,2)]
        records,advantages,targets=fleet_return_targets(observations,'terminal_success')
        self.assertEqual(len(records),7)
        self.assertEqual(targets,[1.]*5+[0.]*2)
        self.assertEqual(advantages,[.5]*5+[-.5]*2)
        _,discounted,_=fleet_return_targets(observations)
        self.assertLess(discounted[1],discounted[0])
        with self.assertRaisesRegex(ValueError,'terminal outcomes'):
            fleet_return_targets([episode(.5,1)],'terminal_success')
        paired=[dict(episode(success,2),factual_success=factual)
                for success,factual in ((1,1),(0,0),(1,0),(0,1))]
        _,paired_advantages,paired_targets=fleet_return_targets(paired,'paired_terminal_success')
        self.assertEqual(paired_targets,[0.,0.,0.,0.,1.,1.,-1.,-1.])
        self.assertEqual(paired_advantages,paired_targets)
        paired[2]['fleet_steps'][0]['value']=.25
        _,paired_advantages,_=fleet_return_targets(paired,'paired_terminal_success')
        self.assertEqual(paired_advantages[4],.75)
        with self.assertRaisesRegex(ValueError,'measured fixed-predecessor'):
            fleet_return_targets([episode(1,1)],'paired_terminal_success')
        reference=torch.distributions.Categorical(logits=torch.tensor([[0.,-90.]]))
        logits=torch.tensor([[0.,-180.]],requires_grad=True)
        current=torch.distributions.Categorical(logits=logits)
        value=fleet_anchor_kl(reference,current)
        self.assertTrue(bool(torch.isfinite(value).all()))
        value.sum().backward()
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        torch.testing.assert_close(fleet_anchor_kl(reference,reference),torch.zeros(1))

    def test_fixed_inference_preserves_rows_when_pair_groups_shrink_and_reorder(self):
        from fusion_prediction import fixed_batch_inference
        packet={'x':torch.arange(21,dtype=torch.float32).reshape(7,3)}
        offsets=torch.arange(7,dtype=torch.float32)[:,None]
        def numerical_kernel(batch,shift):
            # Simulate shape-dependent arithmetic from the actual GPU kernel.
            value=batch['x']+shift+len(shift)*1.e-4
            return value,value[:, :1]*2.
        whole=fixed_batch_inference(numerical_kernel,packet,offsets,width=4)
        order=torch.tensor([6,2,1,5,0,4,3])
        chunks=[order[:2],order[2:5],order[5:]]
        pieces=[fixed_batch_inference(numerical_kernel,{'x':packet['x'][ids]},
            offsets[ids],width=4) for ids in chunks]
        for index,expected in enumerate(whole):
            torch.testing.assert_close(torch.cat([piece[index] for piece in pieces])[order.argsort()],
                expected,rtol=0.,atol=0.)
        self.assertFalse(torch.equal(numerical_kernel(packet,offsets)[0],whole[0]))


    def test_candidate_attention_preserves_inputs_capacity_and_learning(self):
        from actuator_fusion import FeedbackFleetPolicy
        torch.manual_seed(43)
        graph=FeedbackFleetPolicy()
        attention=FeedbackFleetPolicy(representation='candidate_attention')
        self.assertEqual(sum(p.numel() for p in attention.parameters())-
                         sum(p.numel() for p in graph.parameters()),5)
        for defenders in (3,6):
            features=torch.randn(2,7,defenders,37)
            costs=torch.rand(2,7)*3.
            torch.testing.assert_close(graph(features,costs)[0],attention(features,costs)[0],rtol=0.,atol=0.)
        torch.nn.init.normal_(attention.actor.weight,std=.1)
        attention.eval()
        permutation=torch.tensor([5,0,3,1,4,2])
        logits,values=attention(features,costs)
        permuted_logits,permuted_values=attention(features[:,:,permutation],costs)
        torch.testing.assert_close(logits,permuted_logits)
        torch.testing.assert_close(values,permuted_values)
        clone=FeedbackFleetPolicy(representation='candidate_attention',state=attention.state_dict()).eval()
        torch.testing.assert_close(clone(features,costs)[0],logits)
        attention.train();logits,values=attention(features,costs)
        (-logits.log_softmax(-1)[:,0].mean()+values.square().mean()).backward()
        self.assertTrue(any(p.grad is not None and bool(p.grad.any()) for p in attention.attention.parameters()))
        with self.assertRaisesRegex(ValueError,'divisible by four'):
            FeedbackFleetPolicy(width=15,representation='candidate_attention')

    def test_complete_scenario_scores_average_after_nonlinear_fleet_evaluation(self):
        from actuator_fusion import FeedbackFleetPolicy
        torch.manual_seed(741)
        ordinary=FeedbackFleetPolicy(width=8,candidate_contrast=True).eval()
        torch.nn.init.normal_(ordinary.actor.weight,std=.1)
        policy=FeedbackFleetPolicy(width=8,candidate_contrast=True,
            scenario_evaluation=True,state=ordinary.state_dict()).eval()
        features=torch.randn(2,100,2,3,37);costs=torch.rand(2,100,2)*3.
        weights=torch.tensor([.3,.7])
        packed=torch.cat((features,costs[...,None,None].expand(-1,-1,-1,3,1),
            weights[None,None,:,None,None].expand(2,100,-1,3,1)),-1)
        first=ordinary(features[:,:,0],costs[:,:,0]);second=ordinary(features[:,:,1],costs[:,:,1])
        logits,values=policy(packed,(costs*weights).sum(-1))
        torch.testing.assert_close(logits,.3*first[0]+.7*second[0],rtol=0.,atol=0.)
        torch.testing.assert_close(values,.3*first[1]+.7*second[1],rtol=0.,atol=0.)
        self.assertEqual(sum(p.numel() for p in policy.parameters()),sum(p.numel() for p in ordinary.parameters()))
        singleton=packed[:,:,:1].clone();singleton[...,38]=1.
        for actual,expected in zip(policy(singleton,costs[:,:,0]),first):
            torch.testing.assert_close(actual,expected,rtol=0.,atol=0.)
        (-logits.log_softmax(-1)[:,0].mean()+values.square().mean()).backward()
        self.assertTrue(bool(policy.message[0].weight.grad.abs().sum()>0.))
        with self.assertRaisesRegex(ValueError,'Scenario scoring'):
            ordinary(packed,(costs*weights).sum(-1))
        robust=FeedbackFleetPolicy(width=8,candidate_contrast=True,scenario_evaluation=True,
            scenario_regret_weight=.5,state=ordinary.state_dict()).eval()
        robust_logits,robust_values=robust(packed,(costs*weights).sum(-1))
        scores=torch.stack((first[0],second[0]),-1)
        regret=scores.amax(1,keepdim=True)-scores
        expected=-.5*((regret*weights).sum(-1)+regret.amax(-1))
        torch.testing.assert_close(robust_logits,expected,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(robust_values,values,rtol=0.,atol=0.)
        ignored=packed.clone();ignored[:,:,0,:,38]=1.;ignored[:,:,1,:,38]=0.
        ignored_logits,_=robust(ignored,costs[:,:,0])
        torch.testing.assert_close(ignored_logits,first[0]-first[0].amax(-1,keepdim=True),rtol=0.,atol=0.)
        self.assertEqual(sum(p.numel() for p in robust.parameters()),sum(p.numel() for p in ordinary.parameters()))
        with self.assertRaisesRegex(ValueError,'Scenario regret'):
            FeedbackFleetPolicy(width=8,scenario_regret_weight=.5)


    def test_target_scenarios_use_their_own_fleet_feedback_paths(self):
        from actuator_fusion import task_envelope_fusion_factors, task_stopped_fleet_cost
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
        from actuator_fusion import coupled_actuator_risk
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
        from actuator_fusion import feedback_fleet_library, select_feedback_fleet_plan
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
        from actuator_fusion import coupled_actuator_risk
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
        from actuator_fusion import (task_envelope_fusion_factors, task_stopped_fleet_cost,
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
        from actuator_fusion import (target_turning_envelope, task_envelope_fusion_factors,
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

    def test_task_stopped_independent_matches_exhaustive_fleet_expectations(self):
        import itertools
        from actuator_fusion import (task_stopped_fusion_factors, task_stopped_fleet_cost,
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
        from actuator_fusion import task_stopped_fusion_factors, task_stopped_fleet_cost
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

    def test_independent_interception_cost_matches_complete_fleet_expectation(self):
        from itertools import product
        from actuator_fusion import independent_fleet_minimum_cost
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

    def test_actuator_resolution_retains_source_hull_and_exact_reference(self):
        from actuator_fusion import actuator_candidate_grid
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

    def test_componentwise_refinement_rejects_compensating_individual_harm(self):
        from actuator_fusion import pareto_fleet_refinement
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

    def test_feedback_guard_checks_every_horizon_and_restores_boat_order(self):
        from unittest.mock import patch
        from actuator_fusion import feedback_guarded_refinement
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
        with patch('actuator_fusion.coupled_actuator_risk', side_effect=forecast):
            actual = feedback_guarded_refinement(
                object(), packet, source, candidates, order, reference, proposed, (.8, 1.6, 3.2))
        self.assertEqual(calls, [.8, 1.6, 3.2])
        torch.testing.assert_close(actual, torch.tensor([[0, 0], [0, 0], [1, 1], [0, 0]]))
        with patch('actuator_fusion.coupled_actuator_risk') as forecast_mock:
            feedback_guarded_refinement(object(), packet, source, candidates, order, reference, reference, (.8,))
            forecast_mock.assert_not_called()

    def test_feedback_task_guard_rejects_lost_capture_or_asset_progress(self):
        from unittest.mock import patch
        from actuator_fusion import feedback_guarded_refinement
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
        with patch('actuator_fusion.coupled_actuator_risk', side_effect=forecast):
            result = feedback_guarded_refinement(object(), {}, source, candidates, order,
                reference, proposed, (.8, 1.6, 3.2), tail_horizon=1.6, task_guard=True)
        torch.testing.assert_close(result, torch.tensor([[0, 0], [0, 0], [1, 1]]))
        with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
            feedback_guarded_refinement(object(), {}, source, candidates, order,
                reference, proposed, (.8,), tail_horizon=float('nan'))

    def test_coupled_task_guard_tracks_public_target_without_changing_risk(self):
        from types import SimpleNamespace
        from actuator_fusion import coupled_actuator_risk
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

    def test_pareto_refinement_rejects_risk_deviation_tradeoffs_and_ties(self):
        from actuator_fusion import pareto_fleet_refinement
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
        from actuator_fusion import refine_pair_blocks
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
        from actuator_fusion import distinct_fleet_shortlist
        plans = torch.tensor([[[1, 1], [1, 1], [0, 1], [0, 0]],
                              [[0, 0], [0, 0], [0, 0], [0, 0]]])
        scores = torch.tensor([[.1, .2, .4, .3], [.1, .2, .3, .4]])
        selected = distinct_fleet_shortlist(plans, scores, 3)
        torch.testing.assert_close(selected, torch.tensor(
            [[[1, 1], [0, 0], [0, 1]], [[0, 0], [0, 0], [0, 0]]]))


    def test_public_response_preserves_vessel_permutations_and_fleet_sizes(self):
        from joint_fusion import PublicTargetAcceleration,PublicTargetFeedback
        torch.manual_seed(319)
        network=PublicTargetAcceleration(8,True)
        torch.nn.init.normal_(network.readout[-1].weight,std=.1)
        predictor=PublicTargetFeedback(width=8,states=[network.state_dict()],
            representation='interaction_acceleration',kinematic_weight=.5)
        for count in (3,6,8):
            motion=torch.randn(2,count,6);boids=torch.randn(2,count,2).tanh()
            target=torch.tensor([[30.,10.],[-20.,15.]])
            velocity=torch.tensor([[0.,0.],[2.,-.4]])
            permutation=torch.randperm(count)
            expected=predictor.predict_state(motion,boids,target,velocity)
            actual=predictor.predict_state(motion[:,permutation],boids[:,permutation],target,velocity)
            for left,right in zip(expected,actual):
                self.assertTrue(torch.isfinite(left).all())
                torch.testing.assert_close(left,right,atol=1e-6,rtol=1e-6)
            linear=predictor.kinematic_component().predict_state(motion,boids,target,velocity)
            torch.testing.assert_close(linear[0],target+.2*velocity)
            torch.testing.assert_close(linear[1],velocity)

    def test_prediction_validation_keeps_initial_and_later_states_together(self):
        import numpy as np
        from fusion_prediction import target_response_validation_roots
        families=[(1,6,2.),(2,6,2.),(3,6,2.)]
        data=dict(specification=dict(root_scenes=families+families))
        held=target_response_validation_roots(data,np.arange(6),319,[0,3])
        self.assertEqual(held.tolist(),[0,3])
        with self.assertRaisesRegex(ValueError,'same layout'):
            target_response_validation_roots(data,np.arange(6),319,[0])


class JointFusionRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        from pathlib import Path
        from joint_fusion import JointFusionActor, atomic_save
        torch.set_num_threads(1)
        torch.manual_seed(431)
        cls.directory=tempfile.TemporaryDirectory()
        cls.checkpoint=Path(cls.directory.name)/'m1.pth'
        actor=JointFusionActor(risk_horizon=.4)
        cls.record=dict(actor=actor.state_dict(),risk_horizon=.4,
            actuator_fusion_refinement=dict(actor.actuator_fusion_refinement),
            mechanism_ablation=(False,True,False))
        atomic_save(cls.record,cls.checkpoint)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_numerical_and_ros_commands_match_with_padding_and_reset(self):
        import numpy as np
        from joint_fusion import JointMeasuredController, joint_packet, mechanism_tensor, numerical_source
        from source_arboids import source_mixture
        controller=JointMeasuredController(self.checkpoint)
        for n in (3,6):
            controller.reset()
            np.random.seed(59)
            env=numerical_source().TADEnv(n)
            obs,_=env.reset(agility=2.,noisy_agility=False)
            for tick in range(2):
                packet=joint_packet(env,obs)
                expected=controller.actor.control(mechanism_tensor(packet))[0].numpy()
                defenders=np.asarray([[v.x,v.y,v.theta,*v.velocity] for v in env.defender_list])
                a=env.attacker
                attacker=np.asarray([a.x,a.y,a.theta,*a.velocity])
                observed=np.pad(obs,((0,0),(0,2))) if tick==0 else obs
                command,info=controller.control(defenders,attacker,observed,env.boids_actions,env.Current_T)
                np.testing.assert_allclose(info['physical_action'],expected,atol=1e-6,rtol=1e-6)
                np.testing.assert_allclose(command,source_mixture(expected,env.boids_actions),atol=1e-3,rtol=1e-6)
                with self.assertRaisesRegex(ValueError,'new measurement tick'):
                    controller.control(defenders,attacker,observed,env.boids_actions,env.Current_T)
                obs,_,_,_=env.step(expected,'AdaRes')

    def test_loader_rejects_retired_models_and_does_not_silently_drop_them(self):
        from pathlib import Path
        from joint_fusion import joint_load, atomic_save
        path=Path(self.directory.name)/'retired.pth'
        for field in ('local_conditional_value','successor_task_value','joint_intervention_values'):
            atomic_save(dict(self.record,**{field:{'state':torch.ones(1)}}),path)
            with self.assertRaisesRegex(ValueError,'Mechanism two'):
                joint_load(path)

    def test_reload_preserves_source_inference_and_gradient_mode(self):
        from pathlib import Path
        import numpy as np
        from joint_fusion import JointFusionActor, joint_load, atomic_save, numerical_source, joint_packet, mechanism_tensor
        path=Path(self.directory.name)/'roundtrip.pth'
        for frozen in (False,True):
            actor=JointFusionActor(risk_horizon=.4).eval()
            actor.load_state_dict(self.record['actor'])
            actor.base.requires_grad_(not frozen)
            atomic_save(dict(self.record,frozen_proposal=frozen),path)
            restored=joint_load(path)
            self.assertTrue(all(p.requires_grad == (not frozen) for p in restored.base.parameters()))
            for n in (3,6):
                np.random.seed(59)
                env=numerical_source().TADEnv(n)
                obs,_=env.reset(agility=2.,noisy_agility=False)
                packet=mechanism_tensor(joint_packet(env,obs))
                expected=actor.control(packet)
                actual=restored.control(packet)
                self.assertFalse(actual.requires_grad)
                torch.testing.assert_close(actual,expected,atol=0.,rtol=0.)

    def test_loader_rejects_invalid_selector_and_recursive_forecast_settings(self):
        from pathlib import Path
        from joint_fusion import joint_load, atomic_save
        path=Path(self.directory.name)/'invalid.pth'
        for settings in ({'beam_width':0},{'preference_width':0.},{'risk_weight':float('nan')},
                         {'selection':'unknown'},{'packet':{}}):
            atomic_save(dict(self.record,actuator_fusion_refinement=dict(
                self.record['actuator_fusion_refinement'],**settings)),path)
            with self.assertRaises(ValueError):joint_load(path)
        atomic_save(dict(self.record,prediction_iterations=-1),path)
        with self.assertRaisesRegex(ValueError,'forecast'):joint_load(path)

    def test_source_hash_and_checkpoint_revision_are_enforced(self):
        from pathlib import Path
        from joint_fusion import joint_load
        with self.assertRaisesRegex(ValueError,'changed'):
            joint_load(self.checkpoint,self.checkpoint.stat().st_mtime_ns+1)
        source=Path(self.directory.name)/'wrong-source.pth'
        source.write_bytes(b'different source')
        with self.assertRaisesRegex(ValueError,'source hash'):
            joint_load(self.checkpoint,source=source)

    def test_worker_initialization_preserves_autograd(self):
        from joint_fusion import initialize
        for enabled in (True,False):
            with torch.set_grad_enabled(enabled):
                initialize()
                self.assertEqual(torch.is_grad_enabled(),enabled)

    def test_deployment_requires_available_velocity_measurements(self):
        from pathlib import Path
        from joint_fusion import JointMeasuredController, atomic_save
        path=Path(self.directory.name)/'water-relative.pth'
        atomic_save(dict(self.record,current_marginal_prediction=True),path)
        with self.assertRaisesRegex(ValueError,'water-relative velocity'):
            JointMeasuredController(path)

    def test_observation_slots_and_physical_prediction_tick_are_validated(self):
        from joint_fusion import joint_load, JointFusionActor
        actor=joint_load(self.checkpoint)
        with self.assertRaisesRegex(ValueError,'other defender'):
            actor.proposal_distribution(torch.zeros(1,3,20))
        with self.assertRaisesRegex(ValueError,'physical control interval'):
            JointFusionActor(prediction_step=.07)

    def test_generic_student_uses_the_same_source_candidate_library(self):
        import numpy as np
        from pathlib import Path
        from joint_fusion import joint_load, joint_packet, mechanism_tensor, numerical_source, atomic_save
        from mechanism_controls import GenericFusionAdapter, generic_source_action, generic_fusion_features
        teacher=joint_load(self.checkpoint)
        net=GenericFusionAdapter(width=8)
        model=dict(self.record)
        model.pop('actuator_fusion_refinement')
        model['generic_fusion_adapter']=dict(width=8,state=net.state_dict())
        path=Path(self.directory.name)/'generic.pth'
        atomic_save(model,path)
        student=joint_load(path)
        np.random.seed(66)
        env=numerical_source().TADEnv(3);obs,_=env.reset(agility=2.,noisy_agility=False)
        packet=mechanism_tensor(joint_packet(env,obs))
        raw=generic_source_action(teacher,packet)
        torch.testing.assert_close(student(packet)[0],raw)
        _,candidates=generic_fusion_features(teacher,packet,raw)
        torch.testing.assert_close(candidates[:,:,4],teacher(packet)[0][...,2:3].expand(-1,-1,2))

    def test_public_dynamics_collection_records_layout_groups(self):
        import contextlib
        import hashlib
        import io
        import numpy as np
        from pathlib import Path
        from joint_fusion import atomic_save, numerical_source
        from fusion_prediction import collect_target_feedback_data
        np.random.seed(59)
        env=numerical_source().TADEnv(3);obs,_=env.reset(agility=2.,noisy_agility=False)
        bank=Path(self.directory.name)/'states.pth'
        output=Path(self.directory.name)/'responses.pth'
        scene=(59,3,2.)
        record=dict(scene=scene,root_snapshot=dict(environment=env,observation=obs))
        atomic_save(dict(completed=True,predecessor_sha256=hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            records=[record,record]),bank)
        with contextlib.redirect_stdout(io.StringIO()):
            data=collect_target_feedback_data(self.checkpoint,bank,output,steps=1,branches=1,batch_size=1,device='cpu')
        self.assertTrue(data['completed'])
        self.assertEqual(data['physical_steps'],2)
        self.assertEqual(data['specification']['root_scenes'],[scene,scene])

    def test_baseline_resume_finishes_a_fully_populated_incomplete_record(self):
        from pathlib import Path
        from unittest.mock import patch
        import mechanism_controls as controls
        from joint_fusion import atomic_save
        directory=Path(self.directory.name)
        tasks=[((59,3,2.),123)]
        atomic_save(dict(completed=False,policy_sha256='test-source',tasks=tasks,
            evaluation_context=None,rows=[dict(success=1)]),directory/'resume-test.pth')
        with patch.object(controls,'validate_author_source_repeat',return_value=(
                {'selected_checkpoint_sha256':'test-source'},None)),patch.object(controls,'author_source_repeat_episode') as episode:
            result=controls.evaluate_author_source_repeat(directory,tasks,workers=1,result_suffix='resume-test')
        episode.assert_not_called()
        self.assertTrue(result['completed'])
        self.assertEqual((result['evaluated'],result['success']),(1,1))

    def test_generic_replay_rejects_an_empty_work_design(self):
        from mechanism_controls import collect_generic_replay
        for budget,workers,tasks in ((0,1,[((1,3,2.),2)]),(10,0,[((1,3,2.),2)]),(10,1,[])):
            with self.assertRaises(ValueError):
                collect_generic_replay(self.checkpoint,tasks,'unused',budget=budget,workers=workers)


if __name__ == "__main__":
    unittest.main()
