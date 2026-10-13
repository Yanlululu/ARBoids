"""Integration checks for real CBF transitions, replay storage, and resumption."""
from pathlib import Path
import copy
import sys
import tempfile
import unittest
from unittest.mock import patch
import json

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'train'))
import study_runtime
import numpy as np
import torch
import yaml

from envs.TADgame import TADEnv
from envs.snapshot import RandomState, seed_random
from interaction_rollout import (public_packet, execute, safety_controller, SnapshotPool,
                                 InterventionSampler, DeploymentPolicy, CandidateRolePolicy)
from policy.interaction_sac import JointReplay, make_agent
from train_interaction import arm_config, run_training, export_policy, validate, retain_interventions
from interaction_review import check_endpoint, validation_calibration, validation_candidate_response, collect


def small_config(arm='full'):
    c = arm_config(yaml.safe_load((ROOT/'train/configs/interaction-aware-sac.yaml').read_text()), arm)
    c['rl'].update(hidden_dim=32, batch_size=4)
    c['training'].update(gate_steps=6, joint_steps=6, warm_steps=2, replay_capacity=16,
        eval_interval=12, eval_episodes=1, checkpoint_interval=6, log_interval=12,
        allow_random_initialization=True)
    c['interaction'].update(relation_dim=16, workers=0, pairs_per_batch=2, horizon_steps=2,
        intervention_interval=4, snapshot_interval=2, snapshot_capacity=8,entropy_objective='stage-mean-v2')
    return c


def equal(test, a, b):
    if isinstance(a, torch.Tensor):
        test.assertTrue(torch.equal(a, b))
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a: equal(test, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b): equal(test, x, y)
    else:
        test.assertEqual(a, b)


class FusionOpportunityContracts(unittest.TestCase):
    def test_unfiltered_training_and_evaluation_never_create_cbf(self):
        config = small_config('no_both')
        config['interaction']['safety'] = False
        validate(config)
        self.assertFalse(config['interaction']['peer_candidates'])
        self.assertEqual(config['interaction']['auxiliary'], 'none')
        with tempfile.TemporaryDirectory() as directory, patch(
                'interaction_rollout.safety_controller', side_effect=AssertionError('Unexpected CBF')):
            run_training(config, device='cpu', output=Path(directory))
            saved = torch.load(Path(directory)/'policy.pth', weights_only=True)
            self.assertFalse(saved['config']['interaction']['safety'])

    def test_centered_value_preserves_baseline_without_blocking_intervention_gradient(self):
        from diagnose_conditional_critic import centered_scores
        from policy.interaction_sac import TwinTeamCritic, tensor_packet
        seed_random(723119)
        env = TADEnv(3, protocol='paper-parameters-v1')
        env.reset(2.25, noisy_agility=False)
        packet = {k:v[None] for k,v in tensor_packet(public_packet(env)).items()}
        actions = torch.zeros((1,2,3,3))
        actions[:,0,:,2] = .5
        actions[:,1] = actions[:,0]
        actions[:,1,0,2] = 1.
        model = TwinTeamCritic(32,16,'nominal-thrust-v2')
        values = (torch.tensor([[7.]]),torch.tensor([[9.]]))
        scores = centered_scores(model,values,packet,actions)
        self.assertTrue(torch.allclose(scores[0][:,:1],values[0],atol=1e-6))
        baseline_gradient = torch.autograd.grad(sum(q[:,0].sum() for q in scores),tuple(model.parameters()),retain_graph=True)
        self.assertTrue(all(torch.count_nonzero(g)==0 for g in baseline_gradient))
        intervention_gradient = torch.autograd.grad(sum(q[:,1].sum() for q in scores),tuple(model.parameters()))
        self.assertTrue(any(torch.count_nonzero(g)>0 for g in intervention_gradient))

    def test_current_intervention_changes_only_one_gate(self):
        from diagnose_fusion_opportunity import single_boat_actions
        original = np.array([[.2, -.3, .4], [-.5, .1, .7], [.6, .4, .2]], dtype=np.float32)
        actions, interventions = single_boat_actions(original)
        np.testing.assert_array_equal(actions[0], original)
        for action, intervention in zip(actions[1:], interventions[1:]):
            np.testing.assert_array_equal(action[:, :2], original[:, :2])
            boat = intervention['boat']
            np.testing.assert_array_equal(np.delete(action, boat, 0), np.delete(original, boat, 0))
            self.assertEqual(action[boat, 2], intervention['gate'])

    def test_identical_real_branches_match_without_mutating_root(self):
        from diagnose_fusion_opportunity import complete_branch
        from interaction_rollout import compact_snapshot
        class FixedPolicy:
            def choose_action(self, packet, deterministic=True):
                return np.tile(np.array([.1, .2, .5], dtype=np.float32), (len(packet['obs']), 1)), 0.
        seed_random(1720000000)
        env = TADEnv(3, protocol='paper-parameters-v1')
        env.reset(2.25, noisy_agility=False)
        env.Current_T = 59.6
        snapshot = compact_snapshot(env)
        action = FixedPolicy().choose_action(public_packet(env))[0]
        with patch('diagnose_fusion_opportunity.POLICY', FixedPolicy()):
            left = complete_branch(snapshot, action, 1721000000)
            right = complete_branch(snapshot, action, 1721000000)
        self.assertEqual(left, right)
        self.assertEqual(snapshot.environment.Current_T, 59.6)
        self.assertIn(left['outcome'], (1, 2, 3, 4))


class CandidateRoles(unittest.TestCase):
    def test_linked_strata_bootstrap_retains_cross_stratum_pairs(self):
        from analyze_candidate_roles import analyze_linked_strata
        count = 128
        protocol = dict(stage='independent_locked_confirmation', files={},
            methods=['joint', 'additive'], count=count, conditions=[[6, 4., 1000], [6, 6., 2000]],
            primary_method='joint', primary_references=['additive'], familywise_alpha=.05,
            bootstrap_repetitions=2000, primary_population='synthetic linked-strata fixture')
        rows = []
        for cell, (_, agility, bank) in enumerate(protocol['conditions']):
            for i in range(count):
                offset = 4. if (i+cell) % 2 else -4.
                for method in protocol['methods']:
                    rows.append(dict(method=method, defenders=6, agility=agility, scene_seed=bank+i,
                        capture_time=10. if method == 'joint' else 18.+offset,
                        capture=1, success=1, collision=0, breach=0, timeout=0))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for filename, data in [('protocol.json', protocol),
                    ('results.json', dict(complete=True, protocol=protocol, episodes=rows)),
                    ('linked-strata-protocol.json', dict(conditions=protocol['conditions'], count=count))]:
                (root/filename).write_text(json.dumps(data), encoding='utf-8')
            result = analyze_linked_strata(root)
        self.assertEqual(result['independent_random_stream_blocks'], count)
        self.assertEqual(result['contrasts']['additive']['time_interval'], [-8., -8.])
        lower = result['contrasts']['additive']['preservation']['success']['conservative_lower']
        self.assertAlmostEqual(lower, .025**(1./count)-1.)
        self.assertTrue(result['criteria_passed'])

    def test_elapsed_time_bound_preserves_the_best_joint_forecast(self):
        from evaluate_candidate_roles import adaptive_guard_controller
        from feedback_joint_control import observe
        seed_random(1020000000)
        env = TADEnv(6, protocol='paper-parameters-v1')
        obs, _ = env.reset(4., noisy_agility=False)
        measured = observe(env, obs)
        controller = adaptive_guard_controller(6)
        original = [controller.predict(measured, t) for t in controller.options]
        best = min(original, key=lambda p: p['score'])
        self.assertEqual(best['score'][:2], (0, 0))
        controller.prediction_bound = lambda: best['score'][2]
        bounded = [controller.predict(measured, t) for t in controller.options]
        self.assertTrue(any(p.get('pruned') for p in bounded))
        actual = min(bounded, key=lambda p: p['score'])
        self.assertEqual(actual['template'], best['template'])
        for key in ('score', 'positions', 'attackers', 'commands'):
            np.testing.assert_array_equal(actual[key], best[key])

    def test_adaptive_guard_bank_preserves_both_control_families(self):
        from evaluate_candidate_roles import adaptive_guard_controller, ConditionalGuardResidual
        from policy.role_residual import RoleResidualPolicy
        from feedback_joint_control import observe
        seed_random(86)
        env = TADEnv(6, protocol='paper-parameters-v1')
        obs, _ = env.reset(4., noisy_agility=False)
        measured = observe(env, obs)
        packet = dict(obs=np.asarray(measured.observations, dtype=np.float32), motion=measured.defenders)
        controller = adaptive_guard_controller(6)
        point, guard = RoleResidualPolicy(), ConditionalGuardResidual()
        self.assertEqual(len(controller.options), 43)
        for name, mask in controller.options.items():
            expected = (guard if name.startswith('guard-') else point).compose(packet, mask)
            np.testing.assert_array_equal(controller.nominal(measured, name), expected)
        np.testing.assert_array_equal(controller.nominal(measured, 'baseline'),
                                      CandidateRolePolicy().choose_action(packet)[0])

    def test_guard_handover_keeps_zero_residual_and_fills_outer_roles(self):
        from evaluate_candidate_roles import ConditionalGuardResidual
        policy = ConditionalGuardResidual()
        candidates = np.zeros((6, 6, 3), dtype=np.float32)
        candidates[:, :, 0] = np.arange(6)
        candidates[:, :, 2] = 1.
        prior = candidates[np.arange(6), np.arange(6)]
        pursuit = np.full((6, 3), .75, dtype=np.float32)
        cost = np.arange(36).reshape(6, 6)
        packet = dict(motion=np.zeros((6, 6)))
        with patch.object(policy, 'candidate_controls', return_value=(prior, pursuit, np.zeros(6))), \
                patch.object(policy, 'pursuit_controls', return_value=(pursuit, np.zeros(6))), \
                patch.object(policy, 'candidates', return_value=(candidates, cost)):
            np.testing.assert_array_equal(policy.compose(packet, np.zeros(6)), prior)
            action = policy.compose(packet, np.array([1., 1., 0., 0., 0., 0.]))
            np.testing.assert_array_equal(action[:2], pursuit[:2])
            np.testing.assert_array_equal(np.sort(action[2:, 0]), [0, 1, 4, 5])

    def test_residual_zero_preserves_prior_without_central_state(self):
        from policy.role_residual import RoleResidualPolicy
        from policy.role_value import features
        seed_random(83)
        env = TADEnv(6, protocol='paper-parameters-v1')
        env.reset(4.)
        packet = public_packet(env)
        del packet['central']
        policy = RoleResidualPolicy()
        np.testing.assert_array_equal(policy.compose(packet, np.zeros(6)),
                                      CandidateRolePolicy().choose_action(packet)[0])
        x = features(packet, policy.candidate_controls(packet))
        self.assertEqual(x.shape, (6, 16))
        self.assertTrue(np.isfinite(x).all())

    def test_conditional_values_follow_vessel_permutations(self):
        from policy.role_residual import RoleResidualPolicy
        from policy.role_value import ConditionalRoleValue, compositions, features
        seed_random(84)
        env = TADEnv(6, protocol='paper-parameters-v1')
        env.reset(6.)
        packet = public_packet(env)
        policy = RoleResidualPolicy()
        x = torch.from_numpy(features(packet, policy.candidate_controls(packet)))[None]
        masks = torch.from_numpy(compositions(6))
        order = torch.tensor([3, 0, 5, 1, 4, 2])
        model = ConditionalRoleValue().eval()
        with torch.no_grad():
            torch.testing.assert_close(model(x, masks), model(x[:, order], masks[:, order]),
                                       atol=2e-6, rtol=0)

    def test_joint_selection_resolves_competing_candidate_choices(self):
        candidates = np.zeros((3, 3, 3), dtype=np.float32)
        candidates[:, :, 0] = [-1., 0., 1.]
        candidates[:, :, 2] = 1.
        costs = np.array([[1., 2., 40.], [1., 6., 40.], [20., 1., 2.]])
        with patch.object(CandidateRolePolicy, 'candidates', return_value=(candidates, costs)):
            joint, _ = CandidateRolePolicy().choose_action({})
            independent, _ = CandidateRolePolicy(coordinated=False).choose_action({})
        np.testing.assert_array_equal(joint[:, 0], [0., -1., 1.])
        np.testing.assert_array_equal(independent[:, 0], [-1., -1., 0.])

    def test_public_inputs_permutation_and_deployment_roundtrip(self):
        seed_random(73)
        env = TADEnv(6, protocol='paper-parameters-v1')
        env.reset(4.)
        packet = public_packet(env)
        packet.pop('central')
        policy = CandidateRolePolicy()
        expected, _ = policy.choose_action(packet)
        before = copy.deepcopy(packet)
        order = np.array([3, 0, 5, 1, 4, 2])
        shuffled, _ = policy.choose_action({k: v[order] for k, v in packet.items()})
        np.testing.assert_allclose(shuffled, expected[order], atol=1e-6, rtol=0)
        equal(self, before, packet)
        self.assertTrue((np.abs(expected[:, :2]) <= 1.).all())
        np.testing.assert_array_equal(expected[:, 2], np.ones(6))
        config = dict(environment=dict(protocol='paper-parameters-v1'),
                      candidate_control=dict(width=28., coordinated=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'policy.pth'
            torch.save(dict(format='candidate-role-deployment-v1', config=config), path)
            deployed = DeploymentPolicy(path)
            actual, _ = deployed.choose_action(packet)
            np.testing.assert_array_equal(actual, expected)
            physical = deployed.control(packet['obs'], packet['motion'], env.boids_actions)
            _, reference, _ = safety_controller(6).control(packet['motion'], expected, env.boids_actions)
            np.testing.assert_array_equal(physical, reference)
            np.testing.assert_array_equal(deployed.last_action, expected)

    def test_invalid_role_settings_and_public_states_are_rejected(self):
        for width in (0., -1., float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                CandidateRolePolicy(width=width)
        with self.assertRaises(ValueError):
            CandidateRolePolicy(coordinated='false')
        with self.assertRaises(ValueError):
            CandidateRolePolicy().choose_action(dict(obs=np.zeros((3, 18)), motion=np.full((3, 6), np.nan)))


class Pipeline(unittest.TestCase):
    def test_intervention_replay_expires_labels_and_anchored_training_resumes(self):
        first = dict(action=np.zeros((2, 3, 3)), reward=np.array([1., 2.]))
        second = dict(action=np.ones((2, 3, 3)), reward=np.array([3., 4.]))
        pool = retain_interventions(None, first, step=2, capacity=3, max_age=4)
        pool = retain_interventions(pool, second, step=4, capacity=3, max_age=4)
        np.testing.assert_array_equal(pool['reward'], [2., 3., 4.])
        pool = retain_interventions(pool, first, step=9, capacity=3, max_age=4)
        np.testing.assert_array_equal(pool['reward'], [1., 2.])
        with tempfile.TemporaryDirectory() as directory:
            c = small_config()
            c['interaction'].update(safety=False, proposal_kl_weight=10., auxiliary_capacity=8,
                                    auxiliary_max_age=8, deterministic_gate_proposals=True, gate_reference_weight=100.)
            c['training']['actor_start_steps'] = 5
            left, right = Path(directory)/'continuous', Path(directory)/'resumed'
            run_training(c, output=left)
            run_training(c, output=right, max_steps=9)
            run_training(c, output=right)
            whole = torch.load(left/'resume.pth', weights_only=False)
            resumed = torch.load(right/'resume.pth', weights_only=False)
            for key in ('agent', 'replay', 'auxiliary', 'step', 'simulated_steps'):
                equal(self, whole[key], resumed[key])
            self.assertEqual(whole['actor_updates'], 8)
            self.assertEqual(whole['critic_updates'], 11)

    def test_arm_selection_preserves_the_configured_long_horizon(self):
        config = small_config()
        config['interaction']['horizon_steps'] = 300
        for arm in ('full', 'same_info', 'no_peer', 'model_value'):
            self.assertEqual(arm_config(config, arm)['interaction']['horizon_steps'], 300)
        self.assertEqual(arm_config(config, 'short')['interaction']['horizon_steps'], 10)
        self.assertEqual(config['interaction']['horizon_steps'], 300)


    def test_time_objective_charges_every_unsuccessful_termination_the_full_deadline(self):
        from interaction_rollout import transition_reward
        reward = np.ones(3)
        self.assertIs(transition_reward(reward, 0., .2, 60., 0, 3), reward)
        for outcome in (1, 2, 3, 4):
            elapsed = 60. if outcome == 4 else 17.
            first = transition_reward(reward, 0., 10., 60., 0, 3, 'capped-time-v1')
            last = transition_reward(reward, 10., elapsed, 60., outcome, 3, 'capped-time-v1')
            np.testing.assert_allclose(first + last, -elapsed if outcome == 3 else -60.)
        from train_interaction import validate
        c = small_config(); c['interaction']['reward_objective'] = 'capped-time-v1'
        with self.assertRaisesRegex(ValueError, 'undiscounted task returns'): validate(c)
        c['rl']['GAMMA'] = 1.; c['interaction']['entropy_objective'] = 'task-return-v4'
        validate(c)

    def test_complete_real_return_initialization_resumes_across_the_first_policy_update(self):
        c=small_config('same_info')
        c['training'].update(gate_steps=516,joint_steps=4,warm_steps=512,critic_warmup_updates=5,
            critic_warmup_target='complete_real_state_return',replay_capacity=1024,
            eval_interval=520,checkpoint_interval=520,log_interval=520)
        c['rl'].update(actor_learning_rate=1e-5,temperature_learning_rate=1e-5)
        c['interaction']['bootstrap_estimator'] = 'mean'
        c['interaction']['entropy_objective'] = 'proposal-mean-v3'
        with tempfile.TemporaryDirectory() as directory:
            left,right=Path(directory)/'whole',Path(directory)/'resumed'
            run_training(c,output=left)
            run_training(c,output=right,stop_at_step=511)
            run_training(c,output=right,stop_at_step=512)
            run_training(c,output=right)
            whole=torch.load(left/'resume.pth',weights_only=False)
            resumed=torch.load(right/'resume.pth',weights_only=False)
            for key in ('agent','replay','critic_warmup_updates','critic_warmup_complete','step'):
                equal(self,whole[key],resumed[key])
            self.assertEqual(whole['critic_warmup_updates'],5)
            self.assertEqual(whole['agent']['actor_optimizer']['param_groups'][0]['lr'],1e-5)
            self.assertEqual(whole['agent']['alpha_optimizer']['param_groups'][0]['lr'],1e-5)
            self.assertIn('policy_cost',whole['replay']['arrays'])

    def setUp(self):
        torch.set_num_threads(1)
        seed_random(42)

    def test_cold_cbf_does_not_consume_training_rng(self):
        safety_controller.cache_clear()
        state = RandomState.capture()
        safety_controller(3)
        actual = np.random.rand(5), torch.randn(5)
        state.restore()
        equal(self, actual, (np.random.rand(5), torch.randn(5)))

    def test_signal_zero_swap_root_and_history_isolation(self):
        from interaction_review import signal_state_checks
        from interaction_rollout import compact_snapshot, FrozenPolicy, frozen_payload
        env = TADEnv(3, protocol='paper-parameters-v1')
        env.reset(2.25)
        snapshot = compact_snapshot(env)
        policy = FrozenPolicy(frozen_payload(make_agent(small_config())))
        row = signal_state_checks(policy, snapshot, public_packet(env), 390000001, 1, horizon=3)
        self.assertEqual(row['zero_difference'], 0.)
        self.assertEqual(row['swap_residual'], 0.)
        self.assertTrue(row['snapshot_and_rng_unchanged'])
        self.assertGreater(row['nominal_thrust_difference_N'], 0.)
        self.assertGreater(row['executed_thrust_difference_N'], 0.)

    def test_absolute_probe_preserves_training_and_captures_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            result = run_training(small_config(), output=path, stop_at_step=4)
            self.assertEqual(result['step'], 4)
            probe = torch.load(path/'probe-4.pth', weights_only=True)
            self.assertIn('target', probe)
            self.assertGreater(probe['learning_checks']['adapter_gradient_norm'], 0.)
            self.assertGreater(probe['learning_checks']['critic_gradient_norm'], 0.)
            with self.assertRaisesRegex(ValueError, 'absolute development checkpoint'):
                run_training(small_config(), output=path, stop_at_step=3)
            resumed = run_training(small_config(), output=path, stop_at_step=6)
            self.assertEqual(resumed['step'], 6)
            self.assertTrue((path/'probe-4.pth').exists())
            self.assertTrue((path/'probe-6.pth').exists())

    def test_model_holdout_uses_lagged_critic_and_environment_has_no_bootstrap(self):
        import interaction_review as review
        from interaction_rollout import compact_snapshot, FrozenPolicy, frozen_payload
        policy=FrozenPolicy(frozen_payload(make_agent(small_config()),online_critic=True))
        class CountingCritic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls=0
            def forward(self,packet,action):
                self.calls+=1
                value=torch.zeros((len(action),1))
                return value,value
        target=CountingCritic()
        policy.label_critic=target
        env=TADEnv(3,protocol='paper-parameters-v1')
        env.reset(2.25)
        snapshot,packet=compact_snapshot(env),public_packet(env)
        online=policy.critic
        with patch.object(review,'_DIAGNOSTIC_POLICY',policy,create=True), \
             patch.object(review,'_DIAGNOSTIC_SOURCE',None,create=True), \
             patch.object(review,'diagnostic_state',return_value=(snapshot,packet,dict(state=0,prefix_steps=0))):
            review._calibration_worker((392000000,0,2,1))
            self.assertEqual(target.calls,2)
            review._calibration_worker((392100000,0,300,1))
            self.assertEqual(target.calls,2)
        self.assertIs(policy.critic,online)

    def test_development_value_error_matches_the_formal_minimum_head_estimate(self):
        import interaction_review as review
        from interaction_rollout import compact_snapshot, FrozenPolicy, frozen_payload
        policy = FrozenPolicy(frozen_payload(make_agent(small_config()), online_critic=True))
        env = TADEnv(3, protocol='paper-parameters-v1')
        env.reset(2.25)
        snapshot, packet = compact_snapshot(env), public_packet(env)
        heads = [(torch.tensor([[0.]]), torch.tensor([[10.]])),
                 (torch.tensor([[9.]]), torch.tensor([[0.]]))]
        traces = [[dict(thrust=np.zeros((3, 2)))]] * 2
        with patch.object(review, '_DIAGNOSTIC_POLICY', policy, create=True), \
             patch.object(review, '_DIAGNOSTIC_SOURCE', None, create=True), \
             patch.object(review, 'diagnostic_state', return_value=(snapshot, packet, {})), \
             patch.object(policy.critic, 'forward', side_effect=heads), \
             patch.object(review, 'paired_consequences', return_value=(0., 2, traces, None)):
            row = review._calibration_worker((392100000, 0, 300, 2))
        self.assertEqual(row['critic_predictions'], [-9., 10.])
        self.assertEqual(row['predicted_difference'], 0.)
        self.assertEqual(row['absolute_error'], 0.)
        self.assertEqual(review.precision_summary([row])['E_env'], 0.)

    def test_bootstrap_calibration_uses_the_exact_soft_tail_and_one_terminal_branch(self):
        import interaction_review as review
        from interaction_rollout import compact_snapshot, FrozenPolicy, frozen_payload
        class ConstantCritic(torch.nn.Module):
            def __init__(self,value):
                super().__init__()
                self.value=value
            def forward(self,packet,action):
                value=torch.full((len(action),1),self.value)
                return value-1.,value+1.
        before=FrozenPolicy(frozen_payload(make_agent(small_config()),online_critic=True))
        after=copy.deepcopy(before)
        after.bootstrap_estimator='mean'
        before.label_critic=ConstantCritic(0.)
        # Q_100 excludes its own entropy, includes entropy at step 101.
        after.label_critic=ConstantCritic(2.+.99*(2.-.2))
        env=TADEnv(3,protocol='paper-parameters-v1');env.reset(2.25)
        snapshot,packet=compact_snapshot(env),public_packet(env)
        calls=[]
        def consequence(policy,snapshot,action,*args,**kwargs):
            length=102 if not calls else 50
            calls.append(length)
            trace=[dict(packet=packet,action=action,reward=2.,logp=0. if k==0 else 1.,
                outcome=3 if k==length-1 else 0) for k in range(length)]
            value=sum(.99**k*(t['reward']-.2*t['logp']) for k,t in enumerate(trace))
            return value,length,trace
        with patch.object(review,'_BOOTSTRAP_POLICIES',dict(before=before,repaired=after),create=True), \
             patch.object(review,'_BOOTSTRAP_SOURCE',None,create=True), \
             patch.object(review,'diagnostic_state',return_value=(snapshot,packet,dict(prefix_steps=0))), \
             patch('interaction_rollout.branch_return',side_effect=consequence):
            row=review._bootstrap_check_worker((400000000,0,1))
        self.assertEqual(calls,[102,50])
        self.assertGreater(row['variants']['before']['label_errors']['100'][0],1.)
        self.assertAlmostEqual(row['variants']['repaired']['label_errors']['100'][0],0.,places=5)
        self.assertAlmostEqual(row['variants']['repaired']['tail_errors'][0],0.,places=5)

    def test_reconditioning_is_transactional_and_preserves_actor_replay_and_training_rng(self):
        from train_interaction import recondition_bootstrap
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'training/seed-42/full'
            c=small_config()
            c['interaction']['bootstrap_source']='coupled'
            run_training(c,output=path,stop_at_step=4)
            raw=(path/'resume.pth').read_bytes()
            old=torch.load(path/'resume.pth',weights_only=False)
            metric=dict(horizon_100_label_mae=10.,tail_value_mae=10.,environment_mae=10.,zero_prediction_mae=10.)
            calibration=dict(summary=dict(before=dict(all=copy.deepcopy(metric)),repaired=dict(all=copy.deepcopy(metric))))
            with patch('interaction_review.source_for_seed',return_value='unused'), \
                 patch('interaction_review.bootstrap_calibration',return_value=calibration):
                with self.assertRaisesRegex(RuntimeError,'failed heldout calibration'):
                    recondition_bootstrap(path,updates=2,workers=0)
            self.assertEqual((path/'resume.pth').read_bytes(),raw)
            calibration=copy.deepcopy(calibration)
            calibration['summary']['repaired']['all'].update(horizon_100_label_mae=1.,tail_value_mae=1.)
            with patch('interaction_review.source_for_seed',return_value='unused'), \
                 patch('interaction_review.bootstrap_calibration',return_value=calibration):
                recondition_bootstrap(path,updates=2,workers=0)
            new=torch.load(path/'resume.pth',weights_only=False)
            for key in ('replay','step','done','simulated_steps'):
                equal(self,old[key],new[key])
            equal(self,old['agent']['actor'],new['agent']['actor'])
            equal(self,old['agent']['actor_optimizer'],new['agent']['actor_optimizer'])
            for key in ('numpy','python','torch_cpu','torch_cuda'):
                equal(self,getattr(old['random'],key),getattr(new['random'],key))
            self.assertIsNone(new['auxiliary'])
            self.assertEqual(new['bootstrap_updates'],2)
            self.assertIn('bootstrap_critic',new['agent'])
            c['interaction']['bootstrap_source']='real_td'
            run_training(c,output=path,stop_at_step=6)

    def test_four_branch_and_configuration_interventions_hold_candidates_and_noise(self):
        import interaction_review as review
        from interaction_rollout import compact_snapshot, FrozenPolicy, frozen_payload
        policy=FrozenPolicy(frozen_payload(make_agent(small_config()),online_critic=True))
        env=TADEnv(3,protocol='paper-parameters-v1');env.reset(2.25)
        snapshot,packet=compact_snapshot(env),public_packet(env)
        calls=[]
        def consequence(policy,snapshot,a,future,noise,horizon,trace=False):
            calls.append((a.copy(),future,noise,horizon))
            return float(a[0,2]*a[1,2]),1,[dict(action=a.copy(),thrust=a[:,:2],outcome=3)]
        with patch.object(review,'_DIAGNOSTIC_POLICY',policy,create=True), \
             patch.object(review,'_DIAGNOSTIC_SOURCE',None,create=True), \
             patch.object(review,'diagnostic_state',return_value=(snapshot,packet,dict(state=0,prefix_steps=0))), \
             patch('interaction_rollout.branch_return',side_effect=consequence):
            row=review._mechanism_worker((530000000,0))
        self.assertEqual(len(calls),24)
        base=np.asarray(row['configurations']['00'])
        for action,*_ in calls: np.testing.assert_array_equal(action[:,:2],base[:,:2])
        for start in range(0,24,6):
            self.assertEqual(len({tuple(c[1:]) for c in calls[start:start+6]}),1)
        self.assertAlmostEqual(row['contrasts']['conditional_interaction']['mean'],float(base[0,2]*base[1,2]))
        np.testing.assert_allclose(np.asarray(row['configurations']['mean'])[:,2],base[:,2].mean())
        np.testing.assert_array_equal(np.sort(np.asarray(row['configurations']['permuted'])[:,2]),np.sort(base[:,2]))

    def test_exact_development_isolation_preserves_values_and_failed_attempts(self):
        from train_interaction import recondition_bootstrap
        for arm in ('no_peer','model_value'):
            with self.subTest(arm=arm), tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/f'training/seed-42/{arm}'
                c=small_config(arm);c['interaction']['bootstrap_source']='coupled'
                run_training(c,output=path,stop_at_step=4)
                old=torch.load(path/'resume.pth',weights_only=False)
                metric=dict(horizon_100_label_mae=10.,tail_value_mae=10.,environment_mae=10.,zero_prediction_mae=10.)
                calibration=dict(summary=dict(before=dict(all=copy.deepcopy(metric)),repaired=dict(all=copy.deepcopy(metric))))
                record=Path(directory)/f'reviews/bootstrap-repair/42-{arm}/completed.json'
                record.parent.mkdir(parents=True)
                record.write_text(json.dumps(dict(complete=False,passed=False,bootstrap_updates=10000)))
                with patch('interaction_review.source_for_seed',return_value='unused'), \
                     patch('interaction_review.bootstrap_calibration',return_value=calibration):
                    recondition_bootstrap(path,workers=0,isolate=True)
                new=torch.load(path/'resume.pth',weights_only=False)
                for key in ('replay','step','done','simulated_steps'):
                    equal(self,old[key],new[key])
                for key in ('actor','critic','target','actor_optimizer','critic_optimizer','alpha_optimizer','log_alpha'):
                    equal(self,old['agent'][key],new['agent'][key])
                equal(self,old['agent']['critic'],new['agent']['bootstrap_critic'])
                equal(self,old['agent']['critic_optimizer'],new['agent']['bootstrap_optimizer'])
                for key in ('numpy','python','torch_cpu','torch_cuda'):
                    equal(self,getattr(old['random'],key),getattr(new['random'],key))
                self.assertEqual(new['bootstrap_updates'],0)
                self.assertEqual(new['bootstrap_repair']['prior_attempts'][0]['bootstrap_updates'],10000)
                self.assertTrue(new['bootstrap_repair']['passed'])
                c['interaction']['bootstrap_source']='real_td'
                run_training(c,output=path,stop_at_step=6)

    def test_composed_revision_warmup_and_repeated_labels_resume_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            a,b=Path(directory)/'continuous',Path(directory)/'resumed'
            c=small_config()
            c['interaction'].update(critic_control_coordinates='nominal-thrust-v2',label_repetitions=3)
            c['training']['critic_warmup_updates']=3
            run_training(c,output=a)
            run_training(c,output=b,stop_at_step=1)
            run_training(c,output=b,stop_at_step=4)
            run_training(c,output=b)
            left=torch.load(a/'resume.pth',weights_only=False)
            right=torch.load(b/'resume.pth',weights_only=False)
            self.assertEqual(left['critic_warmup_updates'],3)
            self.assertEqual(left['bootstrap_updates'],14)
            for key in ('agent','replay','step','simulated_steps','auxiliary','critic_warmup_updates'):
                equal(self,left[key],right[key])
            self.assertIn('difference_standard_error',left['auxiliary'])
            from policy.interaction_sac import InteractionSAC
            interrupted=Path(directory)/'interrupted'
            original=InteractionSAC.learn_bootstrap;calls=[]
            def interrupt_once(agent,replay):
                calls.append(1)
                if len(calls)==2: raise KeyboardInterrupt
                return original(agent,replay)
            with patch.object(InteractionSAC,'learn_bootstrap',interrupt_once):
                with self.assertRaises(KeyboardInterrupt): run_training(c,output=interrupted)
            partial=torch.load(interrupted/'resume.pth',weights_only=False)
            self.assertEqual(partial['critic_warmup_updates'],1)
            self.assertFalse(partial['critic_warmup_complete'])
            run_training(c,output=interrupted)
            recovered=torch.load(interrupted/'resume.pth',weights_only=False)
            for key in ('agent','replay','step','simulated_steps','auxiliary','critic_warmup_updates'):
                equal(self,left[key],recovered[key])

    def test_same_info_critics_and_optimizers_remain_independent_after_warmup_and_resume(self):
        devices=['cpu']+(['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device), tempfile.TemporaryDirectory() as directory:
                c=small_config('same_info')
                c['interaction']['critic_control_coordinates']='nominal-thrust-v2'
                c['training']['critic_warmup_updates']=3
                path=Path(directory)
                run_training(c,output=path,device=device,stop_after_stage='gate')
                for resume in (False,True):
                    if resume: run_training(c,output=path,device=device)
                    saved=torch.load(path/'resume.pth',map_location='cpu',weights_only=False)
                    a=saved['agent']
                    equal(self,a['critic'],a['bootstrap_critic'])
                    equal(self,a['critic_optimizer'],a['bootstrap_optimizer'])
                    left,right=(a[k]['state'] for k in ('critic_optimizer','bootstrap_optimizer'))
                    for key in left:
                        for field in ('step','exp_avg','exp_avg_sq'):
                            self.assertNotEqual(left[key][field].data_ptr(),right[key][field].data_ptr())
                    self.assertEqual(saved['learning_checks']['bootstrap_optimizer_aliases'],0)
                    self.assertEqual(saved['learning_checks']['same_info_critic_max_difference'],0.)

    def test_lossless_replay_ring_and_reset_boundaries(self):
        env = TADEnv(3, protocol='paper-parameters-v1')
        env.reset(2.25)
        packet = public_packet(env)
        for capacity in (4, 20):
            replay = JointReplay(capacity)
            for i in range(8):
                following = {k: v + i for k, v in packet.items()}
                replay.store(packet, np.zeros((3,3)), np.zeros((3,2)), np.ones(3), following, i%3==0)
                packet = following if i%3 else {k: v*2 for k,v in following.items()}
            restored = JointReplay(1)
            restored.load_state_dict(replay.state_dict())
            for key in replay.arrays:
                equal(self, replay.arrays[key][:replay.size], restored.arrays[key][:restored.size])

    def test_exact_resume_including_stage_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory)/'continuous', Path(directory)/'resumed'
            c = small_config()
            run_training(c, output=a)
            run_training(c, output=b, max_steps=3)
            run_training(c, output=b, stop_after_stage='gate')
            gate = torch.load(b/'gate-endpoint.pth', weights_only=True)
            self.assertEqual((gate['step'],gate['stage']), (6,'gate'))
            again = run_training(c, output=b, stop_after_stage='gate')
            self.assertEqual(again['step'], 6)
            before = (b/'gate-endpoint.pth').read_bytes()
            safety_controller.cache_clear()
            run_training(c, output=b)
            self.assertEqual((b/'gate-endpoint.pth').read_bytes(), before)
            with self.assertRaises(ValueError):
                run_training(c, output=b, stop_after_stage='gate')
            left = torch.load(a/'resume.pth', weights_only=False)
            right = torch.load(b/'resume.pth', weights_only=False)
            for key in ('agent', 'replay', 'auxiliary', 'step', 'done', 'simulated_steps'):
                equal(self, left[key], right[key])
            for key in ('numpy','python','torch_cpu','torch_cuda'):
                equal(self, getattr(left['random'],key), getattr(right['random'],key))

    def test_review_detects_changed_frozen_proposals(self):
        agent = make_agent(small_config())
        source = {k:v.clone() for k,v in agent.actor.base.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            export_policy(agent,directory,6,endpoint=True)
            saved = torch.load(Path(directory)/'gate-endpoint.pth',weights_only=True)
            self.assertTrue(check_endpoint(saved,source,'gate',6)['frozen_proposals_equal'])
            saved['actor']['base.mean_layer.bias'][0] += .01
            with self.assertRaisesRegex(ValueError,'frozen proposal'):
                check_endpoint(saved,source,'gate',6)

    def test_validation_calibration_is_finite_and_isolates_rng(self):
        from interaction_rollout import frozen_payload
        agent = make_agent(small_config())
        state = RandomState.capture()
        rows = validation_calibration(frozen_payload(agent,online_critic=True),agent,42,states=2,repetitions=2)
        actual = np.random.rand(4),torch.randn(4)
        state.restore()
        equal(self,actual,(np.random.rand(4),torch.randn(4)))
        self.assertEqual([r['scene_seed'] for r in rows],[330000000,330000001])
        for row in rows:
            self.assertTrue(all(np.isfinite(value) for value in row.values()))
            self.assertGreater(row['simulated_steps'],0)

    def test_candidate_response_is_controlled_reproducible_and_isolates_rng(self):
        from interaction_rollout import frozen_payload
        agent = make_agent(small_config('no_peer'))
        torch.nn.init.normal_(agent.actor.relation_gate.weight, std=.1)
        state = RandomState.capture()
        payload = frozen_payload(agent, online_critic=True)
        result = validation_candidate_response(payload, agent, 42, states=2, resamples=2)
        actual = np.random.rand(4), torch.randn(4)
        state.restore()
        equal(self, actual, (np.random.rand(4), torch.randn(4)))
        again = validation_candidate_response(payload, agent, 42, states=2, resamples=2)
        self.assertEqual(result, again)
        self.assertEqual([r['scene_seed'] for r in result['states']], [350000000, 350000001])
        self.assertEqual(result['cbf_evaluations'], 16)
        self.assertGreater(result['simulated_steps'], 0)
        self.assertGreater(result['summary']['mean_peer_candidate_change'], 0.)
        for key in ('mean_abs_gate_change', 'mean_nominal_thrust_change_N',
                    'mean_gate_only_executed_thrust_change_N', 'mean_gate_effect_given_changed_peer_N'):
            self.assertEqual(result['summary'][key], 0.)
        visible = copy.deepcopy(payload)
        visible['config']['interaction'].update(arm='full', peer_candidates=True)
        full = validation_candidate_response(visible, agent, 42, states=2, resamples=2)
        self.assertGreater(full['summary']['mean_abs_gate_change'], 0.)
        for masked_state, full_state in zip(result['states'], full['states']):
            for masked_row, full_row in zip(masked_state['perturbations'], full_state['perturbations']):
                for key in ('own_candidate', 'peer_candidate_before', 'peer_candidate_after', 'gate_noise'):
                    self.assertEqual(masked_row[key], full_row[key])

    def test_no_peer_training_resumes_across_the_gate_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            c = small_config('no_peer')
            run_training(c, output=directory, stop_after_stage='gate')
            gate = torch.load(Path(directory)/'gate-endpoint.pth', weights_only=True)
            self.assertFalse(gate['config']['interaction']['peer_candidates'])
            result = run_training(c, output=directory)
            self.assertTrue(result['complete'])
            self.assertEqual(result['step'], 12)
            self.assertFalse(DeploymentPolicy(Path(directory)/'policy.pth').actor.peer_candidates)

    def test_initial_screen_runs_before_endpoints_and_aggregate_reuses_arm_results(self):
        from interaction_review import digest, write_json, CORE_ARMS, METRICS
        with tempfile.TemporaryDirectory() as directory:
            study=Path(directory)
            source=study/'source.pth'
            source.write_bytes(b'pretraining fixture')
            write_json(study/'manifest.json',dict(inherited_pretraining={'pretrain-42':dict(path=str(source))}))
            row={metric:0. for metric in METRICS}
            history=[dict(gates=[.25,.5,.75])]
            with patch('interaction_evaluation.OriginalPolicy'), patch('torch.load',return_value={}), \
                 patch('train_interaction.episode',return_value=(row,history)) as evaluate:
                collect(study,'gate',42,'core','initial_cbf')
                self.assertEqual(evaluate.call_count,100)
                data=study/'reviews/data-gate-42'
                self.assertTrue((data/'arm-initial_cbf-completed.json').exists())
                before=(data/'initial_cbf.json').read_bytes()
                collect(study,'gate',42,'core','initial_cbf')
                self.assertEqual(evaluate.call_count,100)
                for arm in CORE_ARMS:
                    endpoint=study/f'training/seed-42/{arm}/gate-endpoint.pth'
                    endpoint.parent.mkdir(parents=True)
                    endpoint.write_bytes(arm.encode())
                    result=json.loads(before)
                    result['arm']=arm
                    result['input']['endpoint']=digest(endpoint)
                    result['checks']=dict(actor_parameters=1234)
                    write_json(data/f'{arm}.json',result)
                collect(study,'gate',42,'core')
                self.assertEqual(evaluate.call_count,100)
                self.assertEqual((data/'initial_cbf.json').read_bytes(),before)
                completion=json.loads((data/'core-completed.json').read_text())
                self.assertEqual(set(completion['artifacts']),set((*CORE_ARMS,'initial_cbf')))

    def test_attacker_environment_keeps_defender_team_reward(self):
        env = TADEnv(3, protocol='paper-parameters-v1', LearningSide='Att')
        env.reset(2.25)
        packet = public_packet(env)
        _, reward, _, _, info, _ = execute(env, packet, np.full((3,3), .5), attacker_action=np.zeros(2))
        self.assertEqual(reward.shape, (3,))
        equal(self, reward, sum(env.paper_reward_components()))
        self.assertTrue(np.isfinite(info['attacker_reward']))

    def test_parallel_labels_match_serial_and_value_control(self):
        c = small_config()
        c['interaction']['pairs_per_batch'] = 4
        agent = make_agent(c)
        env = TADEnv(3, protocol='paper-parameters-v1')
        env.reset(2.25)
        pool = SnapshotPool(2); pool.add(env)
        serial = InterventionSampler(0)
        parallel = InterventionSampler(2)
        try:
            a, ca = serial.generate(agent,pool,step=1000)
            b, cb = parallel.generate(agent,pool,step=1000)
            # Workers return their interleaved chunks in fixed worker order.
            order = [0,2,1,3]
            for key in a: np.testing.assert_allclose(a[key][order],b[key],rtol=1e-6,atol=1e-6)
            self.assertEqual(ca,cb)
            agent.config['interaction']['auxiliary'] = 'value'
            same, used = serial.generate(agent,pool,step=1000)
            equal(self, a, same)
            self.assertEqual(used,ca)
        finally:
            parallel.close()

    def test_vrx_and_numerical_physical_command_agree(self):
        sys.path.insert(0,str(ROOT/'vrx'))
        from interaction_controller import InteractionController
        agent = make_agent(small_config())
        env = TADEnv(3, protocol='paper-parameters-v1')
        env.reset(2.25)
        p = public_packet(env)
        with tempfile.TemporaryDirectory() as directory:
            export_policy(agent,directory,0)
            vrx = InteractionController(Path(directory)/'policy.pth')
            numerical = DeploymentPolicy(Path(directory)/'policy.pth')
            a = vrx.control(p['obs'],p['motion'],env.boids_actions)
            b = numerical.control(p['obs'],p['motion'],env.boids_actions)
            equal(self,a,b)
            equal(self,vrx.last_action,numerical.last_action)


class ConditionalPeerCreditTests(unittest.TestCase):
    def test_receiver_intervention_keeps_deployed_peers_and_common_future(self):
        import train_response_fusion as response

        class Actor:
            def fusion(self, packet, baseline):
                return baseline.new_full((*baseline.shape[:2], 2), .2)

        n, roots, repeats = 3, 2, 3
        data = dict(obs=torch.zeros(4,n,18), motion=torch.zeros(4,n,6),
            central=torch.zeros(4,30), intent_history=torch.zeros(4,8,8),
            baseline=torch.zeros(4,n,3))

        def rollout(actor, critic, source, packet, innovations, exploration, **kwargs):
            changes = exploration[:,0,:,0].reshape(roots,repeats,2*n,n)
            for receiver in range(n):
                for branch in (2*receiver, 2*receiver+1):
                    others = [j for j in range(n) if j != receiver]
                    self.assertEqual(torch.count_nonzero(changes[:,:,branch,others]),0)
                torch.testing.assert_close(changes[:,:,2*receiver,receiver],
                    -changes[:,:,2*receiver+1,receiver], rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(exploration[:,1:]),0)
            noise = innovations.reshape(roots,repeats,2*n,300,-1)
            for branch in range(1,2*n):
                torch.testing.assert_close(noise[:,:,0],noise[:,:,branch],rtol=0,atol=0)
            self.assertFalse(torch.equal(noise[:,0,0],noise[:,1,0]))
            count=len(packet['obs'])
            return dict(capturer=torch.zeros(count,dtype=torch.long)),torch.arange(count).float()

        with patch.object(response,'score_intervention_rollout',rollout):
            result=response.paired_factor_credits(Actor(),None,data,np.random.default_rng(5),
                roots,repeats,.2,10,mean_peers=True)
        difference=result['returns'][:,:,result['positive']]-result['returns'][:,:,result['negative']]
        torch.testing.assert_close(difference,torch.full((roots,repeats,n),-1.),rtol=0,atol=0)

    def test_conditional_credit_slope_indices_match_analytic_quadratic(self):
        # Both endpoints contribute a derivative of the same receiver. Mixing
        # up branch and receiver dimensions silently changes this estimator.
        n=3
        samples=torch.randn(2,4,2*n,n,requires_grad=True)
        coefficients=torch.tensor([1.,2.,4.])
        value=(samples.square()*coefficients).sum(-1)
        slopes=torch.autograd.grad(value.sum(),samples)[0]
        positive=torch.arange(n)*2;negative=positive+1;receiver=torch.arange(n)
        estimate=.5*(slopes[:,:,positive,receiver]+slopes[:,:,negative,receiver])
        expected=coefficients*(samples[:,:,positive,receiver]+samples[:,:,negative,receiver])
        torch.testing.assert_close(estimate,expected)

    def test_candidate_permutation_preserves_receiver_and_physical_inputs(self):
        import train_response_fusion as response
        b,n=2,4
        baseline=torch.arange(b*n*3,dtype=torch.float32).reshape(b,n,3)/50.
        packet=dict(obs=torch.arange(b*n*20,dtype=torch.float32).reshape(b,n,20),
            motion=torch.randn(b,n,6),central=torch.randn(b,37))
        original={k:v.clone() for k,v in packet.items()};base_original=baseline.clone()
        test=self

        class Probe:
            candidate_probe='permuted'
            def adapted_weights(self,altered,base):
                for scene in range(b):
                    for receiver in range(n):
                        row=scene*n+receiver
                        peers=torch.tensor([j for j in range(n) if j!=receiver])
                        torch.testing.assert_close(base[row,receiver],baseline[scene,receiver])
                        torch.testing.assert_close(altered['obs'][row,receiver],packet['obs'][scene,receiver])
                        torch.testing.assert_close(altered['motion'][row],packet['motion'][scene])
                        torch.testing.assert_close(altered['obs'][row,:,:12],packet['obs'][scene,:,:12])
                        torch.testing.assert_close(base[row,peers,:2],baseline[scene,peers.roll(1),:2])
                        test.assertEqual(base[row].shape,baseline[scene].shape)
                return base[...,2]

        correction=response.CooperativeFactorModel.fusion(Probe(),packet,baseline)
        torch.testing.assert_close(correction,torch.zeros(b,n,2))
        for key in packet:torch.testing.assert_close(packet[key],original[key])
        torch.testing.assert_close(baseline,base_original)


class FrozenCompositionIdentificationTests(unittest.TestCase):
    def setUp(self):
        import identify_fusion_mechanisms as identification
        self.identification=identification
        seed_random(784102)
        self.env=TADEnv(5,protocol='paper-parameters-v1')
        self.env.reset(2.25)
        self.packet=public_packet(self.env)
        self.baseline=np.column_stack((self.packet['obs'][:,12:14],np.full(5,.4))).astype(np.float32)
        self.current=np.full((5,2),.1,dtype=np.float32)

    def features(self,change,packet=None):
        return self.identification.public_features(self.packet if packet is None else packet,
            self.baseline,self.current,np.asarray(change,dtype=np.float32),self.env.boids_actions)

    def test_equal_current_candidates_preserve_instantaneous_controls_but_retain_option(self):
        zero=self.features(np.zeros(5))
        changed=self.features([.4,-.4,.2,0.,0.])
        inputs=self.identification.stack_features([zero,changed])
        columns=[i for i in range(inputs['node'].shape[-1]) if i!=20]
        torch.testing.assert_close(inputs['node'][0,:,columns],inputs['node'][1,:,columns],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(inputs['nominal'][0],inputs['nominal'][1],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(inputs['pair'][0],inputs['pair'][1],rtol=1e-5,atol=1e-6)
        self.assertGreater(float(inputs['node'][1,:,20].abs().sum()),0.)

    def test_public_response_does_not_read_private_dynamics_or_opponent_agility(self):
        altered=copy.deepcopy(self.packet)
        altered['central'][:-1]=np.arange(len(altered['central'])-1)*100.
        a=self.features(np.zeros(5));b=self.features(np.zeros(5),altered)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_value_residual_is_exactly_anchored_when_no_control_changes(self):
        a=self.features(np.zeros(5));b=copy.deepcopy(a)
        # Change only the hypothesized executed response. At zero candidate
        # exchange it cannot produce an artificial intervention advantage.
        b['node'][:,15:17]+=.5;b['pair'][:,:,4:6]-=.2
        inputs=self.identification.stack_features([a,b])
        for representation in ('raw','response'):
            model=self.identification.ConditionalValue(representation=representation).eval()
            value=model(inputs)
            torch.testing.assert_close(value[0],value[1],rtol=0,atol=0)

    def test_no_peer_ablation_removes_candidate_response_information_from_receiver(self):
        a=self.features([.4,0.,0.,0.,0.])
        self.baseline[1,:2]=[.9,-.8]
        b=self.features([.4,0.,0.,0.,0.])
        inputs=self.identification.stack_features([a,b])
        for representation in ('no_peer','response'):
            model=self.identification.ConditionalValue(representation=representation).eval()
            captured=[]
            handle=model.composition.register_forward_hook(lambda module,args,result:captured.append(result.detach()))
            model(inputs);handle.remove()
            left,right=captured[0][:,0]
            if representation=='no_peer':torch.testing.assert_close(left,right,rtol=1e-5,atol=1e-6)
            else:self.assertGreater(float((left-right).abs().max()),1e-6)

    def test_contrast_precision_ignores_arbitrary_shared_future_shocks(self):
        rng=np.random.default_rng(7141)
        costs=rng.normal(size=(16,9))
        shifted=costs+50.*rng.normal(size=(16,1))
        a=self.identification.paired_contrast_precision(costs)
        b=self.identification.paired_contrast_precision(shifted)
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12)

    def test_event_projection_preserves_probability_and_actual_capped_cost(self):
        ident=self.identification
        costs=torch.tensor([[[0.,1.4,4.8],[5.,2.5,3.1]],[[0.,18.,30.],[29.,12.,.2]]])
        remaining=torch.tensor([5.,30.])
        probabilities=ident.event_labels(costs,remaining)
        support=torch.minimum(torch.tensor(ident.EVENT_TIMES)[None],remaining[:,None])
        torch.testing.assert_close(probabilities.sum(-1),torch.ones_like(costs))
        torch.testing.assert_close((probabilities*support[:,None,None]).sum(-1),costs)
        self.assertTrue(bool((probabilities>=0).all()))

    def test_event_value_is_anchored_and_control_gradient_reaches_head(self):
        ident=self.identification
        first=self.features(np.zeros(5));other=copy.deepcopy(first)
        other['node'][:,15:17]+=.5;other['pair'][:,:,4:6]-=.2
        model=ident.ConditionalEventValue().eval()
        # Check the invariant after the initially zero head has been learned.
        torch.nn.init.normal_(model.composition[-1].weight,std=.1)
        features=ident.stack_features([first,other])
        result=model.distribution(features)
        torch.testing.assert_close(result[0],result[1],rtol=0,atol=0)
        changed=ident.stack_features([self.features([.4,0.,0.,0.,0.])])
        model(changed).sum().backward()
        self.assertGreater(float(model.composition[-1].weight.grad.abs().sum()),0.)

    def test_explicit_candidate_noise_preserves_receiver_and_matches_source_at_zero(self):
        from types import SimpleNamespace
        from policy.networks import ActorAdap
        source=SimpleNamespace(actor=ActorAdap(6,8,3,16).eval())
        obs=self.packet['obs'];noise=np.zeros((5,2),dtype=np.float32)
        deterministic=source.actor(torch.as_tensor(obs),True,False)[0].detach().numpy()
        first=self.identification.sampled_candidates(source,obs,noise)
        np.testing.assert_array_equal(first,deterministic)
        noise[1:]=np.array([.7,-.4])
        second=self.identification.sampled_candidates(source,obs,noise)
        np.testing.assert_array_equal(first[0],second[0])
        self.assertGreater(float(np.max(abs(first[1:,:2]-second[1:,:2]))),1e-5)

    def test_actor_no_peer_path_is_invariant_to_peer_candidates_at_fixed_state(self):
        ident=self.identification
        first=self.features(np.zeros(5))
        self.baseline[1:,:2]=np.array([.9,-.8]);other=self.features(np.zeros(5))
        features=ident.stack_features([first,other])
        for peers in (False,True):
            actor=ident.CandidateGateAdapter(peer_candidates=peers)
            torch.nn.init.normal_(actor.gate[-1].weight,std=.1)
            logits=actor(features)[:,0]
            if peers: self.assertGreater(float((logits[0]-logits[1]).abs().max()),1e-6)
            else:torch.testing.assert_close(logits[0],logits[1],rtol=1e-5,atol=1e-6)

    def test_actor_update_receives_only_frozen_critic_control_differences(self):
        ident=self.identification
        actor=ident.CandidateGateAdapter();feature=self.features(np.zeros(5))
        data=dict(features=ident.stack_features([feature]*3),costs=torch.zeros(1,3),focal=np.array([[0,1]]))
        probability=ident.gate_probabilities(actor,data)
        q=torch.tensor([[2.,-3.,7.]],requires_grad=True)
        loss=(probability*q.detach()).sum();loss.backward()
        self.assertIsNone(q.grad)
        # A gradient step must increase the lower-cost branch's logit and
        # decrease the higher-cost branch's logit, independent of Q's offset.
        grad=actor.gate[-1].bias.grad
        self.assertLess(float(grad[1]),0.)
        self.assertGreater(float(grad[2]),0.)

    def test_scalar_vessel_matches_matrix_source_under_nonzero_currents(self):
        first=copy.deepcopy(self.env);second=copy.deepcopy(self.env)
        self.identification.use_source_current_motion(second)
        rng=np.random.default_rng(491833)
        for _ in range(80):
            for a,b in zip([first.attacker,*first.defender_list],[second.attacker,*second.defender_list]):
                thrust=rng.uniform(-800.,2000.,2);current=rng.normal(size=3)*[.3,.3,.02]
                a.step(thrust,current);b.step(thrust,current)
                np.testing.assert_allclose([a.x,a.y,a.theta,*a.velocity_r,*a.velocity],
                    [b.x,b.y,b.theta,*b.velocity_r,*b.velocity],rtol=1e-10,atol=1e-10)

    def test_additive_conditional_loss_cancels_state_value_gradient(self):
        ident=self.identification
        model=ident.ConditionalValue(bounded_link='additive',separate_state=True).double()
        features=ident.stack_features([self.features(np.zeros(5)),self.features([.4,0.,0.,0.,0.])])
        features={k:v.double() for k,v in features.items()}
        difference=model(features).diff().sum()
        (difference-2.).square().backward()
        for parameter in model.value.parameters():
            torch.testing.assert_close(parameter.grad,torch.zeros_like(parameter),rtol=0,atol=1e-12)
        for module in (model.state_node,model.state_edge,model.state_context):
            for parameter in module.parameters():
                torch.testing.assert_close(parameter.grad,torch.zeros_like(parameter),rtol=0,atol=1e-12)
        self.assertGreater(float(model.composition[-1].weight.grad.abs().sum()),0.)

    def test_explicit_value_interaction_vanishes_under_single_receiver_intervention(self):
        ident=self.identification
        model=ident.ConditionalValue(bounded_link='additive',pair_interaction=True).eval()
        with torch.no_grad():model.pair_composition[-1].bias[0]=2.
        features=ident.stack_features([self.features(x) for x in
            ([0.,0.,0.,0.,0.],[.4,0.,0.,0.,0.],[0.,.4,0.,0.,0.],[.4,.4,0.,0.,0.])])
        values=model(features)
        torch.testing.assert_close(values[:3],values[0].expand(3))
        self.assertAlmostEqual(float((values[3]-values[0]).detach()),12.*np.tanh(2./12.),places=5)

    def test_joint_gate_is_nonseparable_and_receiver_exchange_equivariant(self):
        ident=self.identification
        actor=ident.CandidateGateAdapter(joint_actor=True).eval()
        with torch.no_grad():actor.joint_gate[-1].bias.copy_(torch.tensor([3.,-3.,-3.,3.]))
        inputs=ident.stack_features([self.features(np.zeros(5))])
        forward=actor.joint_logits(inputs,[[0,1]])
        reverse=actor.joint_logits(inputs,[[1,0]])
        torch.testing.assert_close(forward,reverse[:,[0,3,4,1,2,5,7,6,8]])
        coordinates=torch.tensor([[0,0],[1,0],[2,0],[0,1],[0,2],[1,1],[1,2],[2,1],[2,2]])
        probability=forward.softmax(-1)[0];table=torch.zeros(3,3)
        table[coordinates[:,0],coordinates[:,1]]=probability
        self.assertGreater(float((table-table.sum(1)[:,None]*table.sum(0)[None]).abs().sum().detach()),.1)

    def test_value_adapter_selects_learned_joint_preference_in_focal_coordinates(self):
        ident=self.identification
        class Energy(torch.nn.Module):
            def forward(self,features):
                return 2.*features['node'][:,3,20]-3.*features['node'][:,1,20]
        policy=object.__new__(ident.ValueGateAdapterPolicy)
        policy.value=Energy();policy.options=9;policy.intervention_steps=10
        choices=policy.select_choices(self.packet,self.baseline,self.current,self.env.boids_actions,np.array([3,1]))
        np.testing.assert_array_equal(choices,np.array([0,2,0,1,0]))

    def test_option_duration_changes_persistence_without_changing_present_controls(self):
        ident=self.identification
        a=self.features([.4,0.,0.,0.,0.]);b=copy.deepcopy(a)
        a['option_remaining']=1.;b['option_remaining']=3.
        features=ident.stack_features([a,b])
        torch.testing.assert_close(features['node'][1,:,20],3.*features['node'][0,:,20])
        columns=[j for j in range(23) if j!=20]
        torch.testing.assert_close(features['node'][0,:,columns],features['node'][1,:,columns])
        torch.testing.assert_close(features['pair'][0],features['pair'][1])

    def test_zero_candidate_authority_still_selects_two_distinct_receivers(self):
        pair,score=self.identification.present_focal_pair(self.packet,self.baseline,self.current,self.env.boids_actions)
        self.assertNotEqual(int(pair[0]),int(pair[1]))
        self.assertEqual(score,0.)


class FleetConditionalControlTests(unittest.TestCase):
    def setUp(self):
        import identify_fusion_mechanisms as identification
        self.identification=identification
        seed_random(880241)
        env=TADEnv(6,protocol='paper-parameters-v1');obs,_=env.reset(2.25,noisy_agility=False)
        self.packet=public_packet(env,obs)
        rng=np.random.default_rng(889)
        self.base=np.column_stack((rng.uniform(-1,1,(6,2)),rng.uniform(0,1,6))).astype(np.float32)
        self.options=identification.fleet_options(6)

    def test_full_fleet_and_conditional_face_cover_distinct_control_configurations(self):
        ident=self.identification
        self.assertEqual(self.options.shape,(65,6))
        self.assertEqual(len(np.unique(self.options,axis=0)),65)
        face=ident.fleet_options(6,np.random.default_rng(82))[1:]
        changes=np.abs(face[:,None]-face[None]).sum(-1)
        np.testing.assert_array_equal((changes==1).sum(-1),np.full(8,3))
        # Actual coefficients are absolute, not a residual erased by clipping.
        np.testing.assert_array_equal(ident.fleet_action(self.base,np.ones(6))[:,2],np.ones(6))
        np.testing.assert_array_equal(ident.fleet_action(self.base,-np.ones(6)),self.base)

    def test_active_face_is_complete_and_keeps_other_feedback_at_reference(self):
        ident=self.identification;packet=copy.deepcopy(self.packet)
        packet['obs'][:,2]=np.array([16.,7.,22.,8.,19.,6.])
        options=ident.fleet_active_options(packet,3)
        np.testing.assert_array_equal(np.ptp(options[1:],axis=0),[0,1,0,1,0,1])
        np.testing.assert_array_equal(options[1:][:,[0,2,4]],np.ones((8,3)))
        self.assertEqual(len(np.unique(options[1:][:,[1,3,5]],axis=0)),8)
        features=dict(full=np.zeros((9,21)),no_peer=np.zeros((9,21)))
        features['full'][:,0]=np.maximum(options[:,1],0.)*np.maximum(options[:,3],0.)
        saved=dict(arm='difference',interaction_residual=True,coefficient=np.r_[np.zeros(21),1.,np.zeros(20)])
        q=ident.fleet_control_scores(features,saved,options,'permuted')
        self.assertEqual(q.shape,(9,));self.assertEqual(q[0],0.)

    def test_fleet_features_are_public_and_permutation_equivariant(self):
        ident=self.identification;features=ident.fleet_features(self.packet,self.base,self.options)
        changed=copy.deepcopy(self.packet);changed['central'][:-1]=1e6
        other=ident.fleet_features(changed,self.base,self.options)
        for key in features:np.testing.assert_array_equal(features[key],other[key])
        order=np.array([4,1,0,5,2,3]);changed['motion']=changed['motion'][order];changed['obs']=changed['obs'][order]
        other=ident.fleet_features(changed,self.base[order],self.options[:,order])
        for key in ('node','weight'):np.testing.assert_allclose(features[key][:,order],other[key],atol=1e-6)
        for key in ('edge','no_peer_edge'):np.testing.assert_allclose(features[key][:,order][:,:,order],other[key],atol=1e-6)

    def test_no_peer_value_has_zero_mixed_control_effect_but_full_can_interact(self):
        ident=self.identification
        # Four alternatives vary two receivers while all current states stay fixed.
        options=np.zeros((4,6),dtype=np.float32);options[1,0]=1;options[2,1]=1;options[3,:2]=1
        features={k:torch.as_tensor(v) for k,v in ident.fleet_features(self.packet,self.base,options).items()}
        torch.manual_seed(971)
        no=ident.FleetConditionalValue(no_peer=True);full=ident.FleetConditionalValue(no_peer=False)
        full.load_state_dict(no.state_dict())
        _,q=no(features);_,r=full(features)
        self.assertLess(abs(float((q[3]-q[2]-q[1]+q[0]).detach())),1e-6)
        self.assertGreater(abs(float((r[3]-r[2]-r[1]+r[0]).detach())),1e-7)

    def test_conditional_control_loss_does_not_update_state_only_baseline(self):
        ident=self.identification;torch.set_grad_enabled(True)
        model=ident.FleetConditionalValue()
        features={k:torch.as_tensor(v) for k,v in ident.fleet_features(self.packet,self.base,self.options[:9]).items()}
        _,q=model(features);(q[1:]-q[0]-2.).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in model.state.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.effect.parameters()),0.)

    def test_feedback_reference_preserves_source_controls_and_zero_speed_lead_is_boids(self):
        ident=self.identification
        from source_arboids import source_mixture
        obs=self.packet['obs'];rule=ident.fleet_lead_rule(obs)
        # At reset the target is stationary, so lead attraction equals Boids.
        np.testing.assert_allclose(rule,obs[:,12:14],atol=2e-6)
        reference=ident.fleet_action(self.base,-np.ones(6),obs,'feedback')
        raw=source_mixture(self.base,750.*obs[:,12:14]+250.)
        actual=source_mixture(reference,750.*obs[:,12:14]+250.)
        np.testing.assert_allclose(raw,actual,atol=2e-4)

    def test_rollout_value_features_ignore_private_state_and_leave_real_random_stream_untouched(self):
        ident=self.identification
        def source(obs):return np.tile(np.array([.6,.4,.3],dtype=np.float32),(len(obs),1))
        options=ident.fleet_options(6,np.random.default_rng(40))
        before=np.random.get_state()
        a=ident.fleet_rollout_features(self.packet,source,options,steps=2)
        after=np.random.get_state()
        self.assertEqual(before[0],after[0]);np.testing.assert_array_equal(before[1],after[1]);self.assertEqual(before[2:],after[2:])
        altered=copy.deepcopy(self.packet);altered['central'][:-1]=np.arange(len(altered['central'])-1)*700.
        b=ident.fleet_rollout_features(altered,source,options,steps=2)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_paired_covariance_is_invariant_to_common_future_nuisance(self):
        ident=self.identification;rng=np.random.default_rng(775)
        costs=[rng.normal(size=(8,9)) for _ in range(5)]
        changed=[row+rng.normal(size=(8,1))*100. for row in costs]
        precision=ident.fleet_contrast_precision(costs)
        np.testing.assert_allclose(precision,ident.fleet_contrast_precision(changed),atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(precision).min(),0.)

    def test_anchored_normal_equations_match_joint_loss_and_isolate_baseline(self):
        ident=self.identification;rng=np.random.default_rng(1472)
        dx=rng.normal(size=(4,5,3));roots=rng.normal(size=(4,2))
        values=rng.normal(size=(4,5));weights=np.array([2.,0.,1.,1.])
        contrast=dx[:,1:]-dx[:,:1];target=values[:,1:]-values[:,:1]
        gram=np.einsum('bki,bkj,b->ij',contrast,contrast,weights)
        rhs=np.einsum('bki,bk,b->i',contrast,target,weights)
        ag,ar,cg,cr=ident.fleet_anchor_equations(gram,rhs,dx,roots,values,weights,4)
        np.testing.assert_array_equal(cg[:2],np.zeros((2,5)))
        np.testing.assert_array_equal(cg[:,:2],np.zeros((5,2)))
        np.testing.assert_array_equal(cr[:2],np.zeros(2))
        theta=torch.tensor(rng.normal(size=5),dtype=torch.float64,requires_grad=True)
        prediction=torch.tensor(roots)@theta[:2]+torch.einsum('bki,i->bk',torch.tensor(dx),theta[2:]).T
        residual=prediction.T-torch.tensor(values)
        conditional=torch.einsum('bki,i->bk',torch.tensor(contrast),theta[2:])-torch.tensor(target)
        auxiliary=.7
        loss=((residual.square().sum(1)+auxiliary*5/4*conditional.square().sum(1))*torch.tensor(weights)).sum()/2
        loss.backward();expected=(ag+auxiliary*cg)@theta.detach().numpy()-ar-auxiliary*cr
        np.testing.assert_allclose(theta.grad.numpy(),expected,atol=1e-12)

    def test_frozen_interaction_surgery_preserves_unary_candidate_values(self):
        ident=self.identification;options=self.options
        unary=np.zeros((len(options),21));unary[:,0]=np.maximum(options,0.)@np.arange(1,7)
        joint=unary.copy();joint[:,1]=np.maximum(options[:,0],0.)*np.maximum(options[:,1],0.)
        coefficient=np.zeros(42);coefficient[0]=1.;coefficient[22]=11.
        saved=dict(arm='difference',interaction_residual=True,coefficient=coefficient)
        features=dict(full=joint,no_peer=unary)
        full=ident.fleet_control_scores(features,saved,options)
        zero=ident.fleet_control_scores(features,saved,options,'zero')
        permuted=ident.fleet_control_scores(features,saved,options,'permuted')
        np.testing.assert_array_equal(zero,unary[:,0])
        np.testing.assert_array_equal(full-zero,11.*joint[:,1])
        self.assertGreater(np.abs(permuted-full).max(),0.)
        self.assertEqual(permuted[0],full[0])

    def test_state_conditioned_coefficients_keep_no_peer_additive(self):
        ident=self.identification;rng=np.random.default_rng(933)
        options=np.maximum(self.options,0.);single=rng.normal(size=(6,21))
        unary=options@single
        features=dict(full=unary,no_peer=unary,root=rng.normal(size=8))
        saved=dict(arm='no_peer',interaction_residual=True,coefficient=rng.normal(size=42*8),
            context_normalization=dict(mean=[0.]*7,scale=[1.]*7))
        q=ident.fleet_control_scores(features,saved,self.options)
        # Four corners differ only in the first two vessel controls.
        self.assertAlmostEqual(float(q[4]-q[3]-q[2]+q[1]),0.,places=10)
        batched=ident.fleet_context_features(np.stack([unary,unary]),np.stack([features['root']]*2),saved['context_normalization'])
        one=ident.fleet_context_features(unary,features['root'],saved['context_normalization'])
        np.testing.assert_array_equal(batched[0],one)

    def test_nonlinear_continuation_is_applied_before_unary_decomposition(self):
        ident=self.identification;options=self.options
        control=1.-np.where(options<0.,1.,options)
        linear=np.zeros((len(options),21));linear[:,5]=control[:,0]+2.*control[:,1]
        features=dict(full=linear,no_peer=linear,root=np.zeros(8))
        result=ident.fleet_response_basis(features,options,dict(indices=[5],mean=[0.],scale=[1.]))
        full=result['full'];unary=result['no_peer']
        np.testing.assert_allclose(unary[4]-unary[3]-unary[2]+unary[1],0.,atol=1e-12)
        self.assertAlmostEqual(full[4,-1]-full[3,-1]-full[2,-1]+full[1,-1],4.)
        np.testing.assert_array_equal(full[0],unary[0])

    def test_monotone_physical_readout_respects_quadratic_optimum(self):
        ident=self.identification
        gram=np.eye(21);penalty=np.ones(21);rhs=np.zeros(21)
        rhs[[0,2,5,7,11]]=[4.,6.,-8.,10.,-12.]
        fitted=ident.fleet_ridge_solution(gram,rhs,penalty,True)
        expected=np.zeros(21);expected[[0,7,11]]=[2.,5.,-6.]
        np.testing.assert_allclose(fitted,expected,atol=1e-8)

    def test_single_coordination_window_returns_to_the_training_continuation(self):
        ident=self.identification;packet=copy.deepcopy(self.packet)
        policy=object.__new__(ident.FleetRolloutValuePolicy)
        policy.saved=dict(rollout_steps=1,candidate_mode='feedback')
        policy.source=lambda obs:self.base.copy();policy.period=2
        policy.decision_limit=1;policy.activation_radius=25.;policy.reset()
        packet['obs'][:,2]=40.
        np.testing.assert_array_equal(policy.choose_action(packet)[0],self.base)
        self.assertEqual(len(policy.decisions),0)
        packet['obs'][:2,2]=10.
        q=np.ones(len(self.options));q[1]=-1.
        with patch.object(ident,'fleet_rollout_features',return_value={}) as projection,patch.object(ident,'fleet_control_scores',return_value=q):
            first=policy.choose_action(packet)[0];policy.choose_action(packet)
            final=policy.choose_action(packet)[0]
            self.assertEqual(projection.call_count,1)
        self.assertEqual(len(policy.decisions),1);self.assertGreater(np.abs(first-self.base).max(),0.)
        np.testing.assert_array_equal(final,self.base)

    def test_synthetic_response_ensemble_preserves_rng_and_unary_additivity(self):
        ident=self.identification
        def source(obs):return np.tile(np.array([.6,.4,.3],dtype=np.float32),(len(obs),1))
        options=ident.fleet_options(6,np.random.default_rng(40))
        numpy_before=np.random.get_state();torch_before=torch.random.get_rng_state()
        features=ident.fleet_rollout_features(self.packet,source,options,steps=2,repeats=3)
        numpy_after=np.random.get_state()
        self.assertEqual(numpy_before[0],numpy_after[0])
        np.testing.assert_array_equal(numpy_before[1],numpy_after[1]);self.assertEqual(numpy_before[2:],numpy_after[2:])
        torch.testing.assert_close(torch_before,torch.random.get_rng_state(),rtol=0,atol=0)
        for array in features.values():self.assertTrue(np.isfinite(array).all())
        unary=features['no_peer']
        np.testing.assert_allclose(unary[4]-unary[3]-unary[2]+unary[1],0.,atol=2e-5)

    def test_measured_target_contract_reads_no_private_dynamics(self):
        ident=self.identification;seed_random(88771)
        env=TADEnv(6,protocol='paper-parameters-v1');obs,_=env.reset(2.25,noisy_agility=False)
        a=ident.fleet_public_packet(env,obs,True)
        env.attacker.velocity_r=np.array([19.,-22.,4.]);env.attacker.agility=9.
        b=ident.fleet_public_packet(env,obs,True)
        for key in ('motion','obs','target_motion'):np.testing.assert_array_equal(a[key],b[key])
        # The learned controller fails explicitly if its declared measurement
        # is absent, rather than substituting privileged state or zero yaw.
        policy=object.__new__(ident.FleetRolloutValuePolicy)
        policy.saved=dict(measured_target=True);policy.source=lambda x:self.base
        with self.assertRaisesRegex(ValueError,'target motion measurement'):
            policy.choose_action(self.packet)


class PersistentFusionProgramTests(unittest.TestCase):
    def setUp(self):
        import identify_fusion_mechanisms as ident
        self.ident=ident;seed_random(894723)
        env=TADEnv(3,protocol='paper-parameters-v1');obs,_=env.reset(2.25,noisy_agility=False)
        self.packet=ident.fleet_public_packet(env,obs,measured_target=True)
        self.base=np.array([[.8,-.5,.15],[-.2,.7,.6],[.3,.6,.9]],np.float32)

    def test_persistent_program_keeps_candidate_feedback_and_clips_gate_only(self):
        ident=self.ident;before=self.base.copy();z=np.array([-.2,.2,.2])
        a=ident.program_action(self.base,z,-.05)
        np.testing.assert_array_equal(a[:,:2],self.base[:,:2]);np.testing.assert_array_equal(before,self.base)
        np.testing.assert_allclose(a[:,2],[0.,.75,1.],atol=1e-7)
        after=self.base.copy();after[:,:2]*=-1.;after[:,2]=.5
        b=ident.program_action(after,z,-.05)
        np.testing.assert_array_equal(b[:,:2],after[:,:2]);np.testing.assert_allclose(b[:,2],[.25,.65,.65],atol=1e-7)

    def test_no_peer_is_additive_and_full_has_nonzero_mixed_control_response(self):
        ident=self.ident;options=ident.program_options(3,.2)
        features={k:torch.as_tensor(v) for k,v in ident.program_features(self.packet,self.base,options,-.05).items()}
        torch.manual_seed(218);model=ident.ProgramConditionalValue(32,no_peer=True)
        _,q=model(features);self.assertLess(float(abs(q[4]-q[3]-q[2]+q[1])),2e-7)
        torch.manual_seed(218);model=ident.ProgramConditionalValue(32,no_peer=False)
        _,q=model(features);self.assertGreater(float(abs(q[4]-q[3]-q[2]+q[1])),1e-8)
        for probe in ('aligned','state-only','permuted'):
            _,value=model({k:v.unsqueeze(0) for k,v in features.items()},probe)
            self.assertEqual(tuple(value.shape),(1,9));self.assertTrue(torch.isfinite(value).all())

    def test_program_representation_cannot_read_private_environment_state(self):
        ident=self.ident;opts=ident.program_options(3,.2)
        before=ident.program_features(self.packet,self.base,opts,-.05)
        other=copy.deepcopy(self.packet);other['central'][:-1]=-1e9
        after=ident.program_features(other,self.base,opts,-.05)
        for key in before:np.testing.assert_array_equal(before[key],after[key])

    def test_candidate_permutation_preserves_geometry_and_identical_candidates(self):
        ident=self.ident;packet=copy.deepcopy(self.packet)
        packet['obs'][:,12:14]=[.2,.4];base=np.tile([.7,-.3,.5],(3,1)).astype(np.float32)
        options=np.zeros((1,3),np.float32)
        features=ident.program_features(packet,base,options,-.05)
        np.testing.assert_allclose(features['edge'],features['permuted_edge'],atol=2e-6)
        features=ident.program_features(self.packet,self.base,ident.program_options(3,.2),-.05)
        np.testing.assert_array_equal(features['edge'][...,:14],features['permuted_edge'][...,:14])

    def test_kernel_is_positive_and_anchors_reference_with_additive_ablation(self):
        ident=self.ident;options=ident.program_options(3,.2,self.packet,2)
        features=ident.program_features(self.packet,self.base,options,-.05)
        description=ident.program_kernel_description(self.packet,features,options,dict(radius=.2))
        bank={k:torch.as_tensor(np.stack((v,v)),dtype=torch.float64) for k,v in description.items()}
        normalization=ident.program_kernel_normalization(bank)
        parts=ident.program_kernel_components(bank,bank,normalization,.7)
        kernel=parts['unary']+parts['pair'];flat=kernel.reshape(10,10)
        torch.testing.assert_close(flat,flat.T,atol=1e-10,rtol=0.)
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(flat).min()),-1e-8)
        torch.testing.assert_close(kernel[:,0],torch.zeros_like(kernel[:,0]))
        torch.testing.assert_close(parts['unary'][:,4]-parts['unary'][:,3]-parts['unary'][:,2]+parts['unary'][:,1],torch.zeros_like(parts['unary'][:,1]),atol=1e-10,rtol=0.)
        self.assertGreater(float((parts['pair'][:,4]-parts['pair'][:,3]-parts['pair'][:,2]+parts['pair'][:,1]).abs().max()),1.)

    def test_conditional_precision_cancels_common_future_nuisance(self):
        ident=self.ident;rng=np.random.default_rng(4281)
        a=rng.normal(size=(16,8,5));nuisance=rng.normal(size=(16,8,1))*20.
        before=[dict(costs=x) for x in a];after=[dict(costs=x) for x in a+nuisance]
        np.testing.assert_allclose(ident.program_kernel_covariance(before,True),ident.program_kernel_covariance(after,True),atol=1e-12)
        self.assertGreater(np.trace(ident.program_kernel_covariance(after,False)),np.trace(ident.program_kernel_covariance(before,False)))

    def test_policy_selects_once_at_training_trigger_and_recomputes_candidates(self):
        ident=self.ident;policy=object.__new__(ident.ProgramPolicy)
        policy.config=dict(bias=-.05,radius=.2,trigger=35.);policy.probe='aligned';policy.source=lambda x:self.base.copy();policy.reset()
        class FakeValue:
            calls=0
            def __call__(this,features,probe):
                this.calls+=1;q=torch.ones(9);q[2]=-1.;return q*0.,q
        policy.model=FakeValue();packet=copy.deepcopy(self.packet);packet['central'][-1]=1.-3./60.
        packet['obs'][:,2]=40.;first=policy.choose_action(packet)[0];self.assertEqual(policy.model.calls,0)
        np.testing.assert_array_equal(first,ident.program_action(self.base,np.zeros(3),-.05))
        packet['obs'][0,2]=30.;policy.choose_action(packet);self.base[:,0]*=-1.
        last=policy.choose_action(packet)[0];self.assertEqual(policy.model.calls,1);self.assertEqual(len(policy.decisions),1)
        np.testing.assert_array_equal(last,ident.program_action(self.base,ident.program_options(3,.2)[2],-.05))


class ReactiveCoordinationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(23);seed_random(23)
        env=TADEnv(4,protocol='paper-parameters-v1');env.reset(2.25,noisy_agility=False)
        self.packet={k:torch.as_tensor(v,dtype=torch.float32)[None] for k,v in public_packet(env).items()}
        self.base=torch.tensor([[[.2,.8,.6],[.7,-.4,.7],[-.1,.9,.4],[.9,.3,.8]]])
        self.factor_packet=dict(self.packet,target_heading=torch.tensor([env.attacker.theta],dtype=torch.float32),time_remaining=torch.ones(1))

    def test_attention_value_is_public_and_follows_vessel_permutations(self):
        from identify_fusion_mechanisms import ReactiveAttentionValue
        model=ReactiveAttentionValue(16)
        for layer in (model.unary[-1],model.contextual[-1],model.mixed[-1]):
            torch.nn.init.normal_(layer.weight,std=.15)
        inputs=model.inputs(self.factor_packet,self.base)
        private=copy.deepcopy(self.factor_packet);private['central'].normal_()
        for key,value in model.inputs(private,self.base).items():
            torch.testing.assert_close(value,inputs[key],rtol=0.,atol=0.)
        programs=torch.tensor([[[0.,0.,0.,0.],[.2,-.1,0.,0.],[0.,.1,-.2,0.]]])
        permutation=torch.tensor([2,0,3,1]);changed=copy.deepcopy(self.factor_packet)
        for key in ('obs','motion'):changed[key]=changed[key][:,permutation]
        permuted=model.inputs(changed,self.base[:,permutation])
        original=model(inputs,programs)[1]
        torch.testing.assert_close(model(permuted,programs[:,:,permutation])[1],original,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(original[:,0],torch.zeros(1),rtol=0.,atol=0.)

    def test_attention_peer_candidates_change_the_current_unary_choice_field(self):
        from identify_fusion_mechanisms import ReactiveAttentionValue
        model=ReactiveAttentionValue(16)
        torch.nn.init.normal_(model.contextual[-1].weight,std=.3)
        first=model.inputs(self.factor_packet,self.base)
        changed=self.base.clone();changed[:,1:,:2]*=-1.
        second=model.inputs(self.factor_packet,changed)
        torch.testing.assert_close(first['node'][:,0],second['node'][:,0],rtol=0.,atol=0.)
        before=model.factors(first)[1][:,0];after=model.factors(second)[1][:,0]
        self.assertGreater(float((before-after).abs().max()),1e-5)
        torch.testing.assert_close(model.factors(first,'interaction-off')[1][:,0],
            model.factors(second,'interaction-off')[1][:,0],rtol=0.,atol=0.)
        model.zero_grad();after.sum().backward()
        self.assertGreater(float(model.query.weight.grad.abs().sum()),1e-8)

    def test_attention_gate_executes_the_learned_joint_energy_minimum(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        settings=dict(bias=-.05,full_gate=True,control_radius=30.,prior=.3)
        actor=subject.ReactiveAttentionGate(settings,16,3,.5)
        for member in actor.members:
            for layer in (member.unary[-1],member.contextual[-1],member.mixed[-1]):
                torch.nn.init.normal_(layer.weight,std=.3)
        programs,_=subject.reactive_prior_options(packet,self.base,settings)
        features=subject.ReactiveAttentionValue.inputs(packet,self.base)
        values=torch.stack([member(features,programs)[1] for member in actor.members])
        score=values.mean(0)+.5*values.std(0,unbiased=False)+.3*programs.square().mean(-1)
        choice,control=actor.decide(packet,self.base,settings)
        self.assertEqual(int(choice[0]),int(score.argmin(-1)[0]))
        torch.testing.assert_close(control,programs[torch.arange(len(programs)),choice])
        self.assertLessEqual(int((control.abs()>1e-7).sum()),2)

    def test_attention_response_extension_preserves_the_actual_parent_function(self):
        from identify_fusion_mechanisms import ReactiveAttentionValue
        original=ReactiveAttentionValue(16);expanded=ReactiveAttentionValue(16,reaction=True)
        for layer in (original.unary[-1],original.contextual[-1],original.mixed[-1]):
            torch.nn.init.normal_(layer.weight,std=.3)
        inherited={}
        for key,value in original.state_dict().items():
            if key.endswith(('geometry.0.weight','mixed.0.weight')):value=torch.nn.functional.pad(value,(0,24))
            inherited[key]=value
        expanded.load_state_dict(inherited)
        programs=torch.tensor([[[0.,0.,0.,0.],[.2,-.1,0.,0.],[0.,.1,-.2,0.]]])
        first=original.inputs(self.factor_packet,self.base)
        second=expanded.inputs(self.factor_packet,self.base,reaction=True)
        torch.testing.assert_close(expanded(second,programs)[1],original(first,programs)[1],rtol=1e-5,atol=1e-6)
        private=copy.deepcopy(self.factor_packet);private['central'].normal_()
        torch.testing.assert_close(expanded.inputs(private,self.base,reaction=True)['reaction'],second['reaction'],rtol=0.,atol=0.)

    def test_attention_unary_anchor_preserves_the_independent_control(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        settings=dict(bias=-.05,full_gate=True,control_radius=30.,prior=.3)
        reference=torch.zeros(5,10,4);reference[:,3,0]=1.
        actor=subject.ReactiveAttentionGate(settings,16,3,2.,False,False,reference,object())
        for member in actor.members:
            for layer in (member.unary[-1],member.contextual[-1],member.mixed[-1]):
                torch.nn.init.normal_(layer.weight,std=2.)
        programs,parts=subject.reactive_prior_options(packet,self.base,settings)
        prior=torch.tensor([[[10.,8.,13.,11.,7.,14.,6.,9.,12.]]]).expand(-1,4,-1).clone()
        singles=subject.reactive_monotone_estimates(reference.double(),prior.double(),parts[:9],packet,False).mean((0,2))
        value=singles[:,parts].sum(-1)+.3*programs.square().mean(-1)
        with patch.object(subject,'reactive_model_prior',return_value=prior) as predictor:
            choice,program=actor.decide(packet,self.base,settings)
        self.assertEqual(predictor.call_args.args[2].shape[1],9)
        self.assertEqual(int(choice[0]),int(value.argmin(-1)[0]))
        torch.testing.assert_close(program,programs[torch.arange(len(programs)),value.argmin(-1)])

    def test_attention_panel_retains_closed_interaction_and_both_decisions(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        settings=dict(bias=-.05,full_gate=True,control_radius=30.,prior=.3)
        programs,parts=subject.reactive_prior_options(packet,self.base,settings)
        score=torch.linspace(1.,2.,33)[None];score[:,27]=-2.
        panel=subject.reactive_attention_panel(programs,parts,score,torch.tensor([27]),torch.tensor([3]))
        self.assertEqual(tuple(panel.shape),(1,7))
        face=programs[torch.arange(1)[:,None],panel[:,:4]]
        torch.testing.assert_close(face[:,0]+face[:,3],face[:,1]+face[:,2],rtol=0.,atol=1e-7)
        self.assertIn(27,panel[0].tolist());self.assertIn(3,panel[0].tolist())
        self.assertEqual(len(set(panel[0].tolist())),7)

    def test_attention_single_prior_reconstructs_the_exact_unary_values(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        _,parts=subject.reactive_prior_options(packet,self.base,dict(bias=-.05,full_gate=True))
        coefficients=torch.randn(3,10,4,dtype=torch.float64)
        prior=torch.randn(1,4,33,dtype=torch.float64)+12.
        expected=subject.reactive_monotone_estimates(coefficients,prior,parts,packet,False).mean((0,2))
        actual=subject.reactive_attention_unary(coefficients,prior[:,:,:9],parts,packet)
        torch.testing.assert_close(actual,expected,rtol=0.,atol=0.)

    def test_attention_linear_readout_matches_joint_field_and_writeback(self):
        import identify_fusion_mechanisms as subject
        torch.manual_seed(429)
        member=subject.ReactiveAttentionValue(16,reaction=True)
        for name,parameter in member.named_parameters():
            if name.startswith(('mixed.4.','contextual.4.')):torch.nn.init.normal_(parameter,std=.1)
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        inputs=member.inputs(packet,self.base,reaction=True)
        options,_=subject.reactive_prior_options(packet,self.base,dict(bias=-.05,full_gate=True))
        design,weights=subject.reactive_attention_readout(member,inputs,options)
        expected=member(inputs,options)[1]-member(inputs,options,'interaction-off')[1]
        torch.testing.assert_close(design@weights,expected,rtol=1e-5,atol=2e-6)
        changed=weights+torch.randn_like(weights)*.1
        subject.reactive_attention_set_readout(member,changed)
        actual=member(inputs,options)[1]-member(inputs,options,'interaction-off')[1]
        torch.testing.assert_close(design@changed,actual,rtol=1e-5,atol=2e-6)
        torch.testing.assert_close(design[:,0],torch.zeros_like(design[:,0]),rtol=0.,atol=0.)

    def test_paired_posterior_recovers_effect_and_preserves_common_return_shift(self):
        import identify_fusion_mechanisms as subject
        torch.manual_seed(429)
        design=torch.randn(40,5,3,dtype=torch.float64)
        effect=torch.tensor([2.,-1.,.5],dtype=torch.float64)
        noise=torch.randn(40,12,5,dtype=torch.float64)*.1
        returns=(design@effect)[:,None]+noise+torch.randn(40,12,1,dtype=torch.float64)*4.
        contrast=design[:,1:]-design[:,:1]
        def fit(values):
            target=(values[:,:,1:]-values[:,:,:1]).mean(1)
            return subject.reactive_gaussian_readout(contrast,target,subject.reactive_precision(values,True))
        mean,covariance,_=fit(returns)
        shifted=fit(returns+torch.randn(40,12,1,dtype=torch.float64)*100.)
        torch.testing.assert_close(mean,effect,rtol=.02,atol=.02)
        torch.testing.assert_close(mean,shifted[0],rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(covariance,shifted[1],rtol=1e-10,atol=1e-10)
        self.assertTrue(bool((torch.linalg.eigvalsh(covariance)>0.).all()))

    def test_component_fusion_reproduces_the_scalar_control_and_true_transition(self):
        import identify_fusion_mechanisms as subject
        settings=dict(bias=-.05,full_gate=True,radius=.3)
        program=torch.tensor([[.2,-.3,0.,.1]])
        scalar=subject.reactive_action(self.base,program,settings)
        component=subject.reactive_action(self.base,program[:,:,None].expand(-1,-1,2),settings)
        torch.testing.assert_close(component[...,:2],scalar[...,:2],rtol=0.,atol=0.)
        torch.testing.assert_close(component[...,2:],scalar[...,2:3].expand(-1,-1,2),rtol=0.,atol=0.)
        state,measured,agility,_=subject.DifferentiableInterventionDynamics.from_packet({k:v.double() for k,v in self.factor_packet.items()})
        noise=torch.randn(1,17,dtype=torch.float64)
        first=subject.reactive_physics_step(state,measured,agility,scalar,noise)
        second=subject.reactive_physics_step(state,measured,agility,component,noise)
        for actual,expected in zip(second,first):torch.testing.assert_close(actual,expected,rtol=0.,atol=0.)

    def test_component_options_refresh_receivers_and_close_every_joint_face(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=torch.tensor([[15.,9.,12.,7.]])
        settings=dict(bias=-.05,full_gate=True)
        programs,parts=subject.reactive_component_options(packet,self.base,settings)
        self.assertEqual(programs.shape,(1,33,4,2))
        torch.testing.assert_close(programs,programs[:,parts].sum(-3),rtol=0.,atol=0.)
        changed=(programs[:,17:].abs().sum(-1)>0.).any(1)
        self.assertEqual(changed.tolist(),[[False,True,False,True]])
        reference=(self.base[...,2]-.05).clamp(0.,1.)
        gates=programs[:,1:5,3]+reference[:,3,None,None]
        torch.testing.assert_close(gates[0],torch.tensor([[0.,0.],[1.,1.],[1.,0.],[0.,1.]]),rtol=0.,atol=1e-7)
        packet['obs'][...,2]=torch.tensor([[6.,19.,4.,17.]])
        refreshed,_=subject.reactive_component_options(packet,self.base,settings)
        changed=(refreshed[:,17:].abs().sum(-1)>0.).any(1)
        self.assertEqual(changed.tolist(),[[True,False,True,False]])

    def test_grouped_nominal_source_is_batch_invariant_and_preserves_candidates(self):
        import identify_fusion_mechanisms as subject
        from policy.networks import ActorAdap
        from types import SimpleNamespace
        device='cuda' if torch.cuda.is_available() else 'cpu'
        torch.manual_seed(431)
        source=SimpleNamespace(actor=ActorAdap(6,8,3,512).to(device).eval(),device=device)
        observations=torch.randn(2800,30)
        fast=subject.ReactiveGroupedSource(source)
        with torch.no_grad():
            actual=fast(observations)[0]
            order=torch.randperm(len(observations))[:53]
            changed=fast(observations[order])[0]
            torch.testing.assert_close(changed,actual[order],rtol=0.,atol=0.)
            reference=source.actor(observations.to(device),True,False)[0].cpu()
            torch.testing.assert_close(actual,reference,rtol=1e-4,atol=5e-5)

    def test_component_native_adapter_preserves_the_fused_physical_thrust(self):
        import identify_fusion_mechanisms as subject
        action=np.array([[.3,-.5,.2,.8],[-.8,.7,1.,0.],[.5,.1,0.,1.],[.8,.6,.4,.4]])
        boids=np.array([[500.,-400.],[700.,600.],[-300.,-100.],[800.,900.]])
        converted=subject.reactive_native_action(action,boids)
        actual=converted[:,2:3]*(750.*converted[:,:2]+250.)+(1.-converted[:,2:3])*boids
        expected=action[:,2:]*(750.*action[:,:2]+250.)+(1.-action[:,2:])*boids
        np.testing.assert_allclose(actual,expected,rtol=0.,atol=1e-12)

    def test_component_all_pairs_retain_every_existing_scalar_control(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=torch.tensor([[15.,9.,12.,7.]])
        settings=dict(bias=-.05,full_gate=True,component_all_pairs=True)
        component,parts=subject.reactive_component_options(packet,self.base,settings)
        scalar,_=subject.reactive_prior_options(packet,self.base,settings)
        self.assertEqual(component.shape,(1,113,4,2))
        expected=scalar[:,:,:,None].expand(-1,-1,-1,2)
        match=(component[:,:,None]==expected[:,None]).all(-1).all(-1)
        self.assertTrue(bool(match.any(1).all()))
        torch.testing.assert_close(component,component[:,parts].sum(-3),rtol=0.,atol=0.)

    def test_component_paired_decision_ignores_common_head_offsets(self):
        import identify_fusion_mechanisms as subject
        torch.manual_seed(429)
        prediction=torch.randn(3,4,9,dtype=torch.float64)
        programs=torch.randn(4,9,4,2,dtype=torch.float64)
        reference=torch.tensor([1,4,5,8])
        first=subject.reactive_component_paired_score(prediction,programs,reference,2.,.3)
        changed=subject.reactive_component_paired_score(prediction+100.*torch.randn(3,4,1),programs,reference,2.,.3)
        torch.testing.assert_close(first,changed,rtol=1e-10,atol=1e-10)
        torch.testing.assert_close(first[torch.arange(4),reference],torch.zeros(4,dtype=torch.float64),rtol=0.,atol=0.)

    def test_total_control_credit_corrects_joint_value_despite_unary_error(self):
        import identify_fusion_mechanisms as subject
        torch.manual_seed(432)
        costs=torch.tensor([0.,2.,3.,4.],dtype=torch.float64)[None,None].expand(8,8,-1)
        common=100.*torch.randn(8,8,1,dtype=torch.float64)
        futures=torch.full((8,8),.125,dtype=torch.float64);rows=torch.ones(8,dtype=torch.float64)
        folds=subject.reactive_crossfit_control_targets(costs+common,futures)
        shifted=subject.reactive_crossfit_control_targets(costs+common+100.*torch.randn_like(common),futures)
        # Half-sized unary predictions plus the true mixed term (-1) give
        # joint value 1.5; the measured total contrast is actually 4 seconds.
        energy=torch.tensor([0.,1.,1.5,1.5],dtype=torch.float64)[None].expand(8,-1).clone().requires_grad_()
        loss=subject.reactive_crossfit_control_loss(energy,folds,rows)
        changed=subject.reactive_crossfit_control_loss(energy,shifted,rows)
        torch.testing.assert_close(loss,changed,rtol=1e-10,atol=1e-10)
        gradient=torch.autograd.grad(loss,energy)[0]
        self.assertTrue(bool((gradient[:,-1]<0.).all()))
        correct=energy.detach().clone();correct[:,-1]=4.
        correct.requires_grad_()
        gradient=torch.autograd.grad(subject.reactive_crossfit_control_loss(correct,folds,rows),correct)[0]
        torch.testing.assert_close(gradient[:,-1],torch.zeros(8,dtype=torch.float64),rtol=0.,atol=1e-12)

    def test_factor_kernel_design_matches_deployed_graph_and_zero_axes(self):
        import identify_fusion_mechanisms as subject
        packet={k:v.expand(2,*v.shape[1:]).clone() for k,v in self.factor_packet.items()}
        packet['obs'][...,2]=10.;base=self.base.expand(2,-1,-1).clone()
        inputs=subject.reactive_factor_inputs(packet,base)
        maps={k:subject.fit_reactive_feature_map(v,8,4) for k,v in inputs.items()}
        programs=subject.reactive_options(packet,True,reference=(base[...,2]-.05).clamp(0.,1.))
        unary,mixed,_=subject.reactive_factor_design(packet,base,programs,maps)
        weights=torch.randn(1,3,unary.shape[-1]+mixed.shape[-1])
        actor=subject.ReactiveFactorKernelGate(dict(bias=-.05),maps,weights,[.5])
        actual=actor.energies(packet,base,programs)[0]
        expected=torch.einsum('bkp,mp->mbk',torch.cat((unary,mixed),-1).float(),weights[0])
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=3e-5)
        singles=(programs.abs()>1e-5).sum(-1)<2
        torch.testing.assert_close(mixed[singles],torch.zeros_like(mixed[singles]),rtol=0.,atol=0.)
        torch.testing.assert_close(actual[...,0],torch.zeros_like(actual[...,0]),rtol=0.,atol=0.)

    def test_factor_inputs_own_candidates_do_not_read_incoming_or_private_state(self):
        from identify_fusion_mechanisms import reactive_factor_inputs
        first=reactive_factor_inputs(self.factor_packet,self.base)
        base=self.base.clone();base[:,1:,:2]=-base[:,1:,:2]
        second=reactive_factor_inputs(self.factor_packet,base)
        torch.testing.assert_close(first['node'][:,0],second['node'][:,0],rtol=0.,atol=0.)
        self.assertGreater(float((first['pair'][:,0,1]-second['pair'][:,0,1]).abs().max()),1e-5)
        private=copy.deepcopy(self.factor_packet);private['central'].normal_()
        for key,value in reactive_factor_inputs(private,self.base).items():
            torch.testing.assert_close(first[key],value,rtol=0.,atol=0.)
        permuted=reactive_factor_inputs(self.factor_packet,self.base,'permuted')
        torch.testing.assert_close(first['node'],permuted['node'],rtol=0.,atol=0.)
        self.assertGreater(float((first['pair']-permuted['pair']).abs().max()),1e-5)

    def test_factor_joint_gate_redecides_from_current_candidate_messages(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        inputs=subject.reactive_factor_inputs(packet,self.base);maps={}
        for key,value in inputs.items():
            if key=='pair_fleet':continue
            size=value.shape[-1]
            maps[key]=dict(mean=torch.zeros(size),scale=torch.ones(size),projection=torch.zeros(size,1),
                centers=torch.empty(0,1),bandwidth=torch.tensor(1.))
        own_width=inputs['node'].shape[-1]
        maps['pair']['projection'][own_width+13,0]=1.
        weights=torch.zeros(1,2,12);weights[...,:2]=.1;weights[...,4+4+3]=2.
        settings=dict(bias=-.05,full_gate=True,factor_trust=.1)
        actor=subject.ReactiveFactorKernelGate(settings,maps,weights,[.5])
        positive=self.base.clone();positive[...,:2]=.5
        negative=self.base.clone();negative[...,:2]=-.5
        _,first=actor.decide(packet,positive,settings)
        _,second=actor.decide(packet,negative,settings)
        torch.testing.assert_close(first,torch.zeros_like(first),rtol=0.,atol=0.)
        self.assertEqual(int((second>0).sum()),4)
        actor.interaction=False
        _,independent=actor.decide(packet,negative,settings)
        torch.testing.assert_close(independent,torch.zeros_like(independent),rtol=0.,atol=0.)

    def test_factor_stage_anchors_current_predecessor_and_paired_uncertainty(self):
        from identify_fusion_mechanisms import ReactiveFactorKernelGate
        programs=torch.tensor([[[0.,0.],[-.5,0.],[0.,.5],[-.5,.5]]])
        actor=ReactiveFactorKernelGate(dict(factor_trust=.1,stage_max_changes=[2,1]),{},torch.zeros(2,2,1),[1.,1.])
        estimate=torch.tensor([[[[0.,-4.,-3.,-2.]],[[0.,-4.,-3.,-2.]]],
                               [[[0.,1.,-8.,-1.]],[[0.,1.,-8.,-1.]]]])
        self.assertEqual(int(actor.choices(estimate,programs)),3)
        shifted=estimate+torch.tensor([100.,-100.])[None,:,None,None]
        self.assertEqual(int(actor.choices(shifted,programs)),3)
        estimate[0,:,:,1]=-2.
        self.assertEqual(int(actor.choices(estimate,programs)),2)

    def test_factor_decision_regret_cancels_shared_futures_and_retains_rare_cost(self):
        from identify_fusion_mechanisms import reactive_pair_regret_weights,reactive_pair_regret_loss
        costs=torch.tensor([[[10.,9.],[10.,9.],[10.,9.],[10.,25.]]])
        future=torch.full((1,4),.25);rows=torch.ones(1)
        first=reactive_pair_regret_weights(costs,future)
        shifted=reactive_pair_regret_weights(costs+torch.tensor([0.,10.,-5.,100.])[None,:,None],future)
        torch.testing.assert_close(first,shifted,rtol=0.,atol=0.)
        energy=torch.zeros(1,2,requires_grad=True)
        gradient=torch.autograd.grad(reactive_pair_regret_loss(energy,first,rows),energy)[0]
        self.assertLess(float(gradient[0,1]),0.)
        self.assertGreater(float(gradient[0,0]),0.)
        null=reactive_pair_regret_weights(costs[:,:,:1].expand(-1,-1,2),future)
        self.assertEqual(float(reactive_pair_regret_loss(energy,null,rows)),0.)

    def test_factor_fleet_context_preserves_previous_policy_and_third_candidate(self):
        import identify_fusion_mechanisms as subject
        packet={k:v.expand(2,*v.shape[1:]).clone() for k,v in self.factor_packet.items()}
        packet['obs'][...,2]=10.;base=self.base.expand(2,-1,-1).clone()
        inputs=subject.reactive_factor_inputs(packet,base)
        maps={k:subject.fit_reactive_feature_map(v,8,4) for k,v in inputs.items() if k!='pair_fleet'}
        programs=subject.reactive_options(packet,True,reference=(base[...,2]-.05).clamp(0.,1.))
        unary,mixed,_=subject.reactive_factor_design(packet,base,programs,maps)
        weights=torch.randn(1,3,unary.shape[-1]+mixed.shape[-1])
        actor=subject.ReactiveFactorKernelGate(dict(bias=-.05),maps,weights,[.5])
        augmented=dict(maps,pair_fleet=subject.fit_reactive_feature_map(inputs['pair_fleet'],8,4))
        _,expanded,_=subject.reactive_factor_design(packet,base,programs,augmented)
        inherited=torch.nn.functional.pad(weights,(0,expanded.shape[-1]-mixed.shape[-1]))
        updated=subject.ReactiveFactorKernelGate(dict(bias=-.05),augmented,inherited,[.5])
        torch.testing.assert_close(actor.energies(packet,base,programs),updated.energies(packet,base,programs),rtol=1e-6,atol=2e-6)
        changed=base.clone();changed[:,2,:2]=-changed[:,2,:2]
        alternate=subject.reactive_factor_inputs(packet,changed)
        torch.testing.assert_close(inputs['pair'][:,0,1],alternate['pair'][:,0,1],rtol=0.,atol=0.)
        self.assertGreater(float((inputs['pair_fleet'][:,0,1]-alternate['pair_fleet'][:,0,1]).abs().max()),1e-5)

    def test_exact_joint_value_preserves_higher_order_candidate_consequences(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        settings=dict(full_gate=True,bias=-.05)
        programs,parts=subject.reactive_joint_options(packet,self.base,settings)
        reconstructed=programs[:,parts].sum(2)
        torch.testing.assert_close(programs,reconstructed,rtol=0.,atol=0.)
        self.assertEqual(programs.shape,(1,81,4))
        prior=torch.full((1,4,81),10.);prior[:,:,-1]=0.
        weights=torch.zeros(2,7,4);weights[:,0,0]=20.;weights[:,3:5,0]=10.
        full=subject.reactive_monotone_estimates(weights,prior,parts,packet,True)
        unary=subject.reactive_monotone_estimates(weights,prior,parts,packet,False)
        self.assertEqual(int(full.mean((0,2)).argmin()),80)
        torch.testing.assert_close(unary,torch.zeros_like(unary),rtol=0.,atol=0.)
        shifted=subject.reactive_monotone_estimates(weights,prior+torch.arange(4.)[None,:,None]*100.,parts,packet,True)
        torch.testing.assert_close(full,shifted,rtol=0.,atol=0.)

    def test_exact_joint_model_deduplication_preserves_paired_control_futures(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        programs=torch.tensor([[[0.,0.,0.,0.],[1.,0.,0.,0.],[0.,0.,0.,0.]]])
        settings=dict(exact_joint_fleet=True,agility=2.25)
        def rollout(actor,source,expanded,noise,settings,first=None):
            self.assertEqual(len(first),8)
            grouped=noise.reshape(2,4,300,17)
            torch.testing.assert_close(grouped[0],grouped[1],rtol=0.,atol=0.)
            return first.sum(-1)+noise[:,0,0],None,None
        with patch.object(subject,'reactive_rollout',side_effect=rollout):
            value=subject.reactive_model_prior(None,packet,programs,settings,4)
        self.assertEqual(value.shape,(1,4,3))
        torch.testing.assert_close(value[:,:,0],value[:,:,2],rtol=0.,atol=0.)
        torch.testing.assert_close(value[:,:,1]-value[:,:,0],torch.ones(1,4),rtol=0.,atol=1e-6)

    def test_joint_response_prefix_preserves_control_aliases_and_paired_averaging(self):
        import identify_fusion_mechanisms as subject
        programs=torch.tensor([[[0.,0.,0.,0.],[.4,0.,0.,0.],[0.,0.,0.,0.]]])
        settings=dict(exact_joint_fleet=True,agility=2.25)
        def rollout(actor,source,packet,noise,settings,first=None,return_prefix=False):
            self.assertTrue(return_prefix)
            costs=first.sum(-1)+noise[:,0,0]
            response=costs[:,None,None,None].expand(-1,4,2,2)
            return costs,None,dict(prefix_response=response)
        with patch.object(subject,'reactive_rollout',side_effect=rollout):
            values,responses=subject.reactive_model_prior(None,self.factor_packet,programs,settings,4,return_response=True)
        self.assertEqual(responses.shape,(1,3,4,2,2))
        torch.testing.assert_close(responses[:,0],responses[:,2],rtol=0.,atol=0.)
        torch.testing.assert_close(responses[:,:,:,0,0],values.mean(1)[:,:,None].expand(-1,-1,4))

    def test_recording_coupled_responses_preserves_real_physical_returns(self):
        import identify_fusion_mechanisms as subject
        from policy.interaction_sac import MirrorCoordinationGate
        packet=copy.deepcopy(self.factor_packet)
        packet['central'][:,-1]=.01;packet['time_remaining'][:]=.01
        settings=dict(period=10,full_gate=True,bias=-.05,coupled_potential=True)
        class Source:
            device='cpu'
            def __call__(self,observation,*args):
                return (torch.full((len(observation),3),.5),None)
        source=Source();noise=torch.zeros(1,300,17);first=torch.zeros(1,4)
        plain=subject.reactive_rollout(MirrorCoordinationGate(16),source,packet,noise,settings,first=first)
        recorded=subject.reactive_rollout(MirrorCoordinationGate(16),source,packet,noise,settings,first=first,return_prefix=True)
        torch.testing.assert_close(plain[0],recorded[0],rtol=0.,atol=0.)
        torch.testing.assert_close(plain[1],recorded[1],rtol=0.,atol=0.)
        response=recorded[2]['prefix_response']
        self.assertEqual(response.shape,(1,4,2,2));self.assertTrue(bool(torch.isfinite(response).all()))
        torch.testing.assert_close(response[:,:,0],response[:,:,1],rtol=0.,atol=0.)

    def test_conditional_interference_cannot_discount_explicit_joint_penalties(self):
        from identify_fusion_mechanisms import reactive_monotone_estimates
        packet={k:v.expand(2,*v.shape[1:]).clone() for k,v in self.factor_packet.items()}
        parts=torch.tensor([[0,0],[1,0],[2,0],[1,2]])
        prior=torch.tensor([[[10.,6.,7.,8.]],[[10.,6.,7.,-2.]]]).expand(-1,4,-1)
        weights=torch.zeros(2,9,4);weights[:,0,0]=20.;weights[:,3,0]=20.;weights[:,8,0]=-4.
        value=reactive_monotone_estimates(weights,prior,parts,packet,True)
        forecast=prior-prior[...,:1]
        self.assertTrue(bool((value>=forecast[None]-1e-5).all()))
        additive=reactive_monotone_estimates(weights,prior,parts,packet,False)
        torch.testing.assert_close(additive[...,3],torch.full_like(additive[...,3],-7.),rtol=0.,atol=1e-5)
        self.assertGreater(float(value[:,0,:,3].mean()),-2.)
        self.assertLess(float(value[:,1,:,3].mean()),-7.)

    def test_joint_values_ignore_duplicate_encodings_of_the_same_physical_controls(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        base=self.base.clone();base[:,2:,2]=0.
        programs,parts=subject.reactive_joint_options(packet,base,dict(full_gate=True,bias=-.05))
        unique,inverse=torch.unique(programs[0],dim=0,return_inverse=True)
        self.assertLess(len(unique),len(parts))
        prior=torch.randn(1,4,len(unique))[...,inverse]
        packet['candidate_interactions']=subject.reactive_candidate_interactions(packet,base,programs,parts)
        for heads in (7,8,9,10):
            value=subject.reactive_monotone_estimates(torch.randn(3,heads,16),prior,parts,packet)
            for index in range(len(unique)):
                selected=value[...,inverse==index]
                torch.testing.assert_close(selected,selected[...,:1].expand_as(selected),rtol=0.,atol=1e-6)

    def test_selected_candidate_value_can_update_one_receiver_from_unchanged_teammate_candidates(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        programs,parts=subject.reactive_joint_options(packet,self.base,dict(full_gate=True,bias=-.05))
        prior=torch.zeros(1,4,len(parts))
        context=subject.reactive_selected_coordination_context(packet,self.base,programs,parts,prior)
        specification=dict(coordinates_version='selected-candidates-v1',bias=-.05,
            centers=context[0,1:2].tolist(),scale=torch.full((context.shape[-1],),.15).tolist(),bandwidth=1.)
        data=dict(packet=packet,base=self.base,options=programs,parts=parts)
        encoded=subject.reactive_local_candidate_data(data,prior,specification)
        weights=torch.zeros(2,9,17);weights[:,7,-1]=-3.
        value=subject.reactive_monotone_estimates(weights,prior,parts,encoded['packet'])
        self.assertEqual(int((programs[0,1]!=0.).sum()),1)
        torch.testing.assert_close(value[...,0],torch.zeros_like(value[...,0]),rtol=0.,atol=0.)
        torch.testing.assert_close(value[...,1],torch.full_like(value[...,1],-3.),rtol=0.,atol=1e-6)
        changed=self.base.clone();changed[:,1:,:2]=-changed[:,1:,:2]
        incoming=subject.reactive_local_candidate_data(dict(data,base=changed),prior,specification)
        other=subject.reactive_monotone_estimates(weights,prior,parts,incoming['packet'])
        self.assertGreater(float((other[...,1]-value[...,1]).abs().max()),1e-3)
        disabled=subject.reactive_monotone_estimates(weights,prior,parts,encoded['packet'],False)
        torch.testing.assert_close(disabled,torch.zeros_like(disabled),rtol=0.,atol=0.)
        private=copy.deepcopy(packet);private['central'].normal_()
        same=subject.reactive_selected_coordination_context(private,self.base,programs,parts,prior)
        torch.testing.assert_close(context,same,rtol=0.,atol=0.)

    def test_centered_cooperative_value_is_continuous_at_the_reference_control(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        programs,parts=subject.reactive_joint_options(packet,self.base,dict(full_gate=True,bias=-.05))
        prior=torch.zeros(1,4,len(parts))
        context=subject.reactive_selected_coordination_context(packet,self.base,programs,parts,prior)
        specification=dict(coordinates_version='selected-candidates-v1',centered=True,bias=-.05,
            centers=context[0,1:2].tolist(),scale=torch.full((context.shape[-1],),.15).tolist(),bandwidth=1.)
        weights=torch.zeros(2,9,17);weights[:,7,-1]=2.
        differences=[]
        for step in (1e-2,1e-4,0.):
            nearby=programs.clone();nearby[:,1]=0.;nearby[:,1,0]=step
            encoded=subject.reactive_local_candidate_data(dict(packet=packet,base=self.base,options=nearby,parts=parts),prior,specification)
            value=subject.reactive_monotone_estimates(weights,prior,parts,encoded['packet'])
            differences.append(float(value[...,1].abs().max()))
            torch.testing.assert_close(value[...,0],torch.zeros_like(value[...,0]),rtol=0.,atol=0.)
        self.assertGreater(differences[0],0.)
        self.assertLess(differences[1],differences[0]/20.)
        self.assertEqual(differences[2],0.)

    def test_adaptive_joint_value_can_correct_overstated_harm_and_understated_synergy(self):
        from identify_fusion_mechanisms import reactive_monotone_estimates
        packet={k:v.expand(2,*v.shape[1:]).clone() for k,v in self.factor_packet.items()}
        parts=torch.tensor([[0,0],[1,0],[2,0],[1,2]])
        prior=torch.tensor([[[10.,6.,7.,8.]],[[10.,6.,7.,-2.]]]).expand(-1,4,-1)
        weights=torch.zeros(2,10,4);weights[:,0,0]=20.;weights[:,3,0]=20.
        weights[:,4,0]=torch.expm1(torch.tensor(2.)).log();weights[:,8,0]=-10.
        value=reactive_monotone_estimates(weights,prior,parts,packet,True)
        self.assertLess(float(value[:,0,:,3].mean()),-6.9)
        self.assertLess(float(value[:,1,:,3].mean()),-16.9)
        independent=reactive_monotone_estimates(weights,prior,parts,packet,False)
        torch.testing.assert_close(independent[...,3],torch.full_like(independent[...,3],-7.),rtol=0.,atol=1e-5)

    def test_joint_decoder_stays_within_identified_coalition_order(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        settings=dict(full_gate=True,bias=-.05,exact_joint_fleet=True,posterior_update=True,
            canonical_model=True,conditional_posterior_update=True,prior=.3,max_coordinated_receivers=2)
        programs,parts=subject.reactive_joint_options(packet,self.base,settings)
        prior=(20.-3.*(programs!=0.).sum(-1))[:,None].expand(-1,4,-1)
        weights=torch.zeros(3,10,16);weights[:,0,0]=20.;weights[:,3,0]=20.
        actor=subject.ReactiveModelPriorGate(settings,weights,0.,4,source=object())
        with patch.object(subject,'reactive_model_prior',return_value=prior):
            _,action=actor.decide(packet,self.base,settings)
        self.assertEqual(int((action!=0.).sum()),2)
        settings=dict(settings,max_coordinated_receivers=4)
        with patch.object(subject,'reactive_model_prior',return_value=prior):
            _,expanded=actor.decide(packet,self.base,settings)
        self.assertEqual(int((expanded!=0.).sum()),4)

    def test_cooperation_removal_recovers_the_same_learned_unary_decision(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        settings=dict(full_gate=True,bias=-.05,exact_joint_fleet=True,posterior_update=True,
            canonical_model=True,conditional_posterior_update=True,prior=.3,max_coordinated_receivers=2)
        programs,parts=subject.reactive_joint_options(packet,self.base,settings)
        prior=(20.-3.*programs.square().sum(-1))[:,None].expand(-1,4,-1)
        unary=torch.zeros(3,10,16);unary[:,0,0]=20.;unary[:,3,0]=20.
        full=torch.randn(3,10,16)
        reference=subject.ReactiveModelPriorGate(settings,unary,0.,4,source=object(),interaction=False)
        for enabled,interaction in ((True,False),(False,True)):
            actor=subject.ReactiveModelPriorGate(settings,full,2.,4,source=object(),interaction=interaction,
                unary_reference=unary,cooperation_enabled=enabled)
            with patch.object(subject,'reactive_model_prior',return_value=prior):
                expected,expected_action=reference.decide(packet,self.base,settings)
                actual,action=actor.decide(packet,self.base,settings)
            self.assertGreater(int((action!=0.).sum()),0)
            torch.testing.assert_close(actual,expected,rtol=0.,atol=0.)
            torch.testing.assert_close(action,expected_action,rtol=0.,atol=0.)
        saved=dict(calibration=full,unary_reference=unary,settings=settings,risk=2.)
        panel,_=subject.reactive_joint_credit_panel(saved,packet,self.base,programs,parts,prior,expected)
        self.assertEqual(int(panel[0,5]),int(expected))
        self.assertLessEqual(int((programs[0,panel[0,3]]!=0.).sum()),2)

    def test_active_joint_credit_frames_do_not_read_realized_returns(self):
        import identify_fusion_mechanisms as subject
        packet={k:v.expand(3,*v.shape[1:]).clone() for k,v in self.factor_packet.items()}
        packet['obs'][...,2]=10.;base=self.base.expand(3,-1,-1).clone()
        settings=dict(full_gate=True,bias=-.05)
        programs,parts=subject.reactive_joint_options(packet,base,settings)
        score=torch.zeros(3,len(parts));score[:,1]=torch.tensor([-1.,-4.,-2.])
        trajectory=dict(packet=packet,base=base,model_prior=torch.zeros(3,4,len(parts)),
            model_objective=score,choice=torch.ones(3,dtype=torch.long),origin=torch.tensor([0,0,1]),returns=torch.tensor([100.,0.,20.]))
        with patch.object(subject,'reactive_credit_unary',return_value=(score,torch.zeros(3,dtype=torch.long),None)):
            first,_=subject.reactive_joint_disagreement_frames(dict(settings=settings),trajectory,2)
            trajectory['returns']=-trajectory['returns']
            second,_=subject.reactive_joint_disagreement_frames(dict(settings=settings),trajectory,2)
        torch.testing.assert_close(first,torch.tensor([1,2]),rtol=0.,atol=0.)
        torch.testing.assert_close(first,second,rtol=0.,atol=0.)

    def test_continuous_cooperation_does_not_inherit_a_previous_joint_choice(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        settings=dict(full_gate=True,bias=-.05,exact_joint_fleet=True,posterior_update=True,
            canonical_model=True,conditional_posterior_update=True,continuous_cooperation_test=True,
            prior=.3,max_coordinated_receivers=2)
        programs,parts=subject.reactive_joint_options(packet,self.base,settings)
        prior=(20.-3.*programs.square().sum(-1))[:,None].expand(-1,4,-1)
        unary=torch.zeros(3,10,16);unary[:,0,0]=20.;unary[:,3,0]=20.
        current=unary.clone();current[:,4,0]=2.
        actor=subject.ReactiveModelPriorGate(settings,current,1.,4,source=object(),unary_reference=unary)
        with_history=subject.ReactiveModelPriorGate(settings,current,1.,4,source=object(),unary_reference=unary,
            history_calibrations=[-current,current*10.],history_risks=[100.,0.],history_initial_posterior=True)
        with patch.object(subject,'reactive_model_prior',return_value=prior):
            first,first_action=actor.decide(packet,self.base,settings)
            second,second_action=with_history.decide(packet,self.base,settings)
        torch.testing.assert_close(first,second,rtol=0.,atol=0.)
        torch.testing.assert_close(first_action,second_action,rtol=0.,atol=0.)
        torch.testing.assert_close(actor.last_prediction['objective'],with_history.last_prediction['objective'],rtol=0.,atol=0.)

    def test_shared_candidate_edges_ignore_boat_identity_and_private_state(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet)
        programs,parts=subject.reactive_prior_options(packet,self.base,dict(full_gate=True,bias=-.05,control_radius=1000.))
        data=dict(packet=packet,base=self.base,options=programs,parts=parts)
        specification=subject.reactive_edge_dictionary(data,8)
        prior=torch.zeros(1,4,len(parts),dtype=torch.float64)
        first=subject.reactive_local_candidate_features(packet,self.base,programs,parts,prior,specification)
        permutation=torch.tensor([2,0,3,1]);permuted=dict(packet)
        for key in ('obs','motion'):permuted[key]=packet[key][:,permutation]
        permuted['central']=torch.randn_like(packet['central'])*100.
        second=subject.reactive_local_candidate_features(permuted,self.base[:,permutation],programs[:,:,permutation],parts,prior,specification)
        torch.testing.assert_close(first,second,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(first[:,0],torch.zeros_like(first[:,0]),rtol=0.,atol=0.)
        self.assertGreater(float(first[:,1:].abs().max()),1e-5)

    def test_shared_edge_messages_respond_to_unchanged_teammate_candidates(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet)
        programs=torch.zeros(1,2,4);programs[:,1,0]=.2
        parts=torch.tensor([[0,0],[1,0]])
        data=dict(packet=packet,base=self.base,options=programs,parts=parts)
        specification=subject.reactive_edge_dictionary(data,8)
        prior=torch.zeros(1,4,2)
        original=subject.reactive_local_candidate_features(packet,self.base,programs,parts,prior,specification)
        altered=self.base.clone();altered[:,1:,:2]=-altered[:,1:,:2]
        changed=subject.reactive_local_candidate_features(packet,altered,programs,parts,prior,specification)
        self.assertGreater(float((original[:,1,12:]-changed[:,1,12:]).abs().max()),1e-5)
        torch.testing.assert_close(changed[:,0],torch.zeros_like(changed[:,0]),rtol=0.,atol=0.)

    def test_complete_edge_value_features_vanish_with_a_vanishing_joint_control(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet)
        parts=torch.tensor([[0,0],[1,0],[2,0],[1,2]])
        programs=torch.zeros(1,4,4);programs[:,1,0]=.1;programs[:,2,1]=.1;programs[:,3]=programs[:,1]+programs[:,2]
        data=dict(packet=packet,base=self.base,options=programs,parts=parts)
        specification=subject.reactive_edge_dictionary(data,8)
        prior=torch.zeros(1,4,4,dtype=torch.float64)
        magnitudes=[]
        for scale in (1.,.01,0.):
            features=subject.reactive_local_candidate_features(packet,self.base,programs*scale,parts,prior,specification)
            magnitudes.append(float(features[:,3].norm()))
        self.assertGreater(magnitudes[0],1e-5)
        self.assertLess(magnitudes[1],magnitudes[0]/20.)
        self.assertEqual(magnitudes[2],0.)

    def test_candidate_edges_distinguish_task_stage_with_equal_local_geometry(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);programs=torch.zeros(1,2,4);programs[:,1,0]=.2
        first=subject.reactive_edge_context(packet,self.base,programs,global_context=True)
        changed=copy.deepcopy(packet);changed['time_remaining']=packet['time_remaining']*.5
        second=subject.reactive_edge_context(changed,self.base,programs,global_context=True)
        torch.testing.assert_close(first[...,:-1],second[...,:-1],rtol=0.,atol=0.)
        self.assertGreater(float((first[...,-1]-second[...,-1]).abs().max()),.1)
        unchanged=subject.reactive_edge_context(changed,self.base,programs,global_context=False)
        torch.testing.assert_close(first[...,:25],unchanged,rtol=0.,atol=0.)

    def test_paired_decision_loss_excludes_infeasible_experimental_controls(self):
        from identify_fusion_mechanisms import reactive_pair_regret_weights,reactive_pair_regret_loss
        costs=torch.tensor([[[2.,4.,-100.],[3.,5.,100.]]])
        energy=torch.tensor([[1.,2.,-1000.]],requires_grad=True)
        regret=reactive_pair_regret_weights(costs,torch.full((1,2),.5))
        loss=reactive_pair_regret_loss(energy,regret,torch.ones(1),support=torch.tensor([[True,True,False]]))
        expected=reactive_pair_regret_loss(energy[:,:2],regret[:,:2,:2],torch.ones(1))
        torch.testing.assert_close(loss,expected)
        loss.backward()
        self.assertEqual(float(energy.grad[0,2]),0.)
        self.assertGreater(float(energy.grad[0,:2].abs().sum()),0.)

    def test_decision_regret_is_invariant_to_state_values_and_control_aliases(self):
        from identify_fusion_mechanisms import reactive_decision_regret_loss,reactive_unique_control_mask
        controls=torch.tensor([[[0.,0.],[.2,0.],[0.,.3],[0.,0.],[.2,0.]]])
        support=reactive_unique_control_mask(controls)
        torch.testing.assert_close(support,torch.tensor([[True,True,True,False,False]]))
        energy=torch.tensor([[0.,-1.,2.,0.,-1.]],requires_grad=True)
        costs=torch.tensor([[3.,5.,1.,3.,5.]])
        loss=reactive_decision_regret_loss(energy,costs,torch.ones(1),support)
        shifted=reactive_decision_regret_loss(energy+37.,costs-91.,torch.ones(1),support)
        unique=reactive_decision_regret_loss(energy[:,:3],costs[:,:3],torch.ones(1))
        torch.testing.assert_close(loss,shifted,rtol=1e-5,atol=1e-5)
        torch.testing.assert_close(loss,unique)
        loss.backward()
        self.assertLess(float(energy.grad[0,1]),0.)
        self.assertGreater(float(energy.grad[0,2]),0.)
        torch.testing.assert_close(energy.grad[:,3:],torch.zeros(1,2))

    def test_joint_credit_panel_is_closed_around_the_actual_current_gate(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        settings=dict(full_gate=True,bias=-.05,exact_joint_fleet=True,posterior_update=True)
        programs,parts=subject.reactive_joint_options(packet,self.base,settings)
        prior=(15.+programs.square().sum(-1))[:,None].expand(-1,4,-1)
        coefficients=torch.zeros(3,9,16);coefficients[:,8,0]=-4.
        saved=dict(calibration=coefficients,settings=settings,risk=2.)
        panel,design=subject.reactive_joint_credit_panel(saved,packet,self.base,programs,parts,prior)
        actor=subject.ReactiveModelPriorGate(settings,coefficients,2.,4,source=object())
        with patch.object(subject,'reactive_model_prior',return_value=prior):
            choice,action=actor.decide(packet,self.base,settings)
        self.assertEqual(int(panel[0,0]),int(choice))
        torch.testing.assert_close(programs[:,panel[0,0]],action,rtol=0.,atol=0.)
        selected=programs[:,panel[0,:4]]
        torch.testing.assert_close(selected[:,3],selected[:,1]+selected[:,2]-selected[:,0],rtol=0.,atol=0.)
        self.assertEqual(int((selected[:,1]!=selected[:,0]).sum()),1)
        self.assertEqual(int((selected[:,2]!=selected[:,0]).sum()),1)
        self.assertEqual(panel.shape,(1,7))
        private=copy.deepcopy(packet);private['central'].normal_()
        changed,_=subject.reactive_joint_credit_panel(saved,private,self.base,programs,parts,prior)
        torch.testing.assert_close(panel,changed,rtol=0.,atol=0.)
        recorded,_=subject.reactive_joint_credit_panel(saved,packet,self.base,programs,parts,prior,torch.tensor([80]))
        self.assertEqual(int(recorded[0,0]),80)
        actual=programs[:,recorded[0,:4]]
        torch.testing.assert_close(actual[:,3],actual[:,1]+actual[:,2]-actual[:,0],rtol=0.,atol=1e-6)

    def test_conditional_mirror_update_preserves_parent_and_cancels_state_values(self):
        from identify_fusion_mechanisms import reactive_mirror_value_update
        previous=torch.tensor([[4.,0.,2.,3.]])
        values=torch.randn(3,1,4,4)
        _,same=reactive_mirror_value_update(previous,values,values,2.)
        self.assertEqual(int(same),1)
        corrected=values.clone();corrected[...,3]-=5.
        score,choice=reactive_mirror_value_update(previous,values,corrected,2.)
        self.assertEqual(int(choice),3)
        shift=torch.tensor([10.,-100.,30.])[:,None,None,None]
        shifted,other=reactive_mirror_value_update(previous,values,corrected+shift,2.)
        torch.testing.assert_close(score,shifted,rtol=1e-5,atol=1e-5)
        self.assertEqual(int(other),3)
        corrected[0,...,3]+=6.;corrected[1,...,3]-=6.
        _,uncertain=reactive_mirror_value_update(previous,values,corrected,2.)
        self.assertEqual(int(uncertain),1)

    def test_conditional_posterior_update_matches_the_supervised_mean_and_paired_uncertainty(self):
        from identify_fusion_mechanisms import reactive_conditional_posterior_update
        programs=torch.tensor([[[0.,0.],[1.,0.],[0.,1.]]])
        values=torch.tensor([0.,-3.,-1.])[None,None,None].expand(3,1,4,-1).clone()
        reference=torch.tensor([0])
        score,choice=reactive_conditional_posterior_update(values,programs,reference,2.)
        self.assertEqual(int(choice),1)
        shifted=values+torch.tensor([10.,-20.,50.])[:,None,None,None]
        shifted[:,:,0,1]+=100.;shifted[:,:,1,1]-=100.
        same,other=reactive_conditional_posterior_update(shifted,programs,reference,2.)
        torch.testing.assert_close(score,same,rtol=0.,atol=1e-6)
        self.assertEqual(int(other),1)
        uncertain=values.clone();uncertain[0,:,:,1:]+=10.;uncertain[1,:,:,1:]-=10.
        _,held=reactive_conditional_posterior_update(uncertain,programs,reference,2.)
        self.assertEqual(int(held),0)

    def test_recursive_joint_gate_recomputes_parent_and_zero_correction_is_exact(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.factor_packet);packet['obs'][...,2]=10.
        settings=dict(full_gate=True,bias=-.05,exact_joint_fleet=True,posterior_update=True,canonical_model=True)
        weights=torch.zeros(3,9,16);weights[:,8,0]=-4.
        parent=subject.ReactiveModelPriorGate(settings,weights,.5,4,source=object())
        child=subject.ReactiveModelPriorGate(settings,weights,.5,4,source=object(),history_calibrations=[weights.clone()],history_risks=[.5])
        programs,_=subject.reactive_joint_options(packet,self.base,settings)
        choices=[]
        for sign in (1.,-1.):
            prior=(15.+sign*programs.square().sum(-1))[:,None].expand(-1,4,-1)
            with patch.object(subject,'reactive_model_prior',return_value=prior):
                old,old_action=parent.decide(packet,self.base,settings)
                new,new_action=child.decide(packet,self.base,settings)
            torch.testing.assert_close(old_action,new_action,rtol=0.,atol=0.)
            choices.append(int(new))
        self.assertNotEqual(*choices)

    def test_canonical_nominal_source_uses_fixed_batches_without_changing_controls(self):
        from identify_fusion_mechanisms import ReactiveCanonicalSource
        calls=[]
        def source(observations,*args):
            calls.append(len(observations))
            return observations[:,:3]*2.,observations[:,:1]+1.
        source.device='cpu';wrapped=ReactiveCanonicalSource(source,32)
        data=torch.arange(195.).reshape(65,3)
        action,logp=wrapped(data,True,False)
        torch.testing.assert_close(action,data*2.,rtol=0.,atol=0.)
        torch.testing.assert_close(logp,data[:,:1]+1.,rtol=0.,atol=0.)
        self.assertEqual(calls,[32,32,32])
        one,_=wrapped(data[64:],True,False)
        torch.testing.assert_close(one,action[64:],rtol=0.,atol=0.)

    def test_response_potential_cannot_invent_value_for_identical_candidates(self):
        from policy.interaction_sac import CandidateCoordinationGate
        from identify_fusion_mechanisms import reactive_options
        model=CandidateCoordinationGate(16,response_encoding=True,cardinal_gates=True,potential_value=True)
        with torch.no_grad():
            model.motion_potential.weight.normal_();model.pair_potential.weight.normal_()
        base=self.base.clone();base[...,:2]=self.packet['obs'][...,12:14]
        options=reactive_options(self.packet,True,reference=(base[...,2]-.05).clamp(0.,1.))
        energy=model(self.packet,base,options)[1]
        torch.testing.assert_close(energy,torch.zeros_like(energy),rtol=0.,atol=2e-6)

    def test_response_potential_own_value_masks_incoming_candidate_information(self):
        from policy.interaction_sac import CandidateCoordinationGate
        changed=self.base.clone();changed[:,1:,:2]=-changed[:,1:,:2]
        for peer in (True,False):
            model=CandidateCoordinationGate(16,peer,response_encoding=True,cardinal_gates=True,potential_value=True)
            with torch.no_grad():
                model.motion_potential.weight.normal_();model.pair_potential.weight.normal_()
            first=model.factors(self.packet,self.base)
            second=model.factors(self.packet,changed)
            if peer:self.assertGreater(float((first[1][:,0]-second[1][:,0]).abs().max()),1e-7)
            else:
                torch.testing.assert_close(first[1][:,0],second[1][:,0],rtol=0.,atol=0.)
                torch.testing.assert_close(first[2],torch.zeros_like(first[2]),rtol=0.,atol=0.)

    def test_response_potential_cached_probe_does_not_reuse_aligned_interaction(self):
        from policy.interaction_sac import CandidateCoordinationGate
        model=CandidateCoordinationGate(16,response_encoding=True,cardinal_gates=True,potential_value=True)
        with torch.no_grad():
            model.motion_potential.weight.normal_();model.pair_potential.weight.normal_()
        cached=model.cache_potentials(self.packet,self.base,-.05,True)
        for probe in ('aligned','permuted','state-only'):
            for expected,actual in zip(model.factors(self.packet,self.base,probe),model.factors(cached,self.base,probe)):
                torch.testing.assert_close(expected,actual,rtol=0.,atol=0.)

    def test_response_potential_conditional_control_gradient_ignores_return_shift(self):
        from policy.interaction_sac import CandidateCoordinationGate
        from identify_fusion_mechanisms import reactive_options,reactive_precision,reactive_value_loss
        model=CandidateCoordinationGate(16,response_encoding=True,cardinal_gates=True,potential_value=True)
        options=reactive_options(self.packet,reference=(self.base[...,2]-.05).clamp(0.,1.))
        costs=torch.randn(1,4,9)*2.+15.
        data=dict(packet=self.packet,base=self.base,options=options,costs=costs,precision=reactive_precision(costs))
        parameters=[model.motion_potential.weight,model.pair_potential.weight]
        first=torch.autograd.grad(reactive_value_loss(model,data,True,True),parameters)
        shifted=dict(data,costs=costs+torch.arange(4.)[None,:,None]*10.)
        second=torch.autograd.grad(reactive_value_loss(model,shifted,True,True),parameters)
        for expected,actual in zip(first,second):torch.testing.assert_close(expected,actual,rtol=1e-4,atol=1e-6)

    def test_control_only_line_search_objective_has_its_actual_derivative(self):
        from policy.interaction_sac import CandidateCoordinationGate
        from identify_fusion_mechanisms import reactive_options,reactive_precision,reactive_value_loss
        model=CandidateCoordinationGate(16,response_encoding=True,cardinal_gates=True,potential_value=True).double()
        packet={k:v.double() for k,v in self.packet.items()};base=self.base.double()
        options=reactive_options(packet,reference=(base[...,2]-.05).clamp(0.,1.))
        costs=torch.randn(1,4,9,dtype=torch.float64)*2.+15.
        data=dict(packet=packet,base=base,options=options,costs=costs,precision=reactive_precision(costs))
        data['old_log_policy']=(-.3*options.square().sum(-1)).log_softmax(-1)
        for preference in (False,True):
            objective=lambda:reactive_value_loss(model,data,True,True,.3,control_only=True,preference=preference)
            gradient=torch.autograd.grad(objective(),model.motion_potential.weight)[0][0,0]
            with torch.no_grad():model.motion_potential.weight[0,0]=1e-4
            plus=float(objective().detach())
            with torch.no_grad():model.motion_potential.weight[0,0]=-1e-4
            minus=float(objective().detach())
            with torch.no_grad():model.motion_potential.weight[0,0]=0.
            self.assertAlmostEqual(float(gradient),(plus-minus)/2e-4,places=7)

    def test_curved_gate_represents_beneficial_interior_with_bad_endpoints(self):
        from policy.interaction_sac import CandidateCoordinationGate
        unary=torch.tensor([[[-1.,2.,-1.,2.]]]);mixed=torch.zeros(1,1,1,4,4)
        factors=(torch.zeros(1),unary,mixed,torch.full((1,1),.5))
        programs=torch.tensor([[[0.],[-.5],[-.25],[.25],[.5]]])
        values=CandidateCoordinationGate.energies(factors,programs)
        torch.testing.assert_close(values,torch.tensor([[0.,2.,-1.,-1.,2.]]),rtol=0.,atol=0.)

    def test_curved_potential_preserves_endpoint_values_and_serialization(self):
        from policy.interaction_sac import CandidateCoordinationGate,MirrorCoordinationGate
        from identify_fusion_mechanisms import reactive_options
        common=dict(response_encoding=True,cardinal_gates=True,potential_value=True)
        linear=CandidateCoordinationGate(16,**common)
        with torch.no_grad():linear.motion_potential.weight.normal_();linear.pair_potential.weight.normal_()
        curved=CandidateCoordinationGate(16,curved_potential=True,**common)
        curved.load_state_dict(linear.state_dict())
        programs=reactive_options(self.packet,True,reference=(self.base[...,2]-.05).clamp(0.,1.))
        torch.testing.assert_close(curved(self.packet,self.base,programs)[1],linear(self.packet,self.base,programs)[1],atol=2e-5,rtol=1e-5)
        actor=MirrorCoordinationGate(16,curved_potential=True,**common);actor.add_value(curved,.3)
        restored=MirrorCoordinationGate(16,increments=1,curved_potential=True,**common);restored.load_state_dict(actor.state_dict())
        torch.testing.assert_close(actor(self.packet,self.base,programs)[1],restored(self.packet,self.base,programs)[1],atol=0.,rtol=0.)

    def test_applied_candidate_paths_use_the_actual_joint_safety_projection(self):
        from policy.interaction_sac import CandidateCoordinationGate
        from train_response_fusion import SafetyDerivative
        paths=CandidateCoordinationGate.applied_candidate_paths(self.packet,self.base,-.05,True,'aligned',True)
        gates=(self.base[...,2]-.05).clamp(0.,1.)
        gates[:,3]=.5*(1.+gates[:,3])
        boids=self.packet['obs'][...,12:14]
        nominal=750.*(boids+gates[...,None]*(self.base[...,:2]-boids))+250.
        applied=SafetyDerivative.apply(self.packet['motion'],nominal)
        expected=CandidateCoordinationGate.candidate_responses(self.packet,self.base,applied_thrust=applied[:,:,None])[1]
        torch.testing.assert_close(paths[:,3,:,3],expected[:,3,:,0],atol=2e-5,rtol=1e-5)

    def test_applied_candidate_path_masks_other_candidates_but_keeps_own(self):
        from policy.interaction_sac import CandidateCoordinationGate
        from train_response_fusion import SafetyDerivative
        changed=self.base.clone();changed[:,1:,:2]=-changed[:,1:,:2]
        def coupled_filter(motion,commands):return commands-.2*commands.mean(1,keepdim=True)
        with patch.object(SafetyDerivative,'apply',side_effect=coupled_filter):
            for peer in (True,False):
                a=CandidateCoordinationGate.applied_candidate_paths(self.packet,self.base,-.05,peer,'aligned',True)
                b=CandidateCoordinationGate.applied_candidate_paths(self.packet,changed,-.05,peer,'aligned',True)
                if peer:self.assertGreater(float((a[:,0]-b[:,0]).abs().max()),1e-5)
                else:torch.testing.assert_close(a[:,0],b[:,0],rtol=0.,atol=0.)

    def test_coupled_evasion_uses_observed_turn_without_private_state(self):
        from policy.interaction_sac import CandidateCoordinationGate
        packet=dict(self.packet,target_heading=torch.tensor([.6]),target_turn=torch.tensor([.7]))
        model=CandidateCoordinationGate(16,response_encoding=True,cardinal_gates=True,potential_value=True,curved_potential=True,coupled_potential=True)
        with torch.no_grad():model.motion_potential.weight.normal_();model.pair_potential.weight.normal_()
        original=model.factors(packet,self.base)
        altered=model.factors(dict(packet,central=torch.randn_like(packet['central'])*100.),self.base)
        for a,b in zip(original,altered):torch.testing.assert_close(a,b,rtol=0.,atol=0.)
        changed=model.factors(dict(packet,target_turn=-packet['target_turn']),self.base)
        self.assertGreater(float((original[1]-changed[1]).abs().max()),1e-5)

    def test_coupled_evasion_no_peer_does_not_see_other_candidates(self):
        from policy.interaction_sac import CandidateCoordinationGate
        packet=dict(self.packet,target_heading=torch.tensor([.6]),target_turn=torch.tensor([.7]))
        model=CandidateCoordinationGate(16,False,response_encoding=True,cardinal_gates=True,potential_value=True,curved_potential=True,coupled_potential=True)
        with torch.no_grad():model.motion_potential.weight.normal_();model.pair_potential.weight.normal_()
        base=self.base.clone();base[:,1:,:2]=-base[:,1:,:2]
        first=model.factors(packet,self.base);second=model.factors(packet,base)
        torch.testing.assert_close(first[1][:,0],second[1][:,0],rtol=0.,atol=0.)
        torch.testing.assert_close(first[2],torch.zeros_like(first[2]),rtol=0.,atol=0.)

    def test_local_conditional_precision_cancels_common_shocks_and_tracks_rows(self):
        from identify_fusion_mechanisms import reactive_precision,reactive_subset
        costs=torch.randn(5,16,9,dtype=torch.float64)
        costs[2:]*=10.
        precision=reactive_precision(costs,True)
        shifted=reactive_precision(costs+torch.randn(5,16,1,dtype=torch.float64)*100.,True)
        torch.testing.assert_close(precision,shifted,rtol=1e-10,atol=1e-10)
        selected=reactive_subset(dict(precision=precision,costs=costs),torch.tensor([4,0]))
        torch.testing.assert_close(selected['precision'],precision[[4,0]],rtol=0.,atol=0.)
        self.assertLess(float(precision[4].trace()),float(precision[0].trace()))

    def test_deployed_turn_belief_uses_heading_history_before_gate_activation(self):
        import identify_fusion_mechanisms as subject
        policy=subject.ReactiveGatePolicy.__new__(subject.ReactiveGatePolicy)
        policy.settings=dict(coupled_potential=True,trigger=20.,period=1,radius=.3,bias=-.05,full_gate=True)
        policy.probe='aligned';policy.model=None;policy.source=lambda obs:self.base[0].numpy().copy();policy.reset()
        packet={k:v[0].numpy().copy() for k,v in self.packet.items() if k!='central'}
        packet.update(target_heading=np.float32(6.2),time_remaining=np.float32(.9))
        packet['obs'][:,2]=100.
        policy(packet)
        packet['target_heading']=np.float32(.02);packet['obs'][:,2]=[8.,10.,20.,25.]
        def decide(model,p,base,settings,probe):
            self.assertAlmostEqual(float(p['target_turn'][0]),float(np.arctan2(np.sin(.02-float(np.float32(6.2))),np.cos(.02-float(np.float32(6.2))))/.2),places=5)
            return torch.zeros(1,dtype=torch.long),torch.zeros(1,4)
        with patch.object(subject,'reactive_decision',side_effect=decide):policy(packet)

    def test_public_response_prior_cannot_read_simulator_state(self):
        from identify_fusion_mechanisms import reactive_public_model_packet
        packet=dict(self.packet,target_heading=torch.tensor([.4]),target_turn=torch.tensor([.7]),time_remaining=torch.tensor([.8]))
        a=reactive_public_model_packet(packet,2.25)
        b=reactive_public_model_packet(dict(packet,central=torch.randn_like(packet['central'])*100.),2.25)
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=0.,atol=0.)

    def test_model_prior_design_has_all_pairs_and_no_unlisted_joint_changes(self):
        from identify_fusion_mechanisms import reactive_prior_options
        packet=copy.deepcopy(self.packet);packet['obs'][...,2]=10.
        options,parts=reactive_prior_options(packet,self.base,dict(full_gate=True,bias=-.05))
        self.assertEqual(options.shape,(1,33,4))
        for i in range(33):torch.testing.assert_close(options[:,i],options[:,parts[i,0]]+options[:,parts[i,1]],rtol=0.,atol=0.)
        self.assertTrue(bool(((options!=0.).sum(-1)<=2).all()))

    def test_prior_calibration_and_uncertainty_cancel_common_future_shocks(self):
        from identify_fusion_mechanisms import fit_prior_calibration,reactive_prior_scores
        count=24;prior=torch.randn(count,4,9)+10.;costs=torch.randn(count,16,9)+15.
        packet=dict(obs=torch.ones(count,4,14),time_remaining=torch.ones(count))
        data=dict(packet=packet,costs=costs,origin=torch.arange(count))
        coefficients,excluded=fit_prior_calibration(data,prior,seed=43,members=3)
        other,mask=fit_prior_calibration(dict(data,costs=costs+torch.randn(count,16,1)*100.),prior,seed=43,members=3)
        torch.testing.assert_close(coefficients,other,rtol=2e-4,atol=2e-5)
        torch.testing.assert_close(excluded,mask,rtol=0.,atol=0.)
        parts=torch.tensor([(3*(i//3),i%3) for i in range(9)])
        a=reactive_prior_scores(coefficients,prior,parts,1.)
        b=reactive_prior_scores(coefficients,prior+torch.randn(count,4,1)*100.,parts,1.)
        torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)

    def test_response_reliability_preserves_reference_and_additive_control_factors(self):
        from identify_fusion_mechanisms import reactive_reliability_estimates
        parts=torch.tensor([(3*(i//3),i%3) for i in range(9)])
        prior=torch.randn(3,4,9,dtype=torch.float64)
        packet=dict(obs=torch.ones(3,4,14,dtype=torch.float64),target_turn=torch.arange(3,dtype=torch.float64))
        coefficients=torch.randn(2,2,2,6,dtype=torch.float64,requires_grad=True)
        additive=reactive_reliability_estimates(coefficients,prior,parts,packet,False)
        joint=reactive_reliability_estimates(coefficients,prior,parts,packet,True)
        self.assertEqual(joint.shape,(2,3,4,9))
        torch.testing.assert_close(joint[...,0],torch.zeros_like(joint[...,0]),rtol=0.,atol=0.)
        torch.testing.assert_close(additive,additive[...,parts[:,0]]+additive[...,parts[:,1]],rtol=0.,atol=0.)
        self.assertGreater(float((joint-additive).abs().max()),.01)
        joint.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(coefficients.grad).all()))
        self.assertGreater(float(coefficients.grad[:,1].abs().sum()),0.)

    def test_reliability_supervision_ignores_common_real_return_shifts(self):
        from identify_fusion_mechanisms import fit_prior_reliability
        generator=torch.Generator().manual_seed(71039)
        prior=torch.randn(16,4,9,generator=generator,dtype=torch.float64)+10.
        costs=torch.randn(16,8,9,generator=generator,dtype=torch.float64)+15.
        packet=dict(obs=torch.ones(16,4,14,dtype=torch.float64),time_remaining=torch.ones(16,dtype=torch.float64))
        data=dict(packet=packet,costs=costs,origin=torch.arange(16))
        first,mask=fit_prior_reliability(data,prior,seed=43,members=1)
        shifted=dict(data,costs=costs+torch.randn(16,8,1,generator=generator,dtype=torch.float64)*100.)
        second,other=fit_prior_reliability(shifted,prior,seed=43,members=1)
        torch.testing.assert_close(first,second,rtol=2e-4,atol=2e-4)
        torch.testing.assert_close(mask,other,rtol=0.,atol=0.)

    def test_conditional_distribution_features_keep_joint_terms_identifiable(self):
        from identify_fusion_mechanisms import reactive_distribution_features,reactive_prior_scores
        generator=torch.Generator().manual_seed(813197)
        prior=torch.randn(3,4,9,generator=generator,dtype=torch.float64)
        parts=torch.tensor([(3*(i//3),i%3) for i in range(9)])
        packet=dict(obs=torch.ones(3,4,14,dtype=torch.float64),target_turn=torch.zeros(3,dtype=torch.float64))
        features=reactive_distribution_features(packet,prior,parts)
        self.assertEqual(features.shape,(3,9,2,20))
        torch.testing.assert_close(features[:,0],torch.zeros_like(features[:,0]),rtol=0.,atol=0.)
        weights=torch.randn(5,2,20,generator=generator,dtype=torch.float64)
        independent=reactive_prior_scores(weights,prior,parts,0.,packet,False)
        torch.testing.assert_close(independent,independent[:,parts[:,0]]+independent[:,parts[:,1]])
        joint=reactive_prior_scores(weights,prior,parts,0.,packet,True)
        self.assertGreater(float((joint-independent).abs().max()),.01)

    def test_cooperative_update_pairs_uncertainty_with_its_current_unary_decision(self):
        import identify_fusion_mechanisms as subject
        parts=torch.tensor([[0,0],[1,0],[2,0],[1,2]])
        prior=torch.tensor([[[10.,8.,9.,6.],[30.,28.,29.,26.],[12.,10.,11.,8.],[20.,18.,19.,16.]]])
        coefficients=torch.tensor([[1.,1.],[1.,.8],[1.,1.2]])
        score,reference=subject.reactive_cooperative_scores(coefficients,prior,parts,2.,self.packet)
        self.assertEqual(int(reference[0]),3)
        self.assertEqual(float(score[0,3]),0.)
        shifted=prior+torch.tensor([[[70.],[-20.],[10.],[5.]]])
        actual,_=subject.reactive_cooperative_scores(coefficients,shifted,parts,2.,self.packet)
        torch.testing.assert_close(actual,score,rtol=0.,atol=0.)
        independent,index=subject.reactive_cooperative_scores(coefficients,prior,parts,100.,self.packet,False)
        self.assertEqual(int(independent.argmin(-1)),int(index[0]))

    def test_fleet_factor_decoder_preserves_every_measured_pair_value(self):
        import identify_fusion_mechanisms as subject
        settings=dict(full_gate=True,bias=-.05,control_radius=30.)
        _,parts=subject.reactive_prior_options(self.packet,self.base,settings)
        energy=torch.randn(3,1,4,len(parts));energy[...,0]=0.
        fleet=subject.reactive_fleet_factors(energy,parts,4)
        self.assertEqual(fleet.shape,(3,1,4,81))
        for k,(left,right) in enumerate(parts.tolist()):
            index=0
            for single in (left,right):
                if single:index+=(1+(single-1)%2)*3**((single-1)//2)
            torch.testing.assert_close(fleet[...,index],energy[...,k])

    def test_robust_cooperative_update_averages_environment_futures_before_model_heads(self):
        import identify_fusion_mechanisms as subject
        estimate=torch.zeros(5,1,4,2)
        estimate[...,1]=torch.tensor([-1.,-1.,-1.,9.,9.])[:,None,None]
        independent=torch.zeros_like(estimate)
        ordinary,_=subject.reactive_paired_improvement(estimate,independent,0.)
        robust,_=subject.reactive_paired_improvement(estimate,independent,0.,robust=True)
        self.assertEqual(int(ordinary.argmin(-1)),0)
        self.assertEqual(int(robust.argmin(-1)),1)
        rare=torch.zeros(1,1,4,2);rare[0,0,:,1]=torch.tensor([-4.,-4.,-4.,20.])
        score,_=subject.reactive_paired_improvement(rare,torch.zeros_like(rare),0.,robust=True)
        self.assertEqual(int(score.argmin(-1)),0)
        shifted=estimate+torch.randn(5,1,4,1)*100.
        actual,_=subject.reactive_paired_improvement(shifted,independent,0.,robust=True)
        torch.testing.assert_close(actual,robust,atol=1e-5,rtol=1e-5)

    def test_fleet_factor_decoder_can_choose_cooperation_across_all_receivers(self):
        import identify_fusion_mechanisms as subject
        settings=dict(full_gate=True,bias=-.05,control_radius=30.)
        _,parts=subject.reactive_prior_options(self.packet,self.base,settings)
        singles=torch.tensor([0.,-1.,1.,-1.,1.,-1.,1.,-1.,1.])
        unary=singles[parts[:,0]]+singles[parts[:,1]]
        values=unary[None,None,None]
        fleet=subject.reactive_fleet_factors(values,parts,4)
        self.assertEqual(int(fleet.argmin(-1)),sum(3**i for i in range(4)))
        shifted=values+37.
        torch.testing.assert_close(subject.reactive_fleet_factors(shifted,parts,4),fleet)

    def test_monotone_joint_value_cannot_reward_consistently_harmful_responses(self):
        from identify_fusion_mechanisms import reactive_monotone_estimates,reactive_prior_options,reactive_subset
        generator=torch.Generator().manual_seed(717233)
        packet=dict(obs=torch.ones(2,4,14,dtype=torch.float64),target_turn=torch.tensor([.1,.5],dtype=torch.float64))
        _,parts=reactive_prior_options(packet,self.base.repeat(2,1,1),dict(full_gate=True,bias=-.05))
        prior=10.+torch.rand(2,4,33,generator=generator,dtype=torch.float64)*20.;prior[:,:,0]=10.
        parameters=torch.randn(3,6,4,generator=generator,dtype=torch.float64,requires_grad=True)
        value=reactive_monotone_estimates(parameters,prior,parts,packet)
        self.assertTrue(bool((value>=0.).all()))
        torch.testing.assert_close(value[...,0],torch.zeros_like(value[...,0]),rtol=0.,atol=0.)
        shifted=prior+torch.randn(2,4,1,generator=generator,dtype=torch.float64)*100.
        torch.testing.assert_close(value,reactive_monotone_estimates(parameters,shifted,parts,packet))
        additive=reactive_monotone_estimates(parameters,prior,parts,packet,False)
        torch.testing.assert_close(additive,additive[...,parts[:,0]]+additive[...,parts[:,1]])
        value.mean().backward()
        self.assertGreater(float(parameters.grad[:,5].abs().sum()),0.)
        subset=reactive_subset(dict(parts=parts,costs=prior),torch.tensor([1]))
        torch.testing.assert_close(subset['parts'],parts,rtol=0.,atol=0.)

    def test_continuation_update_identifies_control_changes_not_common_policy_returns(self):
        from identify_fusion_mechanisms import update_prior_continuation
        generator=torch.Generator().manual_seed(172991)
        packet=dict(obs=torch.ones(8,4,14,dtype=torch.float64),time_remaining=torch.ones(8,dtype=torch.float64))
        reference=15.+torch.randn(8,4,9,generator=generator,dtype=torch.float64)
        costs=reference+torch.randn(8,4,9,generator=generator,dtype=torch.float64)
        data=dict(packet=packet,costs=costs,reference_costs=reference,origin=torch.arange(8))
        prior=15.+torch.randn(8,4,9,generator=generator,dtype=torch.float64)
        anchor=dict(data=data,prior=prior);coefficients=torch.zeros(1,6,4,dtype=torch.float64)
        first,_=update_prior_continuation(coefficients,data,prior,anchor,seed=719)
        shifted=dict(data,costs=costs+torch.randn(8,4,1,generator=generator,dtype=torch.float64)*100.,
            reference_costs=reference+torch.randn(8,4,1,generator=generator,dtype=torch.float64)*100.)
        second,_=update_prior_continuation(coefficients,shifted,prior,anchor,seed=719)
        torch.testing.assert_close(first,second,rtol=2e-4,atol=2e-4)

    def test_joint_credit_cannot_rewrite_orthogonal_unary_controls(self):
        from identify_fusion_mechanisms import fit_prior_orthogonal
        generator=torch.Generator().manual_seed(177039)
        prior=15.+torch.randn(12,4,9,generator=generator,dtype=torch.float64)
        costs=15.+torch.randn(12,4,9,generator=generator,dtype=torch.float64)
        packet=dict(obs=torch.ones(12,4,14,dtype=torch.float64),time_remaining=torch.ones(12,dtype=torch.float64))
        data=dict(packet=packet,costs=costs,origin=torch.arange(12))
        full,_=fit_prior_orthogonal(data,prior,members=1,seed=971)
        additive,_=fit_prior_orthogonal(data,prior,interaction=False,members=1,seed=971)
        altered=costs.clone();altered[:,:,[4,5,7,8]]+=30.
        changed,_=fit_prior_orthogonal(dict(data,costs=altered),prior,members=1,seed=971)
        torch.testing.assert_close(full[:,:4],additive[:,:4],rtol=0.,atol=0.)
        torch.testing.assert_close(full[:,:4],changed[:,:4],rtol=0.,atol=0.)

    def test_active_conditional_credits_use_model_disagreement_without_reading_returns(self):
        import identify_fusion_mechanisms as subject
        packet={k:v.repeat(8,*([1]*(v.ndim-1))) for k,v in self.packet.items()}
        packet['obs'][...,2]=10.
        base=self.base.repeat(8,1,1);settings=dict(full_gate=True,bias=-.05)
        options,parts=subject.reactive_prior_options(packet,base,settings)
        data=dict(packet=packet,base=base,options=options,parts=parts,costs=torch.randn(8,4,33))
        prior=torch.randn(8,4,33)+10.
        saved=dict(calibration=torch.randn(3,7,16),settings=settings,interaction=True,risk=.25)
        selection,panel,design=subject.reactive_decision_credit_design(saved,data,prior,4)
        other,second,_=subject.reactive_decision_credit_design(saved,dict(data,costs=data['costs']*100.+37.),prior,4)
        torch.testing.assert_close(selection,other,rtol=0.,atol=0.)
        torch.testing.assert_close(panel,second,rtol=0.,atol=0.)
        torch.testing.assert_close(panel[:,0],design['deployed'],rtol=0.,atol=0.)
        for i,chosen in zip(selection,panel):
            self.assertEqual(len(torch.unique(options[i,chosen],dim=0)),4)
            self.assertTrue(bool((options[i,chosen].abs().sum(-1)==0.).any()))

    def test_active_credit_bootstrap_preserves_each_observed_control_in_every_head(self):
        from identify_fusion_mechanisms import reactive_positive_credit_weights
        groups=torch.tensor([0,0,1,2,2,2])
        for seed in range(5):
            rows,futures=reactive_positive_credit_weights(groups,4,torch.Generator().manual_seed(seed))
            self.assertTrue(bool((rows>0.).all()))
            self.assertTrue(bool((futures>0.).all()))
            self.assertAlmostEqual(float(rows.mean()),1.)
            torch.testing.assert_close(futures.sum(-1),torch.ones(6,dtype=torch.float64))
            self.assertEqual(float(rows[0]),float(rows[1]))

    def test_local_candidate_field_reads_current_public_interactions_and_preserves_old_weights(self):
        import identify_fusion_mechanisms as subject
        packet={k:v.repeat(8,*([1]*(v.ndim-1))) for k,v in self.packet.items()}
        packet['obs'][...,2]=10.;packet['time_remaining']=torch.ones(8)
        packet['target_heading']=packet['motion'][:,0,2]+packet['obs'][:,0,5]
        base=self.base.repeat(8,1,1)
        options,parts=subject.reactive_prior_options(packet,base,dict(full_gate=True,bias=-.05))
        prior=15.+torch.randn(8,4,33)
        data=dict(packet=packet,base=base,options=options,parts=parts,origin=torch.arange(8),
            evaluation_options=torch.tensor([[12,3,1,21]]).repeat(8,1),
            costs=15.+torch.randn(8,4,4),reference_costs=15.+torch.randn(8,4,4))
        anchor=dict(data=dict(data,costs=prior),prior=prior)
        spec=subject.reactive_coordination_dictionary(data,prior,anchor)
        encoded=subject.reactive_local_candidate_data(data,prior,spec)
        altered=dict(data,packet=dict(packet,central=packet['central']+1000.))
        unchanged=subject.reactive_local_candidate_data(altered,prior+torch.randn(8,4,1)*30.,spec)
        torch.testing.assert_close(encoded['packet']['candidate_interactions'],unchanged['packet']['candidate_interactions'],atol=3e-5,rtol=3e-5)
        calibrated_anchor=dict(anchor,data=subject.reactive_local_candidate_data(anchor['data'],prior,spec))
        coefficients=torch.zeros(1,7,16+len(spec['centers']))
        fitted,_=subject.update_prior_continuation(coefficients,encoded,prior,calibrated_anchor,seed=829,
            joint_only=True,local_start=16)
        torch.testing.assert_close(fitted[...,:16],coefficients[...,:16],atol=0.,rtol=0.)
        self.assertGreater(float(fitted[:,4,16:].abs().sum()),0.)
        expanded_spec=dict(blocks=[spec,dict(spec,bandwidth=2.*spec['bandwidth'])])
        expanded=subject.reactive_local_candidate_data(data,prior,expanded_spec)
        padded=torch.nn.functional.pad(fitted,(0,len(spec['centers'])))
        old=subject.reactive_monotone_estimates(fitted,prior,parts,encoded['packet'])
        new=subject.reactive_monotone_estimates(padded,prior,parts,expanded['packet'])
        torch.testing.assert_close(old,new,atol=1e-6,rtol=1e-6)

    def test_conditional_joint_residual_can_learn_effects_missing_from_the_response_prior(self):
        import identify_fusion_mechanisms as subject
        _,parts=subject.reactive_prior_options(self.packet,self.base,dict(full_gate=True,bias=-.05))
        prior=torch.ones(1,4,33)*10.
        features=torch.zeros(1,33,13);features[...,10]=2.;features[...,12]=1.
        packet=dict(self.packet,candidate_interactions=features)
        coefficients=torch.zeros(1,8,17);coefficients[:,7,16]=-2.
        energy=subject.reactive_monotone_estimates(coefficients,prior,parts,packet)
        joint=(parts>0).all(-1)
        torch.testing.assert_close(energy[...,joint],torch.full_like(energy[...,joint],-2.))
        torch.testing.assert_close(energy[...,~joint],torch.zeros_like(energy[...,~joint]))
        disabled=subject.reactive_monotone_estimates(coefficients,prior,parts,packet,False)
        torch.testing.assert_close(disabled,torch.zeros_like(disabled))
        packet['candidate_interactions'][...,10]=0.
        identical=subject.reactive_monotone_estimates(coefficients,prior,parts,packet)
        torch.testing.assert_close(identical,torch.zeros_like(identical))

    def test_active_continuation_update_matches_evaluated_actions_and_freezes_unary_field(self):
        import identify_fusion_mechanisms as subject
        packet={k:v.repeat(8,*([1]*(v.ndim-1))) for k,v in self.packet.items()}
        packet['obs'][...,2]=10.;packet['time_remaining']=torch.ones(8)
        base=self.base.repeat(8,1,1)
        options,parts=subject.reactive_prior_options(packet,base,dict(full_gate=True,bias=-.05))
        prior=15.+torch.randn(8,4,33)
        reference=15.+torch.randn(8,4,4);costs=reference+torch.randn(8,4,4)
        data=dict(packet=packet,base=base,options=options,parts=parts,origin=torch.arange(8),
            evaluation_options=torch.tensor([[12,3,1,21]]).repeat(8,1),costs=costs,reference_costs=reference)
        anchor=dict(data=dict(data,costs=prior),prior=prior)
        coefficients=torch.zeros(1,7,16)
        after,metrics=subject.update_prior_continuation(coefficients,data,prior,anchor,seed=791,joint_only=True)
        torch.testing.assert_close(after[:,:4],coefficients[:,:4],rtol=0.,atol=0.)
        self.assertGreater(metrics['control_value_change_rms'][0],0.)
        shifted=dict(data,costs=costs+torch.randn(8,4,1)*100.,reference_costs=reference+torch.randn(8,4,1)*100.)
        invariant,_=subject.update_prior_continuation(coefficients,shifted,prior,anchor,seed=791,joint_only=True)
        torch.testing.assert_close(after,invariant,rtol=3e-3,atol=3e-3)
        changed_reference=dict(data,reference_costs=reference+torch.randn_like(reference)*30.)
        actual,_=subject.update_prior_continuation(coefficients,changed_reference,prior,anchor,seed=791,joint_only=True)
        torch.testing.assert_close(after,actual,rtol=0.,atol=0.)

    def test_candidate_messages_are_public_symmetric_and_use_current_controls(self):
        from identify_fusion_mechanisms import reactive_prior_options,reactive_candidate_interactions
        programs,parts=reactive_prior_options(self.packet,self.base,dict(full_gate=True,bias=-.05,control_radius=1000.))
        features=reactive_candidate_interactions(self.packet,self.base,programs,parts)
        self.assertEqual(features.shape,(1,33,12))
        swapped=reactive_candidate_interactions(self.packet,self.base,programs,parts.flip(-1))
        torch.testing.assert_close(features,swapped,rtol=0.,atol=0.)
        altered=dict(self.packet,central=torch.randn_like(self.packet['central'])*100.)
        torch.testing.assert_close(features,reactive_candidate_interactions(altered,self.base,programs,parts),rtol=0.,atol=0.)
        base=self.base.clone();base[:,1,0]=-.6
        changed=reactive_candidate_interactions(self.packet,base,programs,parts)
        self.assertGreater(float((features-changed).abs().max()),.01)
        torch.testing.assert_close(features[:,:9],torch.zeros_like(features[:,:9]),rtol=0.,atol=0.)

    def test_joint_surface_preserves_mixed_choices_and_additive_ablation(self):
        from policy.interaction_sac import CandidateCoordinationGate
        from identify_fusion_mechanisms import reactive_options
        options=reactive_options(self.packet)
        for enabled in (True,False):
            model=CandidateCoordinationGate(16,enabled)
            with torch.no_grad():model.mixed[-1].bias.fill_(1.)
            _,q=model(self.packet,self.base,options)
            # Cartesian order: (0,0),(0,-1),(0,1),(-1,0),(-1,-1),...
            mixed=q[0,8]-q[0,6]-q[0,2]+q[0,0]
            self.assertGreater(float(mixed),1.) if enabled else self.assertEqual(float(mixed),0.)
            self.assertEqual(float(q[0,0]),0.)

    def test_actor_does_not_read_privileged_central_state(self):
        from policy.interaction_sac import CandidateCoordinationGate
        from identify_fusion_mechanisms import reactive_options
        model=CandidateCoordinationGate(16);options=reactive_options(self.packet)
        with torch.no_grad():
            model.unary[-1].weight.normal_();model.mixed[-1].weight.normal_()
        expected=model.log_policy(self.packet,self.base,options)
        altered=dict(self.packet,central=torch.randn_like(self.packet['central'])*100.)
        torch.testing.assert_close(expected,model.log_policy(altered,self.base,options),rtol=0.,atol=0.)

    def test_conditional_supervision_cancels_state_only_return_and_baseline_gradient(self):
        from policy.interaction_sac import CandidateCoordinationGate
        from identify_fusion_mechanisms import reactive_options,reactive_precision,reactive_value_loss
        model=CandidateCoordinationGate(16);costs=torch.randn(1,4,9)*2.+15.
        data=dict(packet=self.packet,base=self.base,options=reactive_options(self.packet),costs=costs,precision=reactive_precision(costs))
        loss=reactive_value_loss(model,data,True)-reactive_value_loss(model,data,False)
        shifted=dict(data,costs=costs+torch.arange(4.)[None,:,None]*10.)
        other=reactive_value_loss(model,shifted,True)-reactive_value_loss(model,shifted,False)
        torch.testing.assert_close(loss,other,rtol=1e-5,atol=1e-5)
        loss.backward()
        self.assertTrue(all(p.grad is None or float(p.grad.abs().max())==0. for p in model.baseline.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.unary.parameters() if p.grad is not None),0.)

    def test_paired_collection_changes_only_first_option_and_shares_futures(self):
        import identify_fusion_mechanisms as subject
        from policy.interaction_sac import CandidateCoordinationGate
        actor=CandidateCoordinationGate(16)
        def source(obs,*unused):return self.base.expand(len(obs)//4,-1,-1).flatten(0,1),None
        def rollout(actor,source,packet,noise,settings,first=None,**kwargs):
            self.assertEqual(noise.shape,(27,300,17))
            grouped=noise.reshape(1,3,9,300,17)
            torch.testing.assert_close(grouped,grouped[:,:,0:1].expand_as(grouped),rtol=0.,atol=0.)
            torch.testing.assert_close(first[:9],subject.reactive_options(self.packet)[0])
            return first.square().sum(-1),torch.full((27,),3),None
        with patch.object(subject,'reactive_rollout',side_effect=rollout):
            data=subject.reactive_credits(actor,source,self.packet,{},3,torch.Generator().manual_seed(2),1)
        self.assertEqual(data['costs'].shape,(1,3,9))

    def test_critic_improvement_updates_joint_policy(self):
        import identify_fusion_mechanisms as subject
        from policy.interaction_sac import CandidateCoordinationGate
        actor=CandidateCoordinationGate(16);critic=copy.deepcopy(actor)
        options=subject.reactive_options(self.packet)
        costs=torch.full((1,4,9),15.);costs[:,:,8]=5.
        data=dict(packet=self.packet,base=self.base,options=options,costs=costs,outcomes=torch.full((1,4,9),3))
        settings=dict(temperature=1.,prior=.3,improvement=.3)
        old=actor.log_policy(self.packet,self.base,options)
        result=subject.reactive_update(actor,critic,data,True,settings,180,33)
        new=actor.log_policy(self.packet,self.base,options)
        self.assertGreater(float(new[0,8]),float(old[0,8]))
        self.assertEqual(int(new.argmax(-1)),8)
        self.assertGreater(result['policy_kl'],.01)

    def test_vector_physics_matches_native_cbf_and_random_current_order(self):
        from identify_fusion_mechanisms import reactive_physics_step
        seed_random(981736)
        env=TADEnv(4,protocol='paper-parameters-v1');env.reset(2.25,noisy_agility=False)
        boats=[*env.defender_list,env.attacker]
        state=torch.tensor([[[b.x,b.y,b.theta,*b.velocity_r] for b in boats]],dtype=torch.float64)
        measured=torch.tensor([[b.velocity for b in boats]],dtype=torch.float64)
        agility=torch.tensor([2.25],dtype=torch.float64)
        for step in range(40):
            noise=np.random.RandomState(97531+step).normal(size=17)
            action=np.random.default_rng(step).uniform(-.8,.8,(4,3))
            action[:,2]=.1+.8*np.random.default_rng(step).uniform(size=4)
            np.random.seed(97531+step)
            execute(env,public_packet(env),action)
            state,measured=reactive_physics_step(state,measured,agility,torch.tensor(action)[None],torch.tensor(noise)[None])
            expected=np.array([[b.x,b.y,b.theta,*b.velocity_r] for b in boats])
            np.testing.assert_allclose(state.numpy()[0],expected,atol=1e-8,rtol=1e-8)

    def test_exact_mirror_update_equals_conditional_improvement_distribution(self):
        from policy.interaction_sac import CandidateCoordinationGate,MirrorCoordinationGate
        from identify_fusion_mechanisms import reactive_options
        actor=MirrorCoordinationGate(16);critic=CandidateCoordinationGate(16)
        with torch.no_grad():
            critic.unary[-1].weight.normal_();critic.mixed[-1].weight.normal_()
        options=reactive_options(self.packet)
        old=actor.log_policy(self.packet,self.base,options,reference_prior=.3)
        _,q=critic(self.packet,self.base,options)
        expected=(old-.3*q).log_softmax(-1)
        actor.add_value(critic,.3)
        torch.testing.assert_close(actor.log_policy(self.packet,self.base,options,reference_prior=.3),expected,atol=1e-6,rtol=1e-6)
        with torch.no_grad():critic.unary[-1].bias.fill_(300.)
        torch.testing.assert_close(actor.log_policy(self.packet,self.base,options,reference_prior=.3),expected,atol=1e-6,rtol=1e-6)

    def test_orthogonal_control_gradient_ignores_nuisance_state_value(self):
        from policy.interaction_sac import CandidateCoordinationGate
        from identify_fusion_mechanisms import reactive_options,reactive_precision,reactive_value_loss
        model=CandidateCoordinationGate(16);costs=torch.randn(1,4,9)*2.+15.
        data=dict(packet=self.packet,base=self.base,options=reactive_options(self.packet),costs=costs,precision=reactive_precision(costs))
        reactive_value_loss(model,data,True,True).backward()
        before=model.unary[-1].bias.grad.clone();model.zero_grad(set_to_none=True)
        with torch.no_grad():model.baseline[-1].bias.add_(100.)
        reactive_value_loss(model,data,True,True).backward()
        torch.testing.assert_close(before,model.unary[-1].bias.grad,rtol=0.,atol=0.)

    def test_measured_target_pose_and_clock_do_not_read_private_state(self):
        from policy.interaction_sac import CandidateCoordinationGate
        from identify_fusion_mechanisms import reactive_options
        packet=dict(self.packet,target_heading=torch.tensor([1.]),time_remaining=torch.tensor([.8]))
        model=CandidateCoordinationGate(16,True,True);options=reactive_options(packet)
        with torch.no_grad():model.unary[-1].weight.normal_();model.mixed[-1].weight.normal_()
        original=model(packet,self.base,options)[1]
        private=dict(packet,central=torch.randn_like(packet['central'])*100.)
        torch.testing.assert_close(original,model(private,self.base,options)[1],rtol=0.,atol=0.)
        public=dict(packet,target_heading=torch.tensor([-1.]),time_remaining=torch.tensor([.2]))
        self.assertGreater(float((original-model(public,self.base,options)[1]).abs().max()),1e-5)

    def test_ensemble_uncertainty_is_anchored_and_checkpoint_restores(self):
        from policy.interaction_sac import EnsembleCoordinationValue,MirrorCoordinationGate
        from identify_fusion_mechanisms import reactive_options
        value=EnsembleCoordinationValue(16,members=3,risk=1.)
        with torch.no_grad():
            for index,member in enumerate(value.members):member.unary[-1].bias[0]=index-1.
        options=reactive_options(self.packet)
        _,energy=value(self.packet,self.base,options)
        self.assertEqual(float(energy[0,0]),0.)
        self.assertTrue(bool((energy>=0.).all()))
        with self.assertRaises(ValueError):MirrorCoordinationGate(16,members=2).add_value(value,.3)
        actor=MirrorCoordinationGate(16,members=3,risk=1.);actor.add_value(value,.3)
        restored=MirrorCoordinationGate(16,increments=1,members=3,risk=.5);restored.load_state_dict(actor.state_dict())
        self.assertEqual(restored.blocks[0].risk,1.)
        torch.testing.assert_close(actor.log_policy(self.packet,self.base,options),restored.log_policy(self.packet,self.base,options),rtol=0.,atol=0.)
        legacy={key:value for key,value in actor.state_dict().items() if not key.endswith('risk_scale')}
        restored=MirrorCoordinationGate(16,increments=1,members=3,risk=1.);restored.load_state_dict(legacy)
        torch.testing.assert_close(actor.log_policy(self.packet,self.base,options),restored.log_policy(self.packet,self.base,options),rtol=0.,atol=0.)

    def test_bootstrap_keeps_different_times_of_one_source_trajectory_together(self):
        from identify_fusion_mechanisms import reactive_state_groups,reactive_group_sample
        data=dict(origin=torch.tensor([11,24,11,37,24,37]))
        groups=reactive_state_groups(data)
        self.assertEqual(int(groups[0]),int(groups[2]))
        self.assertNotEqual(int(groups[0]),int(groups[1]))
        for seed in range(5):
            selected,excluded=reactive_group_sample(groups,torch.Generator().manual_seed(seed))
            for origin in data['origin'].unique():
                members=data['origin']==origin
                self.assertEqual(int(excluded[members].unique().numel()),1)
                if excluded[members][0]:self.assertFalse(bool(members[selected].any()))

    def test_reserved_roots_are_disjoint_and_do_not_depend_on_optimizer_rng(self):
        from identify_fusion_mechanisms import reactive_root_partition
        fit,check=reactive_root_partition(512,128)
        self.assertEqual((len(fit),len(check)),(384,128))
        self.assertFalse(bool(torch.isin(fit,check).any()))
        self.assertEqual(len(torch.cat((fit,check)).unique()),512)
        torch.manual_seed(991734);torch.rand(83)
        fit2,check2=reactive_root_partition(512,128)
        torch.testing.assert_close(fit,fit2);torch.testing.assert_close(check,check2)
        with self.assertRaises(ValueError):reactive_root_partition(32,32)

    def test_legacy_trajectory_origin_recovery_requires_exact_public_states(self):
        import identify_fusion_mechanisms as subject
        training={k:v.repeat(3,*([1]*(v.ndim-1))) for k,v in self.packet.items()}
        trajectory=dict(packet={k:v.repeat(5,*([1]*(v.ndim-1))) for k,v in self.packet.items()},
            base=self.base.repeat(5,1,1),origin=torch.tensor([0,1,2,0,2]))
        trajectory['packet']['obs'][:,0,0]=torch.arange(5.)
        generator=torch.Generator().manual_seed(819+1000000)
        selection=torch.randperm(3,generator=generator)
        torch.randn(3,300,17,generator=generator)
        indices=torch.randperm(5,generator=generator)[:3]
        data=dict(packet={k:v[indices].clone() for k,v in trajectory['packet'].items()},base=trajectory['base'][indices])
        with patch.object(subject,'reactive_rollout',return_value=(None,None,trajectory)):
            result=subject.reactive_recover_origins(data,training,None,dict(defenders=4),3,819)
            torch.testing.assert_close(result['origin'],selection[trajectory['origin'][indices]])
            data['packet']['obs'][0,0,0]+=.01
            with self.assertRaises(ValueError):subject.reactive_recover_origins(data,training,None,dict(defenders=4),3,819)

    def test_bootstrap_excludes_all_interventions_at_the_same_state(self):
        from identify_fusion_mechanisms import reactive_state_groups,reactive_group_sample
        packet={k:v.repeat_interleave(8,0) for k,v in self.packet.items()}
        packet['central'][4:]+=1.
        groups=reactive_state_groups(dict(packet=packet))
        self.assertEqual(int(groups.unique().numel()),2)
        index,excluded=reactive_group_sample(groups,torch.Generator().manual_seed(3))
        self.assertFalse(bool(excluded[index].any()))
        for group in groups.unique():self.assertEqual(int(excluded[groups==group].unique().numel()),1)

    def test_full_gate_can_choose_each_candidate_and_interior_fusion(self):
        from identify_fusion_mechanisms import reactive_options,reactive_reference,reactive_action
        packet=copy.deepcopy(self.packet);packet['obs'][...,2]=10.
        settings=dict(full_gate=True,bias=-.05,radius=.3)
        reference=reactive_reference(self.base,settings)
        programs=reactive_options(packet,True,gate_levels=5,reference=reference)
        self.assertEqual(programs.shape,(1,625,4))
        actual=reactive_action(self.base[:,None].expand(-1,625,-1,-1),programs,settings)[...,2]
        torch.testing.assert_close(actual[:,0],reference)
        for receiver in range(4):
            expected=torch.stack((reference[0,receiver],torch.tensor(0.),torch.tensor(1.),reference[0,receiver]/2,(reference[0,receiver]+1)/2)).sort().values
            torch.testing.assert_close(actual[0,:,receiver].unique().sort().values,expected)

    def test_out_of_bag_risk_uses_control_cost_and_independent_state_groups(self):
        from identify_fusion_mechanisms import reactive_calibrate_risk
        class Critic:
            risk=.5
            def estimates(self,packet,base,options):
                energy=torch.ones(5,64,9);energy[:,:,0]=0.;energy[:,:,2]=-1.
                energy[:,:,1]=torch.tensor([-15.,-10.,-5.,0.,5.])[:,None]
                return torch.zeros(5,64),energy
        data=dict(packet={},base=None,options=None,costs=torch.full((64,4,9),10.))
        data['costs'][:,:,1]=12.;data['costs'][:,:,2]=8.
        critic=Critic()
        result=reactive_calibrate_risk(critic,data,[torch.ones(64,dtype=torch.bool)]*5,
            torch.zeros(64,9),dict(improvement=.3),torch.arange(64)//2)
        self.assertEqual(critic.risk,2.)
        self.assertEqual(result['eligible_states'],32)
        self.assertEqual(result['paired_gains'][0],2.)
        self.assertEqual(result['paired_gains'][-1],-2.)

    def test_step_search_can_expand_a_beneficial_policy_increment(self):
        import identify_fusion_mechanisms as subject
        from policy.interaction_sac import CandidateCoordinationGate,MirrorCoordinationGate
        actor=MirrorCoordinationGate(16);actor.add_value(CandidateCoordinationGate(16),.3)
        def rollout(actor,source,packet,noise,settings,**kwargs):
            cost=torch.ones(len(packet['obs']))*(10.+(float(actor.coefficients[-1])-1.2)**2)
            return cost,torch.full_like(cost,3,dtype=torch.long),None
        with patch.object(subject,'reactive_rollout',side_effect=rollout):
            result=subject.reactive_accept_step(actor,None,self.packet,dict(expand_step=True),torch.Generator().manual_seed(5),repeats=2)
        self.assertEqual(result['selected_scale'],4.)
        self.assertAlmostEqual(float(actor.coefficients[0]),1.2,places=6)

    def test_rejected_whole_policy_step_restores_previous_gate(self):
        import identify_fusion_mechanisms as subject
        from policy.interaction_sac import CandidateCoordinationGate,MirrorCoordinationGate
        actor=MirrorCoordinationGate(16);critic=CandidateCoordinationGate(16)
        options=subject.reactive_options(self.packet);expected=actor.log_policy(self.packet,self.base,options)
        actor.add_value(critic,.6)
        def rollout(actor,source,packet,noise,settings,**kwargs):
            cost=torch.ones(len(packet['obs']))*(10.+float(actor.coefficients[-1]))
            return cost,torch.full_like(cost,3,dtype=torch.long),None
        with patch.object(subject,'reactive_rollout',side_effect=rollout):
            result=subject.reactive_accept_step(actor,None,self.packet,{},torch.Generator().manual_seed(5),repeats=2)
        self.assertEqual(result['selected_scale'],0.)
        self.assertEqual(len(actor.blocks),0)
        torch.testing.assert_close(expected,actor.log_policy(self.packet,self.base,options),rtol=0.,atol=0.)

    def test_deployed_gate_redecides_at_every_option_boundary(self):
        import identify_fusion_mechanisms as subject
        policy=subject.ReactiveGatePolicy.__new__(subject.ReactiveGatePolicy)
        policy.settings=dict(trigger=20.,period=2,radius=.3,bias=-.05,temperature=1.,prior=.3)
        policy.probe='aligned';policy.source=lambda obs:self.base[0].numpy().copy();policy.reset()
        class Gate:
            def __init__(self):self.calls=0
            def log_policy(self,packet,base,options,*args):
                result=torch.full((1,9),-100.);result[0,1 if self.calls%2==0 else 2]=0.;self.calls+=1
                return result
        policy.model=Gate();p={k:v[0].numpy().copy() for k,v in self.packet.items()}
        p['obs'][:,2]=[8.,10.,20.,25.];p['time_remaining']=np.asarray(.8,np.float32);p.pop('central')
        actions=[policy(p) for _ in range(5)]
        self.assertEqual(len(policy.decisions),3)
        np.testing.assert_array_equal(actions[0],actions[1]);np.testing.assert_array_equal(actions[2],actions[3])
        self.assertGreater(float(np.abs(actions[1]-actions[2]).max()),.1)

    def test_complete_pair_factors_see_shared_closing_velocity(self):
        from policy.interaction_sac import CandidateCoordinationGate
        model=CandidateCoordinationGate(16,True,False,True)
        with torch.no_grad():model.mixed[-1].weight.normal_()
        expected=model.factors(self.packet,self.base)[2]
        changed=copy.deepcopy(self.packet);changed['motion'][...,3]+=2.
        actual=model.factors(changed,self.base)[2]
        self.assertGreater(float((actual-expected).abs().max()),1e-5)

    def test_fleet_credits_hold_every_nonintervened_peer_choice(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.packet);packet['obs'][...,2]=10.
        class Policy:
            def log_policy(self,p,b,options,*args):
                result=torch.full(options.shape[:2],-100.);result[:,40]=0.;return result
        def source(obs,*unused):return self.base.expand(len(obs)//4,-1,-1).flatten(0,1),None
        def rollout(actor,source,packet,noise,settings,first=None,**kwargs):
            programs=first.reshape(1,3,9,4)[0,0]
            changed=(programs!=-1.).any(0)
            self.assertEqual(int(changed.sum()),2)
            self.assertTrue(bool((programs[:,~changed]==-1.).all()))
            torch.testing.assert_close(noise.reshape(1,3,9,300,17),noise.reshape(1,3,9,300,17)[:,:,0:1].expand(1,3,9,300,17))
            return first.square().sum(-1),torch.full((27,),3),None
        with patch.object(subject,'reactive_rollout',side_effect=rollout):
            data=subject.reactive_credits(Policy(),source,packet,dict(all_gates=True,temperature=1.,prior=.3),3,torch.Generator().manual_seed(5),1)
        self.assertEqual(subject.reactive_options(packet,True).shape,(1,81,4))
        self.assertEqual(data['options'].shape,(1,9,4))

    def test_complete_pair_credits_cover_every_executed_pair_with_common_futures(self):
        import identify_fusion_mechanisms as subject
        packet=copy.deepcopy(self.packet);packet['obs'][...,2]=10.
        held=torch.tensor([[.1,-.1,.2,-.2]])
        def source(obs,*unused):return self.base.expand(len(obs)//4,-1,-1).flatten(0,1),None
        def rollout(actor,source,packet,noise,settings,first=None,**kwargs):
            paired=noise.reshape(1,3,33,300,17)
            torch.testing.assert_close(paired,paired[:,:,0:1].expand_as(paired),rtol=0.,atol=0.)
            return first.square().sum(-1),torch.full((99,),3),None
        settings=dict(all_gates=True,full_gate=True,bias=-.05,conditional_policy=True)
        with patch.object(subject,'reactive_decision',return_value=(torch.zeros(1,dtype=torch.long),held)),patch.object(subject,'reactive_rollout',side_effect=rollout):
            data=subject.reactive_credits(None,source,packet,settings,3,torch.Generator().manual_seed(5),1,True)
        options=data['options'][0]
        torch.testing.assert_close(options[0],held[0],rtol=0.,atol=0.)
        changed=options!=held[0]
        self.assertEqual(int((changed.sum(-1)==1).sum()),8)
        self.assertEqual(int((changed.sum(-1)==2).sum()),24)
        for i in range(4):
            for j in range(i+1,4):self.assertEqual(int((changed[:,i]&changed[:,j]).sum()),4)

    def test_full_fleet_update_preserves_conditional_mirror_equation(self):
        from policy.interaction_sac import CandidateCoordinationGate,MirrorCoordinationGate
        from identify_fusion_mechanisms import reactive_options
        actor=MirrorCoordinationGate(16,complete_state=True)
        critic=CandidateCoordinationGate(16,complete_state=True)
        with torch.no_grad():critic.unary[-1].weight.normal_();critic.mixed[-1].weight.normal_()
        packet=copy.deepcopy(self.packet);packet['obs'][...,2]=10.
        all_options=reactive_options(packet,True)
        conditional=all_options[:,(all_options[0,:,2:]==-1.).all(-1)]
        self.assertEqual(conditional.shape,(1,9,4))
        old=actor.log_policy(packet,self.base,conditional);_,q=critic(packet,self.base,conditional)
        actor.add_value(critic,.3)
        torch.testing.assert_close(actor.log_policy(packet,self.base,conditional),(old-.3*q).log_softmax(-1),rtol=1e-6,atol=1e-6)

    def test_response_encoding_is_public_and_respects_candidate_ablation(self):
        from policy.interaction_sac import CandidateCoordinationGate
        from identify_fusion_mechanisms import reactive_options
        packet=dict(self.packet,target_heading=torch.tensor([1.]),time_remaining=torch.tensor([.8]))
        options=reactive_options(packet);changed=self.base.clone();changed[:,1,0]-=.7
        for enabled in (True,False):
            model=CandidateCoordinationGate(16,enabled,True,True,True)
            with torch.no_grad():model.unary[-1].weight.normal_();model.mixed[-1].weight.normal_()
            original=model(packet,self.base,options)[1]
            altered=dict(packet,central=torch.randn_like(packet['central'])*100.)
            torch.testing.assert_close(original,model(altered,self.base,options)[1],rtol=0.,atol=0.)
            unary=model.factors(packet,self.base)[1][:,0]
            other=model.factors(packet,changed)[1][:,0]
            if enabled:self.assertGreater(float((unary-other).abs().max()),1e-5)
            else:torch.testing.assert_close(unary,other,rtol=0.,atol=0.)

    def test_nominal_responses_use_consistent_rotating_coordinate_frames(self):
        from policy.interaction_sac import CandidateCoordinationGate
        packet=copy.deepcopy(self.packet);motion=packet['motion']
        motion[...,3:5]=torch.tensor([1.,.3]);motion[...,5]=.1
        response,paths=CandidateCoordinationGate.candidate_responses(packet,self.base)
        angle=torch.tensor(.71);c,s=angle.cos(),angle.sin()
        rotation=torch.stack((torch.stack((c,-s)),torch.stack((s,c))))
        rotated=copy.deepcopy(packet)
        rotated['motion'][...,:2]=motion[...,:2]@rotation.T
        rotated['motion'][...,3:5]=motion[...,3:5]@rotation.T
        rotated['motion'][...,2]+=angle
        actual,actual_paths=CandidateCoordinationGate.candidate_responses(rotated,self.base)
        torch.testing.assert_close(response,actual,atol=2e-6,rtol=2e-6)
        torch.testing.assert_close(paths@rotation.T,actual_paths,atol=2e-5,rtol=2e-5)

    def test_pair_synergy_is_conditioned_on_remaining_fleet_state(self):
        from policy.interaction_sac import CandidateCoordinationGate
        model=CandidateCoordinationGate(16,complete_state=True,response_encoding=True)
        with torch.no_grad():model.mixed[-1].weight.normal_()
        original=model.factors(self.packet,self.base)[2][:,0,1]
        changed=copy.deepcopy(self.packet);changed['motion'][:,2,3]+=3.
        actual=model.factors(changed,self.base)[2][:,0,1]
        self.assertGreater(float((original-actual).abs().max()),1e-6)

    def test_joint_evasion_features_are_public_rotation_consistent_and_nonadditive(self):
        from policy.interaction_sac import CandidateCoordinationGate
        packet=copy.deepcopy(self.packet)
        packet['motion'][0,:,:2]=torch.tensor([[15.,10.],[20.,20.],[35.,-5.],[50.,10.]])
        packet['motion'][...,2]=0.;packet['motion'][...,3:5]=torch.tensor([1.,.3]);packet['motion'][...,5]=0.
        target=torch.tensor([30.,10.])-packet['motion'][...,:2]
        packet['obs'][...,2]=target.norm(dim=-1);packet['obs'][...,3]=torch.atan2(target[...,1],target[...,0])
        packet['obs'][...,4]=2.;packet['obs'][...,5]=.2
        packet['obs'][...,12:14]=torch.tensor([.7,-.3])
        response,paths=CandidateCoordinationGate.candidate_responses(packet,self.base)
        expected=CandidateCoordinationGate.candidate_reactions(packet,paths)
        self.assertEqual(expected.shape,(1,4,4,24))
        self.assertTrue(bool(torch.isfinite(expected).all()))
        self.assertGreater(float(expected.abs().max()),1e-7)
        altered=dict(packet,central=torch.randn_like(packet['central'])*500.)
        torch.testing.assert_close(expected,CandidateCoordinationGate.candidate_reactions(altered,paths),rtol=0.,atol=0.)
        angle=torch.tensor(.73);c,s=angle.cos(),angle.sin()
        rotation=torch.stack((torch.stack((c,-s)),torch.stack((s,c))))
        rotated=copy.deepcopy(packet)
        for columns in (slice(0,2),slice(3,5)):
            rotated['motion'][...,columns]=packet['motion'][...,columns]@rotation.T
        rotated['motion'][...,2]+=angle
        torch.testing.assert_close(expected,CandidateCoordinationGate.candidate_reactions(rotated,paths@rotation.T),atol=2e-5,rtol=2e-5)

    def test_joint_evasion_ablation_removes_incoming_candidate_information(self):
        from policy.interaction_sac import CandidateCoordinationGate,EnsembleCoordinationValue,MirrorCoordinationGate
        model=CandidateCoordinationGate(16,False,False,True,True,True,-.05,True)
        with torch.no_grad():model.unary[-1].weight.normal_();model.mixed[-1].weight.normal_()
        altered=self.base.clone();altered[:,1,0]-=.7
        old=model.factors(self.packet,self.base);new=model.factors(self.packet,altered)
        torch.testing.assert_close(old[1][:,0],new[1][:,0],atol=0.,rtol=0.)
        self.assertEqual(float(old[2].abs().max()),0.)
        critic=EnsembleCoordinationValue(16,True,3,.5,False,True,True,True,-.05,True)
        actor=MirrorCoordinationGate(16,True,members=3,complete_state=True,response_encoding=True,cardinal_gates=True,reaction_encoding=True)
        with torch.no_grad():
            for member in critic.members:member.mixed[-1].weight.normal_()
        actor.add_value(critic,.3)
        from identify_fusion_mechanisms import reactive_options
        options=reactive_options(self.packet)
        torch.testing.assert_close(actor(self.packet,self.base,options)[1],
            actor.prepared_energy(actor.prepare_factors(self.packet,self.base),options),atol=1e-6,rtol=1e-6)

    def test_decision_aligned_credit_is_invariant_to_common_return_shift(self):
        from policy.interaction_sac import CandidateCoordinationGate
        from identify_fusion_mechanisms import reactive_options,reactive_precision,reactive_value_loss
        model=CandidateCoordinationGate(16);costs=torch.randn(1,8,9)+15.
        data=dict(packet=self.packet,base=self.base,options=reactive_options(self.packet),costs=costs,
            precision=reactive_precision(costs),old_log_policy=torch.zeros(1,9).log_softmax(-1))
        reactive_value_loss(model,data,True,True,.3).backward()
        expected=model.unary[-1].bias.grad.clone();model.zero_grad(set_to_none=True)
        shifted=dict(data,costs=costs+20.)
        reactive_value_loss(model,shifted,True,True,.3).backward()
        torch.testing.assert_close(expected,model.unary[-1].bias.grad,rtol=2e-5,atol=2e-5)

    def test_decision_aligned_update_moves_actual_joint_gate(self):
        import identify_fusion_mechanisms as subject
        from policy.interaction_sac import CandidateCoordinationGate,MirrorCoordinationGate
        actor=MirrorCoordinationGate(16);critic=CandidateCoordinationGate(16)
        options=subject.reactive_options(self.packet)
        costs=torch.full((1,8,9),15.);costs[:,:,8]=5.
        data=dict(packet=copy.deepcopy(self.packet),base=self.base,options=options,costs=costs,outcomes=torch.full((1,8,9),3))
        subject.reactive_update(actor,critic,data,True,dict(temperature=1.,prior=.3,improvement=.3,
            exact_mirror=True,decision_loss=True),120,19)
        self.assertEqual(int(actor.log_policy(data['packet'],self.base,options,reference_prior=.3).argmax(-1)),8)

    def test_cardinal_values_keep_candidate_units_at_saturated_gates(self):
        from policy.interaction_sac import CandidateCoordinationGate
        model=CandidateCoordinationGate(16,cardinal_gates=True,gate_bias=0.)
        base=self.base.clone();base[...,2]=torch.tensor([.001,.999,.5,.1])
        with torch.no_grad():model.unary[-1].bias.copy_(torch.tensor([-2.,3.]))
        options=torch.zeros(1,9,4)
        for receiver in range(4):
            options[0,1+receiver*2,receiver]=-base[0,receiver,2]
            options[0,2+receiver*2,receiver]=1.-base[0,receiver,2]
        energy=model(self.packet,base,options)[1]
        torch.testing.assert_close(energy,torch.tensor([[0.,-2.,3.,-2.,3.,-2.,3.,-2.,3.]]))

    def test_prepared_joint_energy_matches_full_ensemble(self):
        from policy.interaction_sac import EnsembleCoordinationValue,MirrorCoordinationGate
        from identify_fusion_mechanisms import reactive_options
        value=EnsembleCoordinationValue(16,members=3,response_encoding=True,cardinal_gates=True)
        with torch.no_grad():
            for member in value.members:member.unary[-1].weight.normal_();member.mixed[-1].weight.normal_()
        actor=MirrorCoordinationGate(16,members=3,response_encoding=True,cardinal_gates=True);actor.add_value(value,.3)
        reference=(self.base[...,2]-.05).clamp(0.,1.)
        options=reactive_options(self.packet,True,reference=reference)
        prepared=actor.prepare_factors(self.packet,self.base)
        torch.testing.assert_close(actor.prepared_energy(prepared,options),actor(self.packet,self.base,options)[1],rtol=1e-6,atol=1e-6)

    def test_joint_refinement_improves_fusion_inside_candidate_interval(self):
        import identify_fusion_mechanisms as subject
        from policy.interaction_sac import CandidateCoordinationGate,MirrorCoordinationGate
        value=CandidateCoordinationGate(16)
        with torch.no_grad():value.unary[-1].bias.copy_(torch.tensor([-.37,1.]))
        actor=MirrorCoordinationGate(16);actor.add_value(value,1.)
        packet=copy.deepcopy(self.packet);packet['obs'][...,2]=torch.tensor([10.,10.,10.,40.])
        base=self.base.clone();base[...,2]=.5
        settings=dict(full_gate=True,all_gates=True,bias=-.05,temperature=1.,prior=.3,gate_levels=5)
        _,grid=subject.reactive_decision(actor,packet,base,settings)
        _,refined=subject.reactive_decision(actor,packet,base,dict(settings,refine_gates=True))
        def score(program):return actor(packet,base,program[:,None])[1][:,0]+.3*program.square().sum(-1)
        self.assertLess(float(score(refined)),float(score(grid))-.01)
        self.assertLess(float((refined[:,:3]-.37/2.6).abs().max()),.016)
        self.assertEqual(float(refined[0,3]),0.)

    def test_conditional_policy_can_act_on_pure_pair_synergy(self):
        from policy.interaction_sac import CandidateCoordinationGate,MirrorCoordinationGate
        from identify_fusion_mechanisms import reactive_decision
        packet=copy.deepcopy(self.packet);packet['obs'][...,2]=10.
        settings=dict(full_gate=True,all_gates=True,conditional_policy=True,refine_gates=True,
            bias=-.05,temperature=1.,prior=.3)
        for enabled in (True,False):
            critic=CandidateCoordinationGate(16,enabled,cardinal_gates=True)
            with torch.no_grad():critic.unary[-1].bias.fill_(1.);critic.mixed[-1].bias.fill_(-3.)
            actor=MirrorCoordinationGate(16,enabled,cardinal_gates=True);actor.add_value(critic,1.)
            _,program=reactive_decision(actor,packet,self.base,settings)
            self.assertEqual(int((program.abs()>1e-6).sum()),2 if enabled else 0)

    def test_conditional_policy_updates_pairs_successively_and_zero_step_is_exact(self):
        from policy.interaction_sac import CandidateCoordinationGate,MirrorCoordinationGate
        from identify_fusion_mechanisms import reactive_decision
        packet=copy.deepcopy(self.packet);packet['obs'][...,2]=10.
        settings=dict(full_gate=True,all_gates=True,conditional_policy=True,refine_gates=True,
            bias=-.05,temperature=1.,prior=.3)
        critic=CandidateCoordinationGate(16,cardinal_gates=True)
        with torch.no_grad():critic.unary[-1].bias.fill_(-1.)
        actor=MirrorCoordinationGate(16,cardinal_gates=True);actor.add_value(critic,1.)
        _,first=reactive_decision(actor,packet,self.base,settings)
        self.assertEqual(int((first.abs()>1e-6).sum()),2)
        actor.add_value(critic,0.)
        _,unchanged=reactive_decision(actor,packet,self.base,settings)
        torch.testing.assert_close(first,unchanged,rtol=0.,atol=0.)
        actor.coefficients[-1]=1.
        _,second=reactive_decision(actor,packet,self.base,settings)
        self.assertEqual(int((second.abs()>1e-6).sum()),4)
        self.assertLessEqual(int(((second-first).abs()>1e-6).sum()),2)

    def test_conditional_credits_include_exact_collecting_policy_reference(self):
        import identify_fusion_mechanisms as subject
        from policy.interaction_sac import CandidateCoordinationGate,MirrorCoordinationGate
        packet=copy.deepcopy(self.packet);packet['obs'][...,2]=10.
        settings=dict(full_gate=True,all_gates=True,conditional_policy=True,refine_gates=True,
            bias=-.05,temperature=1.,prior=.3)
        critic=CandidateCoordinationGate(16,cardinal_gates=True)
        with torch.no_grad():critic.unary[-1].bias.fill_(-1.)
        actor=MirrorCoordinationGate(16,cardinal_gates=True);actor.add_value(critic,.3)
        _,reference=subject.reactive_decision(actor,packet,self.base,settings)
        def source(obs,*unused):return self.base.expand(len(obs)//4,-1,-1).flatten(0,1),None
        def rollout(actor,source,packet,noise,settings,first=None,**kwargs):
            grouped=noise.reshape(1,3,10,300,17)
            torch.testing.assert_close(grouped,grouped[:,:,0:1].expand_as(grouped),rtol=0.,atol=0.)
            torch.testing.assert_close(first.reshape(1,3,10,4)[:,:,0],reference[:,None].expand(-1,3,-1))
            return first.square().sum(-1),torch.full((30,),3),None
        with patch.object(subject,'reactive_rollout',side_effect=rollout):
            data=subject.reactive_credits(actor,source,packet,settings,3,torch.Generator().manual_seed(5),1)
        self.assertEqual(data['costs'].shape,(1,3,10))
        self.assertEqual(subject.reactive_precision(data['costs']).shape,(9,9))

    def test_conditional_policy_update_uses_its_deployed_reference_and_ten_branches(self):
        import identify_fusion_mechanisms as subject
        from policy.interaction_sac import CandidateCoordinationGate,MirrorCoordinationGate
        packet=copy.deepcopy(self.packet);packet['obs'][...,2]=10.
        settings=dict(full_gate=True,all_gates=True,conditional_policy=True,refine_gates=True,
            exact_mirror=True,decision_loss=True,bias=-.05,temperature=1.,prior=.3,improvement=.3)
        critic=CandidateCoordinationGate(16,cardinal_gates=True)
        actor=MirrorCoordinationGate(16,cardinal_gates=True)
        with torch.no_grad():critic.unary[-1].bias.fill_(-1.)
        actor.add_value(critic,.3)
        _,reference=subject.reactive_decision(actor,packet,self.base,settings)
        options=subject.reactive_options(packet,reference=subject.reactive_reference(self.base,settings))
        options=torch.cat((reference[:,None],options),1)
        costs=(15.+options.square().sum(-1))[:,None].expand(-1,8,-1).clone()
        data=dict(packet=packet,base=self.base,options=options,costs=costs,outcomes=torch.full((1,8,10),3))
        result=subject.reactive_update(actor,critic,data,True,settings,8,71)
        torch.testing.assert_close(data['reference_program'],reference,rtol=0.,atol=0.)
        self.assertEqual(int(data['old_log_policy'].argmax(-1)),0)
        self.assertTrue(np.isfinite(result['value_loss']))


if __name__ == '__main__':
    unittest.main()
