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
from train_interaction import arm_config, run_training, export_policy
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


class CandidateRoles(unittest.TestCase):
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
    def test_arm_selection_preserves_the_configured_long_horizon(self):
        config = small_config()
        config['interaction']['horizon_steps'] = 300
        for arm in ('full', 'same_info', 'no_peer', 'model_value'):
            self.assertEqual(arm_config(config, arm)['interaction']['horizon_steps'], 300)
        self.assertEqual(arm_config(config, 'short')['interaction']['horizon_steps'], 10)
        self.assertEqual(config['interaction']['horizon_steps'], 300)

    def test_direct_gate_task_training_resumes_with_sparse_actor_updates(self):
        c = small_config()
        c['rl']['GAMMA'] = 1.
        c['training'].update(gate_steps=12, joint_steps=4, warm_steps=4,
                             eval_interval=16, checkpoint_interval=16, log_interval=16)
        c['interaction'].update(entropy_objective='task-return-v4', reward_objective='capped-time-v1',
            gate_objective='paired-improvement-v1', deterministic_rollouts=True,
            gate_updates_per_batch=2, horizon_steps=300)
        with tempfile.TemporaryDirectory() as directory:
            left, right = Path(directory)/'whole', Path(directory)/'resumed'
            run_training(c, output=left)
            run_training(c, output=right, stop_at_step=7)
            run_training(c, output=right)
            whole = torch.load(left/'resume.pth', weights_only=False)
            resumed = torch.load(right/'resume.pth', weights_only=False)
            for key in ('agent', 'replay', 'actor_updates', 'critic_updates', 'step', 'simulated_steps'):
                equal(self, whole[key], resumed[key])
            self.assertEqual(whole['actor_updates'], 9)
            self.assertEqual(whole['critic_updates'], 4)
            for path in (left, right):
                self.assertEqual(json.loads((path/'progress.json').read_text())['updates'], 9)

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


if __name__ == '__main__':
    unittest.main()
