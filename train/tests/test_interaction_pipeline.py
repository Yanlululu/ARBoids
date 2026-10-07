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
                                 InterventionSampler, DeploymentPolicy)
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
        intervention_interval=4, snapshot_interval=2, snapshot_capacity=8)
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


class Pipeline(unittest.TestCase):
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

    def test_bootstrap_calibration_uses_the_exact_soft_tail_and_one_terminal_branch(self):
        import interaction_review as review
        from interaction_rollout import compact_snapshot, FrozenPolicy, frozen_payload
        class ConstantCritic(torch.nn.Module):
            def __init__(self,value):
                super().__init__()
                self.value=value
            def forward(self,packet,action):
                value=torch.full((len(action),1),self.value)
                return value,value
        before=FrozenPolicy(frozen_payload(make_agent(small_config()),online_critic=True))
        after=copy.deepcopy(before)
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
