"""Checks of state restoration, execution assumptions and diagnostic selection."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np

from source_arboids import snapshot, source_mixture
from evaluate_gate_contribution import setup_scene
from gate_study_control import GateController, joint_indices
from gate_mechanism_diagnostics import freeze, restore, held_paths, state_paths, project_to_segments, ranking_metrics, shortlist, branch, FineGate, oracle_key


def policy(observation):
    return np.tile(np.array([.6, .1, .35], dtype=np.float32), (len(observation), 1))


class MechanismTests(unittest.TestCase):
    def setUp(self):
        env, obs = setup_scene(dict(scene_seed=17561, noise_seed=28700, defenders=3, agility=2.25))
        for _ in range(7):
            obs, _, done, _ = env.step(policy(obs), 'AdaRes')
            self.assertFalse(done)
        self.case = dict(meta=dict(case_id='test-case', scene_seed=17561, control_step=7), state=freeze(env, obs))

    def test_complete_restore_and_rng_are_exact(self):
        first, obs1 = restore(self.case['state'])
        outputs = []
        for _ in range(4):
            obs1, reward, done, _ = first.step(policy(obs1), 'AdaRes')
            outputs.append((obs1.copy(), reward.copy(), snapshot(first), first.attacker.pos.copy(), done))
        second, obs2 = restore(self.case['state'])
        for expected in outputs:
            obs2, reward, done, _ = second.step(policy(obs2), 'AdaRes')
            for a,b in zip(expected, (obs2,reward,snapshot(second),second.attacker.pos,done)):
                np.testing.assert_array_equal(a,b)

    def test_separable_held_paths_match_original_full_environment(self):
        env, obs = restore(self.case['state'])
        thrust = source_mixture(policy(obs), env.boids_actions).astype(float)
        paths = held_paths(self.case, thrust[:, None, :])
        env, _ = restore(self.case['state'])
        for step in range(10):
            env.step(env.thrust_to_action(thrust), 'RL')
            np.testing.assert_allclose(paths[:,0,step+1], snapshot(env), rtol=0, atol=1e-12)

    def test_position_and_heading_match_nominal_source(self):
        env, obs = restore(self.case['state'])
        _, thrusts = GateController().candidates(policy(obs), env.boids_actions)
        predicted = state_paths(np.repeat(snapshot(env), 6, axis=0), thrusts.reshape(-1,2)).reshape(3,6,11,6)
        actual = held_paths(self.case, thrusts, nominal=True)
        np.testing.assert_allclose(predicted[...,:3], actual[...,:3], rtol=0, atol=1e-6)
        double=state_paths(np.repeat(snapshot(env),6,axis=0),thrusts.astype(float).reshape(-1,2)).reshape(3,6,11,6)
        np.testing.assert_allclose(double[...,:3],actual[...,:3],rtol=0,atol=1e-10)
        original=GateController().model.trajectories(np.repeat(snapshot(env),6,axis=0),thrusts.reshape(-1,2),2.)
        np.testing.assert_array_equal(predicted[...,:2],original.reshape(3,6,41,2)[:,:,::4])

    def test_no_override_restoration_matches_original(self):
        result, trace = branch(self.case, policy, 'original', stop_steps=4)
        env, obs = restore(self.case['state'])
        for row in trace:
            obs, _, _, _ = env.step(policy(obs), 'AdaRes')
            np.testing.assert_array_equal(snapshot(env), row['positions'])
        self.assertEqual(result['steps'], 4)

    def test_projection_and_degenerate_segment(self):
        action = np.array([[1.,1.,.5], [0.,0.,.5]], dtype=np.float32)
        boids = np.array([[0.,0.], [250.,250.]])
        theta, point, distance = project_to_segments(np.array([[500.,700.], [300.,250.]]), action, boids)
        np.testing.assert_allclose(theta,[.6,0.])
        np.testing.assert_allclose(point,[[600.,600.],[250.,250.]])
        np.testing.assert_allclose(distance,[np.sqrt(20000.),50.])

    def test_shortlist_contains_every_selected_option(self):
        env, obs = restore(self.case['state'])
        action = policy(obs)
        theta,_ = GateController().candidates(action, env.boids_actions)
        ids,selected = shortlist(self.case, snapshot(env), action, env.boids_actions,theta,joint_indices(3,6))
        self.assertEqual(len(ids),96)
        self.assertEqual(len(set(ids)),96)
        self.assertTrue(set(selected.values()).issubset(ids))
        self.assertIn(0,ids)

    def test_missed_danger_denominator_and_undefined_rank(self):
        row = ranking_metrics([8.,8.,8.], [4.,6.,9.])
        self.assertEqual(row['miss_rate_5'],1.)
        self.assertEqual(row['miss_rate_7'],1.)
        self.assertEqual(row['false_safe_rate_7'],2/3)
        self.assertIsNone(row['risk_rank_spearman'])
        self.assertIsNone(ranking_metrics([10.],[10.])['miss_rate_5'])

    def test_fine_grid_contains_coarse_and_original(self):
        env,obs=restore(self.case['state'])
        action=policy(obs)
        theta,thrust=FineGate().candidates(action,env.boids_actions)
        self.assertEqual(thrust.shape,(3,10,2))
        self.assertTrue({0.,.25,.5,.75,1.}.issubset(set(theta[0])))
        np.testing.assert_array_equal(theta[:,0],action[:,2])
        np.testing.assert_array_equal(thrust[:,0],source_mixture(action,env.boids_actions))

    def test_oracle_priorities_use_task_outcomes(self):
        base=dict(collision=0,source_loss=0,capture=0,team_return=-100.,minimum_distance=5.1)
        collision=dict(base,collision=1,team_return=100.)
        loss=dict(base,source_loss=1,team_return=100.)
        self.assertGreater(oracle_key(base,.5),oracle_key(collision,0.))
        self.assertGreater(oracle_key(base,.5),oracle_key(loss,0.))

    def test_hazard_summary_keeps_early_collision_and_unknown_loss_separate(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'figures'))
        from gen_fig_gate_mechanisms import matched_candidate_metrics
        states=[dict(case_id='c',suite='test',scene_seed='1',category='failure')]
        samples=[dict(case_id='c',horizon_steps='10',complete=str(complete),outcome=str(outcome),
                      predicted_dmin=str(pred),replan_dmin=str(actual),held_dmin=str(actual))
                 for complete,outcome,pred,actual in [(0,2,8,4.9),(0,1,9,9),(1,0,8,6),(1,0,9,9)]]
        with patch('gen_fig_gate_mechanisms.read_csv',side_effect=lambda path:states if path.name=='states.csv' else samples):
            matched,coverage,hazards=matched_candidate_metrics(Path('.'))
        self.assertEqual(matched[0]['candidates'],2)
        self.assertEqual(coverage[0]['complete_fraction'],.5)
        self.assertEqual(coverage[0]['early_collisions'],1)
        self.assertEqual(hazards[0]['observed_danger'],1)
        self.assertEqual(hazards[0]['missed'],1)
        self.assertEqual(hazards[0]['unobserved_full_horizon'],1)
        self.assertEqual(hazards[1]['observed_danger'],2)
        self.assertEqual(hazards[1]['missed'],2)

    def test_scene_mean_does_not_count_states_as_independent_scenes(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'figures'))
        from gen_fig_gate_mechanisms import cluster_mean
        rows=[dict(suite='test',scene_seed='1',value=0.),dict(suite='test',scene_seed='1',value=0.),
              dict(suite='test',scene_seed='1',value=0.),dict(suite='test',scene_seed='2',value=1.)]
        mean,_,_,count=cluster_mean(rows,'value')
        self.assertEqual(mean,.5)
        self.assertEqual(count,2)


if __name__ == '__main__':
    unittest.main()
