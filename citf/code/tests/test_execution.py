"""Execution-spec regressions for the four final review items; no performance claims."""
import copy,json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict,replace
import numpy as np
import torch
from citf.bridge import World,Source,Controller,factorize
from citf.search import candidate_grid,initialize_bank,source_center_config
from citf.collect import sample_root,load_behavior
from citf.contracts import Protocol,canonical_hash,file_hash,validate_predictor_protocol,runtime_identity,verify_freeze
from citf.model import Config,Predictor
from citf.train import snapshot_steps,preserve_snapshot,parameter_digest
from citf.opponents import APFOpponent,TorchScriptOpponent,load_opponent,OBSERVATION,attacker_observation,attacker_training_reward
from test_core import PROJECT,CK,sample

class ExportActor(torch.nn.Module):
    def forward(self,x):
        return torch.tanh(torch.stack((.02*x[:,0]+.03*x[:,1],-.01*x[:,0]+.04*x[:,1]),-1))


def export_actor(root):
    checkpoint=Path(root)/'actor.pt'
    torch.jit.trace(ExportActor().eval(),torch.zeros(1,8)).save(str(checkpoint))
    spec={'kind':'torchscript','checkpoint':checkpoint.name,'checkpoint_sha256':file_hash(checkpoint),
          'observation':OBSERVATION,'stateful':False,'stochastic':False,'training_identity':'test-only-untrained-actor'}
    path=Path(root)/'opponent.json';path.write_text(json.dumps(spec))
    return checkpoint,path,spec

class Execution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1);cls.source=Source(PROJECT,CK)
    def world(self):
        w=World(PROJECT);w.reset(101,102,2.);return w
    def checkpoint(self,path):
        w=self.world();m=Predictor().eval()
        torch.save({'state':m.state_dict(),'config':asdict(m.cfg),'protocol':{
            'source_sha256':self.source.sha256,'source_implementation_sha256':self.source.implementation_sha256,
            'environment_sha256':w.environment_sha256,'physical_protocol':asdict(w.protocol)}},path)
        return w
    def test_49_source_centre_in_actual_default_bank(self):
        w=self.world();_,meta=Controller(self.source,bank=12,budget=96,rounds=1).act(w)
        self.assertEqual(meta['source_center_indices'],[4]*6)
        self.assertTrue(meta['source_center_scored'])
        self.assertIn([4]*6,meta['initial_configurations'])
        self.assertLessEqual(meta['evaluated_unique'],60)
        self.assertEqual(meta['budget_limit'],96)
    def test_50_reservation_does_not_add_bank_slots(self):
        prior=torch.full((6,2),.5);grid,valid=candidate_grid(prior)
        ranked=np.array([[i]*6 for i in range(9)])
        bank,centre=initialize_bank(ranked,grid,valid,prior,bank=3,budget=3)
        self.assertEqual(len(bank),3);np.testing.assert_array_equal(bank[0],centre)
        self.assertEqual(len(set(map(tuple,bank))),3)
    def test_51_deduplicated_endpoint_centre_is_valid(self):
        prior=torch.tensor([[0.,1.],[1.,0.],[0.,0.]])
        grid,valid=candidate_grid(prior);index=source_center_config(grid,valid,prior)
        self.assertTrue(valid[torch.arange(3),torch.tensor(index)].all())
        torch.testing.assert_close(grid[torch.arange(3),torch.tensor(index)],prior)
    def test_52_tiny_budget_still_contains_centre(self):
        _,m=Controller(self.source,bank=12,budget=1,rounds=1).act(self.world())
        self.assertEqual(m['evaluated_unique'],1);self.assertTrue(m['source_center_scored'])
    def test_53_all_risk_weight_scan_values_load_same_predictor(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ck.pt';w=self.checkpoint(path)
            for weight in [0.,8.,32.,64.]:
                other=World(PROJECT,replace(w.protocol,risk_weight=weight));other.reset(101,102,2.)
                loaded=load_behavior(path,self.source,other,'cpu');self.assertIsInstance(loaded,Predictor)
    def test_54_predictor_physical_change_rejected(self):
        p=Protocol()
        for changed in [replace(p,time_limit=60.),replace(p,prediction_steps=12),replace(p,prior_scale=.5),replace(p,risk_clearance=8.)]:
            with self.assertRaises(ValueError):validate_predictor_protocol(asdict(p),changed)
    def test_55_final_freeze_locks_changed_scoring_weight(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'tasks.json').write_text(json.dumps({'role':'final_test'}))
            runtime=runtime_identity(PROJECT,CK,None,Protocol(),{'budget':96},'cpu')
            freeze={'frozen':True,'runtime':runtime,'runtime_sha256':canonical_hash(runtime),
              'tasks_sha256':file_hash(p/'tasks.json'),'selection_rule':'fixed','method_matrix':['CITF'],
              'primary_metric':'success','registry_sha256':'registry'}
            (p/'f.json').write_text(json.dumps(freeze))
            changed=copy.deepcopy(runtime);changed['protocol']['risk_weight']=8.
            with self.assertRaises(ValueError):verify_freeze(p/'f.json',None,p/'tasks.json',changed)
    def test_56_stage_schedule_has_zero_intermediates_final(self):
        self.assertEqual(snapshot_steps([0,400,1200,2400,6000,12000],12000),[0,400,1200,2400,6000,12000])
        self.assertEqual(snapshot_steps([0,2,4,400],4),[0,2,4])
    def test_57_stage_snapshot_is_immutable(self):
        with tempfile.TemporaryDirectory() as d:
            ck={'state':{'x':torch.arange(3.)},'update':2,'training_identity':'test'}
            path=preserve_snapshot(d,ck);before=file_hash(path)
            preserve_snapshot(d,ck);self.assertEqual(before,file_hash(path))
            changed=copy.deepcopy(ck);changed['state']['x']+=1
            with self.assertRaises(ValueError):preserve_snapshot(d,changed)
    def test_58_stage_snapshot_corruption_detected(self):
        with tempfile.TemporaryDirectory() as d:
            ck={'state':{'x':torch.arange(3.)},'update':2,'training_identity':'test'}
            path=preserve_snapshot(d,ck)
            edited=copy.deepcopy(ck);edited['extra']='corrupted';torch.save(edited,path)
            with self.assertRaises(ValueError):preserve_snapshot(d,ck)
    def test_59_learned_opponent_observation_recomputed(self):
        with tempfile.TemporaryDirectory() as d:
            _,spec,_=export_actor(d);actor=load_opponent(spec);w=self.world()
            a=actor.act(w);expected=ExportActor()(torch.tensor(attacker_observation(w)[None])).detach().numpy()[0]
            np.testing.assert_allclose(a,expected)
            w.env.attacker.pos*=.8;b=actor.act(w);self.assertGreater(float(np.max(abs(a-b))),1e-4)
    def test_60_learned_opponent_branches_do_not_fall_back_to_apf(self):
        with tempfile.TemporaryDirectory() as d:
            _,spec,_=export_actor(d);actor=load_opponent(spec);w=self.world()
            def forbidden(*args,**kwargs):raise AssertionError('Learned opponent fell back to APF')
            w.env._APF_navi_step=forbidden
            data,meta,count=sample_root(w,self.source,np.array([[4]*6,[0]*6]),[87,88],
              {'root_id':'r','layout_id':'l','split':'train'},opponent=actor)
            self.assertGreater(count,0)
            self.assertEqual(meta['opponent_identity']['kind'],'torchscript')
            actions=data['executed_attacker_actions'];mask=data['time_valid']
            self.assertTrue(torch.isfinite(actions[mask]).all())
            self.assertLessEqual(float(actions[mask].abs().max()),1.)
            torch.testing.assert_close(actions[0,:,0],actions[1,:,0])
    def test_61_apf_default_matches_explicit_apf(self):
        w=self.world();plans=np.array([[4]*6]);meta={'root_id':'r','layout_id':'l','split':'train'}
        a,_,_=sample_root(w,self.source,plans,[87],meta)
        b,_,_=sample_root(w,self.source,plans,[87],meta,opponent=APFOpponent())
        torch.testing.assert_close(a['real_positions'],b['real_positions'])
    def test_62_wrong_opponent_checkpoint_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ck,_,spec=export_actor(d);spec['checkpoint_sha256']='wrong'
            with self.assertRaises(ValueError):TorchScriptOpponent(ck,spec)
    def test_63_hidden_state_opponent_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ck,_,spec=export_actor(d);spec['stateful']=True
            with self.assertRaises(ValueError):TorchScriptOpponent(ck,spec)
    def test_64_wrong_opponent_observation_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            ck,_,spec=export_actor(d);spec['observation']='invented'
            with self.assertRaises(ValueError):TorchScriptOpponent(ck,spec)
    def test_65_attacker_reward_has_unambiguous_terminal_rules(self):
        dense=(40-39)/60-.01
        for code,bonus in [(0,0),(1,100),(2,0),(3,-100),(4,-100)]:
            self.assertAlmostEqual(attacker_training_reward(40,39,{'outcome':code}),dense+bonus)
    def test_66_radial_and_physical_flags_not_double_rewarded(self):
        x=attacker_training_reward(40,39,{'outcome':1,'physical_breach':1,'radial_only_failure':0})
        y=attacker_training_reward(40,39,{'outcome':1,'physical_breach':0,'radial_only_failure':1})
        self.assertEqual(x,y)
    def test_67_future_opponent_action_trace_is_not_model_input(self):
        b=sample(1,2);m=Predictor().eval();reference=m(b)
        b['executed_attacker_actions']=torch.randn(1,2,3,16,2)*100
        after=m(b)
        for key in ['xy','mu','sigma','risk']:torch.testing.assert_close(reference[key],after[key])
    def test_68_runtime_identity_contains_opponent(self):
        a=runtime_identity(PROJECT,CK,None,Protocol(),{'budget':96},'cpu')
        self.assertEqual(a['opponent']['kind'],'apf')
        b=runtime_identity(PROJECT,CK,None,Protocol(),{'budget':96},'cpu',opponent={'kind':'torchscript','checkpoint_sha256':'fake'})
        self.assertNotEqual(canonical_hash(a),canonical_hash(b))

if __name__=='__main__':unittest.main(verbosity=2)
