import copy,json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from citf.contracts import Protocol,geometry_identity,LayoutRegistry,check_split_families,canonical_hash,file_hash,verify_freeze
from citf.model import Predictor
from citf.bridge import World,Source,factorize,inputs,Controller
from citf.collect import sample_root
from citf.losses import objective
from test_core import sample,PROJECT,CK

class Revision(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);cls.source=Source(PROJECT,CK)
    def test_32_time_changes_input_and_legal_prefix(self):
        a=World(PROJECT);a.reset(911,912,2.);b=a.clone();b.env.Current_T=79.8
        fa=factorize(a,self.source);fb=factorize(b,self.source);plans=np.zeros((1,6),int)
        ia,_,_=inputs(a,fa,plans);ib,_,_=inputs(b,fb,plans)
        self.assertTrue(torch.equal(ia['history'],ib['history']))
        self.assertFalse(torch.equal(ia['time_context'],ib['time_context']))
        self.assertEqual(int(ia['known_time_valid'].sum()),16);self.assertEqual(int(ib['known_time_valid'].sum()),1)
    def test_33_real_timeout_label_matches_known_prefix(self):
        a=World(PROJECT);a.reset(911,912,2.);a.env.Current_T=79.8
        data,meta,steps=sample_root(a,self.source,np.zeros((1,6),int),[91],{'root_id':'x','layout_id':'x','split':'train'})
        self.assertEqual(int(data['time_valid'].sum()),1);self.assertEqual(steps,1)
        self.assertAlmostEqual(meta['remaining_seconds'],.2)
    def test_34_masked_nominal_tail_cannot_change_risk(self):
        b=sample(1,2);b['known_time_valid'][:,3:]=False;b['time_context'][:,1]=3/16
        m=Predictor().eval()
        torch.nn.init.normal_(m.pair[-1].weight,std=.1)
        before=m(b);x=copy.deepcopy(b);x['nominal'][...,3:,:]+=1000;after=m(x)
        for k in ['mu','sigma','risk']:torch.testing.assert_close(before[k],after[k],atol=1e-6,rtol=1e-6)
    def test_35_distance_head_does_not_use_auxiliary_motion_tail(self):
        b=sample(1,2);m=Predictor().eval();before=m(b)
        with torch.no_grad():m.motion[-1].weight.add_(100);m.motion[-1].bias.add_(100)
        after=m(b)
        for k in ['mu','sigma','risk']:torch.testing.assert_close(before[k],after[k])
        self.assertFalse(torch.equal(before['xy'],after['xy']))
    def test_36_protocol_rejects_silent_changes(self):
        for kw in [{'control_dt':.1},{'target_radius':18.},{'radial_early_failure':'false'}]:
            with self.assertRaises(ValueError):Protocol(**kw)
    def test_37_effective_protocol(self):
        a=World(PROJECT);self.assertEqual(a.env.Action_T,a.protocol.control_dt)
        self.assertEqual(a.env.early_attacker_win,a.protocol.radial_early_failure)
    def test_38_geometry_alias_cannot_cross_split(self):
        with tempfile.TemporaryDirectory() as d:
            reg=LayoutRegistry(Path(d)/'registry.json');x=geometry_identity(np.zeros((7,6)),6)
            reg.register(x,'train','original_name')
            with self.assertRaises(ValueError):reg.register(x,'validation','renamed')
    def test_39_geometry_fingerprint_integrity(self):
        x=geometry_identity(np.zeros((7,6)),6);r={'geometry_identity':x,'geometry_family_id':x['fingerprint']}
        y=copy.deepcopy(r);y['geometry_identity']['initial_state'][0][0]=1
        with self.assertRaises(ValueError):check_split_families([('train',[y])])
    def test_40_same_geometry_cross_agility(self):
        a=World(PROJECT);b=World(PROJECT);a.reset(100,101,2.);b.reset(100,201,2.5)
        self.assertEqual(a.initial_identity['fingerprint'],b.initial_identity['fingerprint'])
    def test_41_cache_identical_and_state_specific(self):
        m=Predictor().eval();b=sample(1,2);cache=m.encode_history(b)
        a=m(b);c=m(b,history_cache=cache)
        for k in ['xy','mu','sigma','risk']:torch.testing.assert_close(a[k],c[k])
    def test_42_controller_history_encoded_once(self):
        w=World(PROJECT);w.reset(101,102,2.);m=Predictor().eval();calls=[]
        original=m.encode_history
        def wrapped(b):calls.append(1);return original(b)
        m.encode_history=wrapped;Controller(self.source,m,bank=2,budget=5,rounds=1).act(w)
        self.assertEqual(len(calls),1)
    def test_43_step_trace_contains_controls_and_candidates(self):
        w=World(PROJECT);w.reset(101,102,2.);m=Predictor().eval();u,info=Controller(self.source,m,bank=2,budget=3,rounds=0).act(w)
        for k in ['gates','source_learned','source_rules','prior','configurations','scores','physical_thrust','prediction_batches','timings','time_context']:
            self.assertIn(k,info)
        np.testing.assert_allclose(info['physical_thrust'],u)
    def test_44_attacker_action_interface(self):
        w=World(PROJECT);w.reset(101,102,2.);r=w.step(np.ones((6,2))*300,attacker_action=np.zeros(2))
        self.assertIn('outcome',r)
        with self.assertRaises(ValueError):World(PROJECT).step(np.ones((6,2))*300,attacker_action=np.ones(2)*2)
    def test_45_full_freeze_rejects_control_change(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'ck').write_text('weight');(p/'tasks.json').write_text(json.dumps({'role':'final_test'}))
            runtime={'checkpoint_sha256':file_hash(p/'ck'),'controller':{'budget':96},'schema':'v2'}
            m={'frozen':True,'runtime':runtime,'runtime_sha256':canonical_hash(runtime),'tasks_sha256':file_hash(p/'tasks.json'),
               'registry_sha256':'x','selection_rule':'registered','method_matrix':['CITF'],'primary_metric':'success'}
            (p/'freeze.json').write_text(json.dumps(m));verify_freeze(p/'freeze.json',p/'ck',p/'tasks.json',runtime)
            changed=copy.deepcopy(runtime);changed['controller']['budget']=97
            with self.assertRaises(ValueError):verify_freeze(p/'freeze.json',p/'ck',p/'tasks.json',changed)
            with self.assertRaises(ValueError):verify_freeze(p/'freeze.json',p/'ck',p/'tasks.json')
    def test_46_deterministic_mode_uses_actual_mean_distance(self):
        b=sample(1,2);m=Predictor().eval();o=m(b)
        self.assertTrue((o['mean_distance']>=0).all())
        torch.testing.assert_close(o['det_risk'],(7-o['mean_distance']).clamp_min(0).square())
    def test_47_stopped_trajectory_gradients_masked(self):
        b=sample(1,2);b['time_valid'][...,4:]=False;m=Predictor().eval();out=m(b);out['xy'].retain_grad()
        objective(out,b)['total'].backward()
        self.assertEqual(float(out['xy'].grad[...,4:,:].abs().sum()),0.)
    def test_48_third_party_candidate_structural_probe(self):
        b=sample(1,2);m=Predictor().eval();torch.nn.init.normal_(m.pair[-1].weight,std=.05)
        x=copy.deepcopy(b);x['candidate'][:,:,2,:]+=0.7;x['nominal'][:,:,2]+=1
        self.assertGreater(float((m(b)['mu'][:,:,0]-m(x)['mu'][:,:,0]).abs().max().detach()),1e-7)
        m.cfg=type(m.cfg)(fleet_candidate_context=False)
        torch.testing.assert_close(m(b)['mu'][:,:,0],m(x)['mu'][:,:,0],atol=1e-6,rtol=1e-6)

if __name__=='__main__':unittest.main(verbosity=2)
