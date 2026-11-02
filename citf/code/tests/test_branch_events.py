"""Saved labels are checked against real World.step returns, including terminals."""
import copy
import unittest
from unittest.mock import patch
import numpy as np
import torch
from citf.bridge import World,Source
from citf.collect import sample_root
from citf.contracts import (COLLECTION_SCHEMA,BRANCH_EVENT_FIELDS,SCHEMA,FEATURES,
                            CONTEXT,geometry_identity,validate_dataset)
from citf.model import Predictor
from test_core import PROJECT,CK


class BranchEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source=Source(PROJECT,CK)

    def collect(self,kind):
        world=World(PROJECT);world.reset(911,912,2.)
        positions=np.array([[20.,0.],[-20.,0.],[0.,20.],[0.,-20.],[14.,14.],[-14.,-14.],[60.,0.]])
        if kind=='capture':positions[-1]=[21.,0.]
        if kind in ('collision','simultaneous'):positions[1]=[21.,0.]
        if kind in ('breach','simultaneous'):positions[-1]=[10.,0.]
        if kind=='radial':positions[-1]=[18.,0.]
        for boat,pos in zip([*world.env.defender_list,world.env.attacker],positions):
            boat.reset(pos.copy(),0.)
        if kind=='timeout':world.env.Current_T=79.8
        world.history.clear();world.history.append(world.state().copy())
        world.initial_identity=geometry_identity(world.state(),6)
        actual=[];step=World.step
        def observe(branch,*args,**kwargs):
            result=step(branch,*args,**kwargs)
            actual.append((result.copy(),branch.state().copy()))
            return result
        with patch.object(World,'step',observe):
            data,meta,cost=sample_root(world,self.source,np.array([[4]*6,[0]*6]),[91,92],
                                      {'root_id':'root','layout_id':'case','split':'train'})
        blob=dict(schema=SCHEMA,collection_schema=COLLECTION_SCHEMA,branch_event_fields=BRANCH_EVENT_FIELDS,
                  history_features=FEATURES,context_features=CONTEXT,protocol=world.protocol.__dict__,
                  roots=[meta],tensors={k:v[None] for k,v in data.items()})
        validate_dataset(blob)
        self.assertEqual(cost,len(actual))
        offset=0
        for c in range(2):
            for r in range(2):
                count=int(data['branch_stop_step'][c,r]);offset+=count
                result,state=actual[offset-1]
                self.assertEqual(int(data['branch_outcome'][c,r]),result['outcome'])
                self.assertEqual(bool(data['branch_terminated'][c,r]),bool(result['outcome']))
                self.assertEqual(data['branch_events'][c,r].tolist(),[bool(result[k]) for k in BRANCH_EVENT_FIELDS])
                self.assertEqual(float(data['branch_stop_time'][c,r]),result['duration'])
                self.assertTrue(data['time_valid'][c,r,:count].all())
                self.assertFalse(data['time_valid'][c,r,count:].any())
        return blob

    def test_real_terminal_classes_and_horizon(self):
        for kind,code,event in [('capture',3,'capture'),('collision',2,'physical_collision'),
                                 ('breach',1,'physical_breach'),('radial',1,'radial_only_failure'),
                                 ('timeout',4,'timeout'),('horizon',0,None)]:
            with self.subTest(kind=kind):
                b=self.collect(kind);d=b['tensors']
                self.assertTrue((d['branch_outcome']==code).all())
                self.assertTrue((d['branch_prediction_truncated']==(code==0)).all())
                if event:self.assertTrue(d['branch_events'][...,BRANCH_EVENT_FIELDS.index(event)].all())

    def test_independent_events_survive_primary_priority(self):
        d=self.collect('simultaneous')['tensors']
        self.assertTrue((d['branch_outcome']==1).all())
        for key in ('physical_collision','physical_breach'):
            self.assertTrue(d['branch_events'][...,BRANCH_EVENT_FIELDS.index(key)].all())

    def test_inconsistent_saved_event_is_rejected(self):
        b=self.collect('timeout')
        for field,value in [('branch_outcome',0),('branch_stop_step',2),
                            ('branch_terminated',False),('branch_prediction_truncated',True),
                            ('branch_stop_time',79.)]:
            with self.subTest(field=field):
                x=copy.deepcopy(b);x['tensors'][field].fill_(value)
                with self.assertRaises(ValueError):validate_dataset(x)
        x=copy.deepcopy(b);x['tensors']['branch_events'].zero_()
        with self.assertRaises(ValueError):validate_dataset(x)
        x=copy.deepcopy(b);del x['collection_schema']
        with self.assertRaises(ValueError):validate_dataset(x)

    def test_events_are_not_predictor_features(self):
        data=self.collect('timeout')['tensors'];model=Predictor().eval()
        with torch.no_grad():
            original=model(data)
            altered={k:v for k,v in data.items() if not k.startswith('branch_')}
            other=model(altered)
        for key in ('xy','mu','sigma','risk'):torch.testing.assert_close(original[key],other[key],atol=0,rtol=0)


if __name__=='__main__':unittest.main()
