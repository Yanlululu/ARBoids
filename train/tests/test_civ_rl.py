import sys
from pathlib import Path
import unittest
import tempfile

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import civ_rl as c
from envs.TADgame import TADEnv
from conditional_fusion_value import _refine_actuator_interactions


class CIVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source=Path(__file__).resolve().parents[1]/'experiments/author-source-fusion-20261011/two-hour-development-20261026/best-m1-400.pth'
        if not cls.source.exists():raise unittest.SkipTest('Pilot CI source is unavailable.')
        cls.actor=c.load_source(cls.source)

    def packet(self,n=3):
        np.random.seed(261027);env=TADEnv(n,protocol=self.actor.environment_protocol)
        obs,_=env.reset(agility=2.5,noisy_agility=False)
        return c.d.mechanism_tensor(c.d.joint_packet(env,obs))

    def test_export_preserves_reference_and_bounds(self):
        for generator in ('joint','independent'):
            config=c.CIVConfig(generator=generator)
            controller=c.CIVController(self.actor,config);packet=self.packet()
            source,catalog,features,_=c.fixed_inference(controller.decide,packet,64)
            settings=dict(self.actor.actuator_fusion_refinement,selection=generator)
            expected=c.fixed_inference(lambda p:_refine_actuator_interactions(self.actor,p,
                self.actor(p,deterministic=True,gate_only=True)[0],
                self.actor(p,deterministic=True,gate_only=True)[0],**settings),packet,64)
            actual=controller.execute(source,packet['node'][...,11:13],catalog['gates'][:,0])
            torch.testing.assert_close(actual,expected,rtol=0.,atol=0.)
            self.assertEqual(features['node'].shape[-1],23)
            self.assertEqual(features['edge'].shape[-1],7)
            self.assertEqual(features['global_features'].shape[-1],13)
            self.assertTrue(bool(torch.isfinite(features['node']).all()))
            self.assertTrue(bool(((features['gates']>=0)&(features['gates']<=1)).all()))
            valid=catalog['gates'][0,catalog['valid'][0]]
            self.assertEqual(len(valid),len(torch.unique(valid.flatten(1),dim=0)))

    def test_value_fleet_size_and_permutation(self):
        from civ_generalization import StandardizedRelationalValueNetwork
        torch.manual_seed(42)
        networks=[c.CIVValueNetwork().eval(),
                  StandardizedRelationalValueNetwork(encounter_features=True,focus_changes=True).eval()]
        for n in (3,6,8):
            controller=c.CIVController(self.actor,c.CIVConfig(generator='local'))
            _,_,features,_=controller.decide(self.packet(n))
            order=torch.arange(n-1,-1,-1)
            permuted=dict(features,node=features['node'][:,:,order],edge=features['edge'][:,:,order][:,:,:,order])
            for network in networks:
                result=network(features)
                torch.testing.assert_close(result,network(permuted),rtol=1e-5,atol=1e-6)
                self.assertEqual(result.shape,(1,11,3))
                self.assertTrue(bool(torch.isfinite(result).all()))

    def test_shared_executor_forced_branch_and_deployment(self):
        config=c.CIVConfig(decision_step=0,durations=(2,),delay=1)
        controller=c.CIVController(self.actor,config);packet=self.packet()
        _,_,features,_=c.fixed_inference(controller.decide,packet,64)
        choice=int(torch.nonzero(features['valid'][0])[1])
        state=c.ExecutionState();executor=c.CIVExecutor(controller)
        first,_=executor.actions(packet,[state],forced=[choice])
        branch=c.ExecutionState();branch.commit(choice,features['gates'][0,choice].numpy(),2,1)
        source,reference=c.fixed_inference(lambda p:controller.proposals(p,reference_only=True),packet,64)
        for tick in range(5):
            state.step=branch.step=tick
            actual,_=executor.actions(packet,[state])
            expected=controller.execute(source,packet['node'][...,11:13],source.new_tensor(
                branch.coefficients(reference['gates'][0,0].numpy())[None])).numpy()
            np.testing.assert_array_equal(actual,expected)
        with self.assertRaises(ValueError):state.commit(choice,np.ones((3,2)),2,1)

    def test_conditional_supervision_and_null_selection(self):
        with torch.enable_grad():
            logits=torch.zeros(2,3,3,requires_grad=True)
            target=torch.tensor([[0.,1.,0.],[1.,0.,1.]])
            valid=torch.ones(2,3,dtype=torch.bool)
            c.conditional_loss(logits,target,valid).backward()
            self.assertEqual(float(logits.grad[0,:,2].abs().sum()),0.)
            self.assertEqual(float(logits.grad[1,:,1].abs().sum()),0.)
            self.assertLess(float(logits.grad[0,1,1]),0.)
            self.assertLess(float(logits.grad[1,1,2]),0.)
        gain=torch.tensor([[0.,.1,9.]])
        selected,_=c.choose_options(gain,torch.zeros_like(gain),torch.tensor([[True,True,False]]),.2,.5)
        self.assertEqual(int(selected),0)

    def test_measured_transport_matches_numerical_execution(self):
        from civ_deployment import CIVMeasuredController
        from source_arboids import source_mixture
        class ChooseFirst(torch.nn.Module):
            def forward(self,features):
                logits=features['node'].new_full((*features['valid'].shape,3),-5.)
                logits[:,0]=0.;logits[:,1]=5.
                return logits
        config=c.CIVConfig(decision_step=0,durations=(2,),delay=1)
        controller=c.CIVController(self.actor,config,[ChooseFirst()],kind='ordinary')
        measured=CIVMeasuredController(None,controller=controller);executor=c.CIVExecutor(controller)
        state=c.ExecutionState();np.random.seed(281027)
        env=TADEnv(3,protocol=self.actor.environment_protocol);obs,_=env.reset(agility=2.5,noisy_agility=False)
        for _ in range(5):
            packet=c.d.mechanism_tensor(c.d.joint_packet(env,obs))
            physical,_=executor.actions(packet,[state])
            defenders=np.asarray([[b.x,b.y,b.theta,*b.velocity] for b in env.defender_list])
            a=env.attacker;attacker=np.asarray([a.x,a.y,a.theta,*a.velocity])
            padded=np.concatenate((obs,np.zeros((3,2))),-1)
            actual,_=measured.control(defenders,attacker,padded,env.boids_actions,env.Current_T,env.Total_T)
            np.testing.assert_array_equal(actual,source_mixture(physical[0],env.boids_actions))
            obs,_,_,_=c.d.joint_task_step(env,physical[0]);state.step+=1
        self.assertEqual(measured.state.choice,1)

    def test_paired_gain_loss_anchors_whole_candidate_consequences(self):
        from civ_rl_experiment import fit_objective
        with torch.enable_grad():
            logits=torch.zeros(2,3,3,requires_grad=True)
            targets=torch.tensor([[0.,1.,0.],[1.,0.,1.]])
            valid=torch.tensor([[True,True,False],[True,True,False]])
            base=c.conditional_loss(logits,targets,valid)
            loss=fit_objective(logits,targets,valid,'conditional',1.)
            torch.testing.assert_close(loss-base,torch.ones(()))
            loss.backward()
            self.assertEqual(float(logits.grad[:,2].abs().sum()),0.)
            with self.assertRaises(ValueError):fit_objective(logits,targets,valid,'ordinary',1.)

    def test_checkpoint_relocation_preserves_source_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'civ.pth'
            record=c.save_checkpoint(path,self.source,c.CIVConfig(),[c.CIVValueNetwork()],
                'conditional',dict(completed=True))
            record['source_path']='/absent-training-host/source.pth';torch.save(record,path)
            restored=c.load_controller(path)
            self.assertEqual(restored.config.identity(),record['execution_identity'])
            record['source_sha256']='0'*64;torch.save(record,path)
            with self.assertRaisesRegex(ValueError,'source identity'):c.load_controller(path)

    def test_paired_margin_uses_exact_null_and_ignores_padded_candidates(self):
        from civ_generalization import loss_function
        with torch.enable_grad():
            raw=torch.zeros(2,3,3,requires_grad=True)
            target=torch.tensor([[[0.,1.,0.]],[[1.,0.,1.]]])
            future_mask=torch.ones(2,1,dtype=torch.bool)
            valid=torch.tensor([[True,True,False],[True,True,False]])
            importance=torch.ones(2)
            def loss(value,rank):
                return loss_function(value,target,future_mask,valid,importance,
                                     'conditional',2.,rank,'paired_margin')
            ranking=loss(raw,1.)-loss(raw,0.)
            changed=raw.detach().clone();changed[:,0,1:]=torch.tensor([9.,-7.])
            torch.testing.assert_close(ranking,loss(changed,1.)-loss(changed,0.),rtol=1e-5,atol=1e-6)
            loss(raw,1.).backward()
            self.assertTrue(bool(torch.isfinite(raw.grad).all()))
            self.assertEqual(float(raw.grad[:,2].abs().sum()),0.)
            self.assertLess(float(raw.grad[0,1,1]),0.)
            self.assertLess(float(raw.grad[1,1,2]),0.)

    def test_paired_event_value_matches_signed_event_probability(self):
        from civ_generalization import loss_function
        with torch.enable_grad():
            raw=torch.tensor([[[.3,7.,-4.],[.1,1.2,-.7],[.2,-.4,.8]]],requires_grad=True)
            class FixedNetwork(torch.nn.Module):
                paired_event_value=True
                def forward(self,features):return raw
            gain,uncertainty=c.ensemble_values([FixedNetwork()],{},'conditional')
            probability=torch.cat((torch.zeros_like(raw[...,:1]),raw[...,1:]),-1).softmax(-1)
            expected=probability[...,1]-probability[...,2];expected=expected.clone();expected[:,0]=0.
            torch.testing.assert_close(gain,expected)
            torch.testing.assert_close(uncertainty,torch.zeros_like(gain))
            target=torch.tensor([[[0.,1.,0.],[1.,1.,0.]]])
            loss=loss_function(raw,target,torch.ones(1,2,dtype=torch.bool),
                torch.tensor([[True,True,False]]),torch.ones(1),'conditional',4.,4.,'paired_events')
            loss.backward()
            self.assertTrue(bool(torch.isfinite(loss)))
            self.assertTrue(bool(torch.isfinite(raw.grad).all()))
            self.assertEqual(float(raw.grad[:,2].abs().sum()),0.)


if __name__=='__main__':unittest.main()
