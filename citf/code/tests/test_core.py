import copy,importlib.util,json,math,tempfile,unittest
from pathlib import Path
import numpy as np
from scipy.integrate import quad
import torch
from citf.model import Config,Predictor
from citf.risk import rectified_gaussian_risk,rectified_gaussian_nll,branch_minima
from citf.losses import objective
from citf.search import candidate_grid,nominal_score,nominal_beam,whole_configuration_search
from citf.contracts import validate_dataset,FEATURES,SCHEMA,CONTEXT,Protocol,geometry_identity
from citf.physics import step
from citf.bridge import Source,World,factorize,inputs,Controller
ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'source_archive'
CK=PROJECT/'train/experiments/author-source-fusion-20261011/two-hour-development-20261026/best-m1-400.pth'

def sample(B=2,C=3,E=7,T=16):
    torch.manual_seed(41)
    pos=torch.randn(B,E,2)*8
    nom=pos[:,None,:,None,:]+torch.randn(B,C,E,T,2).cumsum(-2)*.1
    cand=torch.rand(B,C,E,8);cand[...,-1,:]=0
    b=dict(history=torch.randn(B,E,8,8),positions=pos,nominal=nom,candidate=cand,
       entity_valid=torch.ones(B,E,dtype=torch.bool),history_valid=torch.ones(B,E,8,dtype=torch.bool),
       candidate_valid=torch.ones(B,C,dtype=torch.bool),unary=torch.rand(B,C),
       real_positions=nom[:,:,None].repeat(1,1,3,1,1,1)+.03*torch.randn(B,C,3,E,T,2),
       time_valid=torch.ones(B,C,3,T,dtype=torch.bool),
       time_context=torch.ones(B,2),known_time_valid=torch.ones(B,T,dtype=torch.bool))
    return b

class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def setUp(self):self.b=sample();self.m=Predictor().eval()
    def randomized(self):
        for head in (self.m.motion[-1],self.m.pair[-1]):torch.nn.init.normal_(head.weight,std=.02)
    def test_01_risk_integral(self):
        for mu,sd in [(-2.,.8),(0.,2.),(4.,1.5),(7.,.2),(8.,3.),(20.,.1)]:
            atom=49*.5*(1+math.erf(-mu/sd/math.sqrt(2)))
            val=atom+quad(lambda x:(7-x)**2*math.exp(-.5*((x-mu)/sd)**2)/(sd*math.sqrt(2*math.pi)),0,7,epsabs=1e-10)[0]
            pred=float(rectified_gaussian_risk(torch.tensor(mu,dtype=torch.float64),torch.tensor(sd,dtype=torch.float64)))
            self.assertAlmostEqual(val,pred,places=7)
    def test_02_risk_bounds_and_grad(self):
        mu=torch.linspace(-10,30,40,requires_grad=True);s=torch.ones_like(mu,requires_grad=True)
        r=rectified_gaussian_risk(mu,s);r.sum().backward()
        self.assertTrue(((r>=0)&(r<=49)).all());self.assertTrue(torch.isfinite(mu.grad).all());self.assertTrue(torch.isfinite(s.grad).all())
    def test_03_zero_variance(self):
        x=torch.tensor([1.,5.,7.,12.]);y=rectified_gaussian_risk(x,torch.full_like(x,1e-5))
        torch.testing.assert_close(y,(7-x).clamp_min(0).square(),atol=1e-5,rtol=1e-5)
    def test_04_likelihood_atom(self):
        d=torch.tensor([0.,3.]);m=torch.tensor([1.,2.]);s=torch.tensor([2.,1.])
        l=rectified_gaussian_nll(d,m,s);self.assertAlmostEqual(float(l[0]),float(-torch.special.log_ndtr(torch.tensor(-.5))),places=6)
    def test_05_first_step_anchor(self):
        self.randomized();o=self.m(self.b);torch.testing.assert_close(o['xy'][...,0,:],self.b['nominal'][...,0,:])
    def test_06_zero_head_nominal(self):
        o=self.m(self.b);torch.testing.assert_close(o['xy'],self.b['nominal'])
    def test_07_candidate_independence(self):
        self.randomized();a=self.m(self.b);one={k:(v[:,:1] if k in ('candidate','nominal','unary','candidate_valid','real_positions','time_valid') else v) for k,v in self.b.items()}
        b=self.m(one)
        for key in ['xy','mu','sigma','risk']:torch.testing.assert_close(a[key][:,:1],b[key],atol=2e-5,rtol=2e-5)
    def test_08_entity_permutation(self):
        self.randomized();b=copy.deepcopy(self.b);perm=torch.tensor([4,1,5,2,0,3,6]);n=6
        for key in ['history','positions','entity_valid','history_valid']:b[key]=b[key][:,perm]
        for key in ['candidate','nominal']:b[key]=b[key][:,:,perm]
        a=self.m(self.b);c=self.m(b);torch.testing.assert_close(c['xy'],a['xy'][:,:,perm],atol=3e-5,rtol=3e-5)
        ij=torch.triu_indices(n,n,1);M=torch.zeros(2,3,n,n);M[:,:,ij[0],ij[1]]=a['risk'];M=M+M.transpose(-1,-2)
        expected=M[:,:,perm[:n]][:,:,:,perm[:n]][:,:,ij[0],ij[1]]
        torch.testing.assert_close(c['risk'],expected,atol=4e-5,rtol=4e-5)
    def test_09_padded_entity(self):
        self.randomized();b=copy.deepcopy(self.b);B,C,E,T,_=b['nominal'].shape
        for key in ['history','positions','entity_valid','history_valid']:
            pad=torch.zeros_like(b[key][:,:1]);b[key]=torch.cat((b[key][:,:-1],pad,b[key][:,-1:]),1)
        for key in ['candidate','nominal']:
            pad=torch.zeros_like(b[key][:,:,:1]);b[key]=torch.cat((b[key][:,:,:-1],pad,b[key][:,:,-1:]),2)
        a=self.m(self.b);c=self.m(b);ix=list(range(E-1))+[E]
        torch.testing.assert_close(c['xy'][:,:,ix],a['xy'],atol=3e-5,rtol=3e-5)
        # Pair ordering changes when one dummy defender is appended, mask recovers old edges.
        mask=c['pair_valid'][0];torch.testing.assert_close(c['risk'][:,:,mask],a['risk'],atol=4e-5,rtol=4e-5)
    def test_10_variable_fleet(self):
        for E in [4,6,9]:
            o=self.m(sample(1,2,E));self.assertEqual(o['risk'].shape[-1],(E-1)*(E-2)//2)
    def test_11_temporal_padding_finite(self):
        self.b['history_valid'][:,:,:5]=False;o=self.m(self.b);self.assertTrue(torch.isfinite(o['risk']).all())
    def test_12_minima_synchronous(self):
        real=torch.zeros(1,1,1,3,2,2);real[0,0,0,0,:,0]=torch.tensor([0.,5.]);real[0,0,0,1,:,0]=torch.tensor([5.,10.])
        d,_=branch_minima(real,torch.ones(1,1,1,2,dtype=torch.bool),torch.ones(1,3,dtype=torch.bool));self.assertEqual(float(d),5.)
    def test_13_terminal_padding_ignored(self):
        self.b['time_valid'][...,8:]=False;self.b['real_positions'][...,8:,:]=0
        d,_=branch_minima(self.b['real_positions'],self.b['time_valid'],self.b['entity_valid'])
        direct,_=branch_minima(self.b['real_positions'][...,:8,:],self.b['time_valid'][...,:8],self.b['entity_valid']);torch.testing.assert_close(d,direct)
    def test_14_terminal_reactivation_rejected(self):
        self.b['time_valid'][0,0,0,1]=False
        with self.assertRaises(ValueError):branch_minima(self.b['real_positions'],self.b['time_valid'],self.b['entity_valid'])
    def test_15_gradient_all_heads(self):
        loss=objective(self.m(self.b),self.b);loss['total'].backward()
        self.assertGreater(float(self.m.motion[-1].weight.grad.abs().sum()),0);self.assertGreater(float(self.m.pair[-1].weight.grad.abs().sum()),0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.m.parameters() if p.grad is not None))
    def test_16_no_terminal_supervision(self):
        a=objective(self.m(self.b),self.b)['total'];b=copy.deepcopy(self.b);b['terminal_success']=torch.randn(2,3)
        c=objective(self.m(b),b)['total'];torch.testing.assert_close(a,c)
    def test_17_motion_ablation(self):
        a=objective(self.m(self.b),self.b,mode='motion_only');torch.testing.assert_close(a['total'],a['motion'])
    def test_18_label_no_grad(self):
        self.b['real_positions'].requires_grad_(True);objective(self.m(self.b),{**self.b,'real_positions':self.b['real_positions'].detach()})['total'].backward();self.assertIsNone(self.b['real_positions'].grad)
    def test_19_candidate_duplicates(self):
        g,v=candidate_grid(torch.tensor([[0.,0.],[1.,1.],[.3,.3]]));self.assertEqual(v.sum(1).tolist(),[4,4,9])
    def test_20_score_no_double_count(self):
        u=np.zeros((3,2));p=np.ones((3,3,2,2));self.assertEqual(float(nominal_score([[0,0,0]],u,p,2)[0]),6)
    def test_21_search_cache_budget(self):
        calls=[]
        def f(x):calls.extend(map(tuple,x));return ((x-1)**2).sum(1)
        x,m=whole_configuration_search([[0,0],[0,0]],f,[range(2),range(2)],budget=4,rounds=3)
        self.assertEqual(x.tolist(),[1,1]);self.assertEqual(len(calls),len(set(calls)));self.assertLessEqual(m['evaluated_unique'],4)
    def test_22_small_beam_exact(self):
        import itertools
        rng=np.random.default_rng(5);u=rng.random((3,2));p=rng.random((3,3,2,2));s,c=nominal_beam(u,p,width=8)
        allp=np.array(list(itertools.product(range(2),repeat=3)));self.assertAlmostEqual(c[0],nominal_score(allp,u,p).min())
    def test_23_dataset_split_leak(self):
        b=sample();meta=dict(source_sha256='s',source_implementation_sha256='impl',environment_sha256='e',future_seeds=[1,2,3],capture_stage='approach',sampling_probability=1.)
        blob=dict(schema=SCHEMA,history_features=FEATURES,tensors=b,roots=[dict(meta,root_id=0,layout_id='x',split='train'),dict(meta,root_id=1,layout_id='x',split='validation')])
        with self.assertRaises(ValueError):validate_dataset(blob)
    def test_24_dataset_valid(self):
        meta=dict(source_sha256='s',source_implementation_sha256='impl',environment_sha256='e',future_seeds=[1,2,3],capture_stage='approach',sampling_probability=1.,elapsed_seconds=0.,remaining_seconds=80.,behavior_identity={'method':'test'})
        roots=[]
        for i in range(2):
            ident=geometry_identity(np.full((7,6),float(i)),6)
            roots.append(dict(meta,root_id=i,layout_id=str(i),split='train' if i==0 else 'validation',geometry_identity=ident,geometry_family_id=ident['fingerprint']))
        blob=dict(schema=SCHEMA,history_features=FEATURES,context_features=CONTEXT,protocol=Protocol().__dict__,tensors=self.b,roots=roots);self.assertEqual(validate_dataset(blob)['roots'],2)

class RealArchive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1);cls.source=Source(PROJECT,CK)
    def setUp(self):self.world=World(PROJECT);self.world.reset(9987,9988,2.)
    def test_25_physics_parity(self):
        boat=copy.deepcopy(self.world.env.defender_list[0]);boat.velocity_r=np.array([1.2,-.3,.2]);boat.current_velocity=np.zeros(3);boat.update_velocity(np.zeros(3));force=np.array([300.,870.])
        nominal=step(torch.tensor(boat.motion_state(),dtype=torch.float64),torch.tensor(force,dtype=torch.float64))
        boat.step(force,np.zeros(3));np.testing.assert_allclose(nominal.numpy(),boat.motion_state(),atol=1e-9)
    def test_26_moving_measured_velocity(self):
        force=np.ones((6,2))*500;self.world.step(force);hist,pos,mask,R=self.world.observation()
        np.testing.assert_allclose(hist[:,-1,2:4].numpy()*5,self.world.state()[:,3:5]@R,atol=1e-6)
    def test_27_source_bounds_and_feedback(self):
        a,b,g=self.source.commands(self.world.env);self.assertTrue(((g>=0)&(g<=1)).all());self.assertTrue(((a>=-1)&(a<=1)).all())
        self.world.step(np.ones((6,2))*500);a2,b2,g2=self.source.commands(self.world.env);self.assertGreater(float((a2-a).abs().sum()),0.)
    def test_28_branch_rng_replay(self):
        another=self.world.clone();a=self.world.step(np.ones((6,2))*700);b=another.step(np.ones((6,2))*700)
        self.assertEqual(a,b);np.testing.assert_allclose(self.world.state(),another.state())
    def test_29_inputs_no_private_future(self):
        f=factorize(self.world,self.source);plans=np.array([[0]*6,[8]*6]);b,g,R=inputs(self.world,f,plans)
        self.assertEqual(b['nominal'].shape,(1,2,7,16,2));self.assertEqual(b['candidate'].shape,(1,2,7,8))
        self.assertTrue(torch.equal(b['candidate'][0,:,-1],torch.zeros(2,8)))
    def test_30_real_action_execution(self):
        model=Predictor().eval();controller=Controller(self.source,model,bank=2,budget=4,rounds=0)
        force,trace=controller.act(self.world);self.assertEqual(force.shape,(6,2));self.assertTrue(np.isfinite(force).all());self.world.step(force)
    def test_31_no_CV_checkpoint(self):
        m=Predictor();self.assertTrue(all('value' not in name and 'critic' not in name for name in m.state_dict()))

if __name__=='__main__':unittest.main(verbosity=2)
