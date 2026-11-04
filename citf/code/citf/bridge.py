"""Self-contained, audited adapter to the PROVIDED source80 environment.

No old CV loader, data collector, vector_world or missing study_runtime import.
This is a new runtime revision and must not inherit 46/48 as its own result.
"""
from __future__ import annotations
import copy,sys,hashlib,time
from pathlib import Path
from collections import deque
import numpy as np
import torch
from .contracts import Protocol,file_hash,geometry_identity,canonical_hash
from .physics import forecast
from .search import candidate_grid,nominal_beam,nominal_score,whole_configuration_search,initialize_bank

class Source:
    def __init__(self,project,checkpoint,device='cpu'):
        path=str(Path(project)/'train')
        if path not in sys.path:sys.path.insert(0,path)
        from policy.networks import ActorAdap
        self.net=ActorAdap(6,8,3,512).to(device);self.device=device
        raw=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if 'actor' in raw:
            weights={k[5:]:v for k,v in raw['actor'].items() if k.startswith('base.')}
        else:weights=raw
        self.net.load_state_dict(weights,strict=True);self.net.eval().requires_grad_(False)
        self.sha256=file_hash(checkpoint)
        self.implementation_sha256=file_hash(Path(project)/'train/policy/networks.py')
    @torch.no_grad()
    def commands(self,env):
        obs,_=env._get_obs();a,_=self.net(torch.tensor(obs,dtype=torch.float32,device=self.device),deterministic=True,with_logprob=False)
        a=a.cpu();g=torch.sigmoid(2*torch.tanh(torch.logit(.02+.96*a[:,2:3])/2)).expand(-1,2)
        return a[:,:2],torch.tensor((env.boids_actions-250)/750,dtype=torch.float32),g

class World:
    def __init__(self,project,protocol=Protocol(),defenders=6):
        path=str(Path(project)/'train')
        if path not in sys.path:sys.path.insert(0,path)
        from envs.TADgame import TADEnv
        self.env=TADEnv(defenders,protocol=protocol.environment,total_time=protocol.time_limit)
        self.protocol=protocol;self.history=deque(maxlen=protocol.history_steps);self.done=False
        self.rng=np.random.RandomState(0)
        self.initial_identity=None
        checks={'Action_T':protocol.control_dt,'Total_T':protocol.time_limit,'Target_R':protocol.target_radius,'Defend_R':protocol.capture_radius,'Collision_R':protocol.collision_radius,'early_attacker_win':protocol.radial_early_failure}
        for key,value in checks.items():
            if getattr(self.env,key)!=value:raise ValueError('Effective environment differs from protocol '+key)
        for boat in [*self.env.defender_list,self.env.attacker]:
            if boat.dt!=protocol.integration_dt or abs(boat.dt*boat.N-protocol.control_dt)>1e-9:raise ValueError('Integration timing mismatch')
        self.environment_sha256=hashlib.sha256((Path(project)/'train/envs/TADgame.py').read_bytes()+(Path(project)/'train/envs/modules.py').read_bytes()).hexdigest()
    def _call(self,fun,*args,**kwargs):
        old=np.random.get_state();np.random.set_state(self.rng.get_state())
        try:r=fun(*args,**kwargs);self.rng.set_state(np.random.get_state());return r
        finally:np.random.set_state(old)
    def state(self):return np.stack([x.motion_state() for x in [*self.env.defender_list,self.env.attacker]])
    def reset(self,scene_seed,future_seed,agility):
        self.rng=np.random.RandomState(scene_seed);self._call(self.env.reset,agility=agility,noisy_agility=False)
        self.rng=np.random.RandomState(future_seed);self.history.clear();self.history.append(self.state().copy());self.done=False
        self.initial_identity=geometry_identity(self.state(),self.env.defender_num)
    def step(self,force,attacker_action=None):
        if self.done:raise ValueError('Cannot step terminated task')
        force=np.asarray(force,dtype=float)
        if force.shape!=(self.env.defender_num,2) or not np.isfinite(force).all() or (force< -500.001).any() or (force>1000.001).any():raise ValueError('Invalid physical command')
        if attacker_action is not None:
            attacker_action=np.asarray(attacker_action,dtype=float)
            if attacker_action.shape!=(2,) or not np.isfinite(attacker_action).all() or (abs(attacker_action)>1.000001).any():raise ValueError('Invalid normalized attacker action')
        _,_,code,_=self._call(self.env.step,np.zeros_like(force),defender_thrust=force,att_action=attacker_action)
        self.history.append(self.state().copy());self.done=bool(code)
        # Independent physical event flags supplement the mutually-exclusive primary outcome.
        events=self.env.physical_events()
        breach=bool(events['breach']);collision=bool(events['collision'])
        return dict(outcome=int(code),success=int(code>2),capture=int(code==3),collision=int(code==2),
            physical_collision=int(collision),physical_breach=int(breach),source_failure=int(code==1),
            radial_only_failure=int(code==1 and not breach),timeout=int(code==4),duration=float(self.env.Current_T))
    def clone(self):return copy.deepcopy(self)
    def observation(self):
        # Rotate ALL history into the same CURRENT target-centred frame.
        hist=np.stack(self.history);E=hist.shape[1];L=self.protocol.history_steps
        angle=np.arctan2(-hist[-1,-1,1],-hist[-1,-1,0]);c,s=np.cos(angle),np.sin(angle)
        R=np.array([[c,-s],[s,c]])
        padding=np.repeat(hist[:1],L-len(hist),axis=0);hist=np.concatenate((padding,hist),0)
        x=hist[...,:2]@R;v=hist[...,3:5]@R;yaw=hist[...,2]-angle
        role=np.zeros((L,E,1));role[:,-1]=1
        feat=np.concatenate((x/60,v/5,np.sin(yaw)[...,None],np.cos(yaw)[...,None],hist[...,5:6],role),-1)
        hv=np.zeros((E,L),bool);hv[:,-len(self.history):]=True
        return torch.tensor(feat.transpose(1,0,2),dtype=torch.float32),torch.tensor(x[-1],dtype=torch.float32),torch.tensor(hv),R

@torch.no_grad()
def factorize(world,source):
    p=world.protocol;a,b,g=source.commands(world.env);grid,valid=candidate_grid(g)
    cmd=grid*a[:,None]+(1-grid)*b[:,None];force=750*cmd+250
    actual=torch.tensor(world.state()[:-1],dtype=torch.float64)
    state=actual[:,None].expand(-1,9,-1)
    paths=forecast(state,force.double(),p.prediction_steps,p.control_dt).float()
    L=p.known_steps(world.env.Current_T)
    if L<1:raise ValueError('No legal future control step')
    distances=(paths[:,None,:,None,:L]-paths[None,:,None,:,:L]).norm(dim=-1).amin(-1)
    pair=(p.risk_clearance-distances).clamp_min(0).square()
    unary=((grid-g[:,None])/p.prior_scale).square().mean(-1).masked_fill(~valid,torch.inf)
    return dict(grid=grid,valid=valid,unary=unary,pair=pair,paths=paths,learned=a,rules=b,prior=g)

@torch.no_grad()
def inputs(world,f,plans):
    plans=torch.as_tensor(plans,dtype=torch.long);C,N=plans.shape
    rows=torch.arange(N)[None].expand(C,-1)
    gates=f['grid'][rows,plans];paths=f['paths'][rows,plans]
    hist,pos,hv,R=world.observation();T=world.protocol.prediction_steps
    paths=torch.matmul(paths,torch.tensor(R,dtype=paths.dtype))
    vel=torch.tensor(world.state()[-1,3:5]@R,dtype=torch.float32)
    enemy=pos[-1]+torch.arange(1,T+1)[:,None]*world.protocol.control_dt*vel
    nominal=torch.cat((paths,enemy[None,None].expand(C,1,T,2)),1)
    a=f['learned'][None].expand(C,-1,-1);b=f['rules'][None].expand(C,-1,-1);mix=gates*a+(1-gates)*b
    cand=torch.cat((gates,a,b,mix),-1);cand=torch.cat((cand,torch.zeros(C,1,8)),1)
    u=f['unary'][rows,plans].sum(-1)
    return dict(history=hist[None],positions=pos[None],history_valid=hv[None],entity_valid=torch.ones(1,N+1,dtype=torch.bool),
        nominal=nominal[None],candidate=cand[None],unary=u[None],candidate_valid=torch.ones(1,C,dtype=torch.bool),
        time_context=torch.tensor([[max(0.,world.protocol.time_limit-world.env.Current_T)/world.protocol.time_limit,world.protocol.known_steps(world.env.Current_T)/T]],dtype=torch.float32),
        known_time_valid=(torch.arange(T)<world.protocol.known_steps(world.env.Current_T))[None]),gates,R

class Controller:
    def __init__(self,source,model=None,device='cpu',bank=12,budget=96,rounds=1,risk_mode='expected'):
        self.source=source;self.model=model;self.device=device;self.bank=bank;self.budget=budget;self.rounds=rounds;self.risk_mode=risk_mode
        if risk_mode not in ('expected','mean_distance'):raise ValueError('Unknown geometric risk mode')
        if bank<1 or budget<1 or rounds<0:raise ValueError('Invalid search configuration')
        if model is not None:model.to(device).eval()
    @torch.no_grad()
    def act(self,world):
        start=time.perf_counter();f=factorize(world,self.source);p=world.protocol
        physical_done=time.perf_counter()
        if self.model is not None:
            cfg=self.model.cfg
            if (cfg.horizon,cfg.history,cfg.dt,cfg.clearance)!=(p.prediction_steps,p.history_steps,p.control_dt,p.risk_clearance):raise ValueError('Predictor and online protocol differ')
        initial,_=nominal_beam(f['unary'].numpy(),f['pair'].numpy(),p.risk_weight,128)
        initial,centre=initialize_bank(initial,f['grid'].numpy(),f['valid'].numpy(),f['prior'].numpy(),self.bank,self.budget)
        history_cache=None;prediction_trace=[]
        if self.model is not None:
            first,_,_=inputs(world,f,initial[:1]);first={k:v.to(self.device) for k,v in first.items()}
            history_cache=self.model.encode_history(first)
        encoded=time.perf_counter()
        # Each complete candidate is scored independently from its batch companions.
        def evaluate(plans):
            if self.model is None:return nominal_score(plans,f['unary'].numpy(),f['pair'].numpy(),p.risk_weight)
            b,_,_=inputs(world,f,plans);b={k:v.to(self.device) for k,v in b.items()};out=self.model(b,history_cache=history_cache)
            prediction_trace.append(dict(configurations=np.asarray(plans).tolist(),mu=out['mu'][0].cpu().numpy().tolist(),sigma=out['sigma'][0].cpu().numpy().tolist(),risk=out['risk'][0].cpu().numpy().tolist()))
            risk=out['risk'] if self.risk_mode=='expected' else out['det_risk']
            return (b['unary']+p.risk_weight*(risk*out['pair_valid'][:,None]).sum(-1))[0].cpu().numpy()
        if getattr(self,'selection','joint')=='independent':
            from experiments.independent_select import independent_select
            ix,meta=independent_select(centre,[x.nonzero()[:,0].numpy() for x in f['valid']],evaluate,budget=self.budget)
        else:
            ix,meta=whole_configuration_search(initial,evaluate,[x.nonzero()[:,0].numpy() for x in f['valid']],self.budget,self.rounds)
        g=f['grid'][torch.arange(len(ix)),torch.tensor(ix)];force=(750*(g*f['learned']+(1-g)*f['rules'])+250).numpy()
        meta.update(initial_configurations=initial.tolist(),source_center_indices=centre.tolist(),source_center_scored=centre.tolist() in meta['configurations'],budget_limit=self.budget,
            budget_utilization=meta['evaluated_unique']/self.budget,gates=g.numpy().tolist(),selected_indices=ix.tolist(),source_learned=f['learned'].numpy().tolist(),source_rules=f['rules'].numpy().tolist(),prior=f['prior'].numpy().tolist(),physical_thrust=force.tolist(),prediction_batches=prediction_trace,
            time_context=first['time_context'][0].cpu().tolist() if self.model is not None else [max(0.,p.time_limit-world.env.Current_T)/p.time_limit,p.known_steps(world.env.Current_T)/p.prediction_steps],
            timings={'source_and_nominal_s':physical_done-start,'history_and_initial_search_s':encoded-physical_done,'search_and_readout_s':time.perf_counter()-encoded})
        return force,meta
