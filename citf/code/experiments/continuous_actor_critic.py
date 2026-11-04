"""Full-task joint continuous-thrust actor and centralized state value.

The strong controller supplies an initial reference, not an action-space hull.
The actor changes both thrusters of every defender throughout the episode.
Only the common hardware bounds constrain executed force. The state-value
critic is used for learning, not as a privileged online outcome input.
"""
from dataclasses import asdict,dataclass
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from citf.bridge import Controller
from citf.contracts import canonical_hash,file_hash
from experiments.contact_event_policy import retained_teacher
from experiments.interaction_dynamics import DynamicsConfig,InteractionDynamics,InteractionLayer
from experiments.synchronized_configuration_policy import joint


@dataclass(frozen=True)
class ContinuousConfig:
    revision:str='CITF-full-task-continuous-actor-critic-v1'
    teacher_checkpoint:str=''
    encoder_checkpoint:str=''
    width:int=64
    layers:int=2
    heads:int=4
    history:int=8
    context:bool=True
    residual_scale:float=1500.
    initial_std:float=.06


class ObservedEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=DynamicsConfig(history=cfg.history,width=cfg.width,layers=cfg.layers,
            heads=cfg.heads,context=cfg.context)
        self.embedding=nn.Sequential(nn.Linear(cfg.history*8+8,cfg.width),nn.SiLU(),nn.Linear(cfg.width,cfg.width))
        self.layers=nn.ModuleList([InteractionLayer(cfg.width,cfg.heads) for _ in range(cfg.layers)])

    def forward(self,history,remaining):
        return InteractionDynamics.encode(self,history,remaining)


class ContinuousActorCritic(nn.Module):
    def __init__(self,cfg=ContinuousConfig()):
        super().__init__();self.cfg=cfg;w=cfg.width
        self.encoder=ObservedEncoder(cfg)
        self.command=nn.Linear(2,w)
        self.action_context=InteractionLayer(w,cfg.heads)
        self.actor=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,2))
        self.log_std=nn.Parameter(torch.full((2,),float(np.log(cfg.initial_std))))
        self.critic=nn.Sequential(nn.LayerNorm(3*w+3),nn.Linear(3*w+3,2*w),nn.SiLU(),nn.Linear(2*w,1))
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)

    def forward(self,history,remaining,reference=None):
        value=self.encoder(history,remaining)
        position=history[:,-1,:,:2]
        radius=position.norm(dim=-1)
        distance=(position[:,:-1]-position[:,-1:]).norm(dim=-1).amin(1)
        n=position.shape[1]-1;i,j=torch.triu_indices(n,n,1,device=position.device)
        separation=(position[:,i]-position[:,j]).norm(dim=-1).amin(1)
        radial=radius[:,-1]-radius[:,:-1].amin(1)
        geometry=torch.stack((distance/60,separation/20,radial/30),-1)
        pooled=torch.cat((value[:,:-1].mean(1),value[:,:-1].amax(1),value[:,-1],geometry),-1)
        state_value=1.05*self.critic(pooled).squeeze(-1).tanh()
        if reference is None:return dict(value=state_value)
        command=torch.cat(((reference-250)/750,reference.new_zeros((len(reference),1,2))),1)
        conditioned=value+self.command(command)
        angle=torch.atan2(-position[:,-1,1],-position[:,-1,0]);c,s=angle.cos(),angle.sin()
        canonical=torch.stack((c[:,None]*position[...,0]+s[:,None]*position[...,1],
            -s[:,None]*position[...,0]+c[:,None]*position[...,1]),-1)
        delta=canonical[:,:,None]-canonical[:,None,:];pair=delta.norm(dim=-1)
        relative=torch.cat((delta/20,(pair/20)[...,None],(1/(pair+1))[...,None],
            ((radius[:,:,None]-radius[:,None,:])/20)[...,None]),-1)
        conditioned=self.action_context(conditioned,relative,self.cfg.context)
        mean=self.actor(conditioned[:,:-1])
        return dict(mean=mean,value=state_value,std=self.log_std.clamp(-5.,-1.).exp().expand_as(mean))

    def distribution(self,history,remaining,reference):
        output=self(history,remaining,reference)
        return Normal(output['mean'],output['std']),output['value']

    def force(self,reference,latent):
        return (reference+self.cfg.residual_scale*latent.tanh()).clamp(-500.,1000.)

    def initialize_encoder(self,path):
        value=torch.load(path,map_location='cpu',weights_only=False)
        state={k:v for k,v in value['state'].items() if k.startswith(('embedding.','layers.'))}
        self.encoder.load_state_dict(state)


def load_policy(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=ContinuousActorCritic(ContinuousConfig(**payload['config'])).to(device)
    model.load_state_dict(payload['state']);model.eval()
    return model


def observed_history(world,length=8):
    history=np.stack(world.history).astype(np.float32)
    return np.concatenate((np.repeat(history[:1],length-len(history),0),history),0)


class ContinuousController(Controller):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.teacher=joint.JointPolicyController(self.source,retained_teacher(self.model.cfg.teacher_checkpoint),
            self.device,self.bank,self.budget,self.rounds,self.risk_mode)

    @torch.no_grad()
    def act(self,world):
        started=time.perf_counter();reference,trace=self.teacher.act(world)
        history=torch.as_tensor(observed_history(world),device=self.device)[None]
        remaining=torch.tensor([world.protocol.time_limit-world.env.Current_T],device=self.device)
        reference_tensor=torch.as_tensor(reference,dtype=torch.float32,device=self.device)[None]
        output=self.model(history,remaining,reference_tensor)
        force=self.model.force(reference_tensor,output['mean'])[0].cpu().numpy()
        trace.update(revision=self.model.cfg.revision,physical_thrust=force.tolist(),
            reference_physical_thrust=reference.tolist(),latent_residual=output['mean'][0].cpu().tolist(),
            physical_residual=(force-reference).tolist(),actor_forward_batches=1,
            source_domain='full common physical thrust bounds,reference plus learned continuous residual',
            control_period=world.protocol.control_dt,compute_seconds=time.perf_counter()-started)
        trace['forward_batches']+=1
        return force,trace


def main():
    import citf.evaluate as evaluation
    import experiments.interaction_dynamics as dynamics
    original=evaluation.runtime_identity
    def identity(*args,**kwargs):
        result=original(*args,**kwargs)
        checkpoint=args[2] if len(args)>2 else kwargs.get('checkpoint')
        value=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=value['config']
        result.update(method='CITF-full-task-continuous-actor-critic-development',revision=cfg['revision'])
        result['controller'].update(action='joint continuous12-channel physical residual,1500*tanh(latent),then common[-500,1000]N bounds',
            control='all task states,every0.2s,no event/activation margin',
            learning='centralized state-value and on-policy complete-episode reward withPPO/GAE',
            online_inputs='joint observed history,remaining time,current strong-reference force',
            teacher_checkpoint_sha256=file_hash(cfg['teacher_checkpoint']),
            encoder_initialization_sha256=file_hash(cfg['encoder_checkpoint']),
            training_update=value.get('update'),training_environment_steps=value.get('environment_steps'),
            reference_has_learning=True,expanded_control_domain=True)
        result['runtime_code']=canonical_hash(dict(core=result['runtime_code'],policy=file_hash(__file__),
            encoder=file_hash(dynamics.__file__)))
        return result
    evaluation.runtime_identity=identity
    evaluation.load_behavior=lambda checkpoint,source,world,device:load_policy(checkpoint,device)
    evaluation.Controller=ContinuousController
    evaluation.main()


if __name__=='__main__':main()
