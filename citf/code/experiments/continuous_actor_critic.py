"""Full-task continuous control with selectable teacher, source or direct reference."""
from dataclasses import asdict,dataclass
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from citf.bridge import Controller
from citf.contracts import canonical_hash,file_hash,Protocol
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
    reference_mode:str='teacher'
    teacher_sha256:str=''

    def __post_init__(self):
        if self.reference_mode not in ('teacher','source','direct'):
            raise ValueError('Unknown control reference mode')
        if self.history < 1 or self.residual_scale <= 0 or not np.isfinite(self.residual_scale):
            raise ValueError('History and residual scale must be positive')
        if self.initial_std <= 0 or not np.isfinite(self.initial_std):
            raise ValueError('Initial exploration scale must be positive')


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

    def validate_runtime(self,source,world):
        if world is None:return
        if world.protocol.history_steps<self.cfg.history:
            raise ValueError('Runtime history buffer is shorter than the trained actor window')
        trained=getattr(self,'training_protocol',None)
        if trained:
            before=asdict(Protocol(**trained));after=asdict(world.protocol)
            changes={key:dict(training=before[key],execution=value) for key,value in after.items() if before[key]!=value}
            if set(changes)-{'name','radial_early_failure','apf_obstacle_rule','history_steps','prediction_steps','risk_weight','risk_clearance','prior_scale'}:
                raise ValueError('Continuous policy evaluation changed task duration or physical dynamics')
            self.protocol_transfer=changes


def load_policy(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=ContinuousActorCritic(ContinuousConfig(**payload['config'])).to(device)
    model.load_state_dict(payload['state']);model.eval()
    model.training_protocol=payload.get('physical_protocol',payload.get('protocol',{}).get('physical_protocol'))
    return model


def observed_history(world,length=8):
    history=np.stack(list(world.history)[-length:]).astype(np.float32)
    return np.concatenate((np.repeat(history[:1],length-len(history),0),history),0)


class ReferenceController:
    """Explicit online reference, including a teacher-free direct policy mode."""
    def __init__(self, source, cfg, device='cpu', bank=12, budget=96, rounds=1, risk_mode='expected'):
        self.source, self.mode = source, cfg.reference_mode
        teacher=retained_teacher(cfg.teacher_checkpoint,cfg.teacher_sha256) if self.mode=='teacher' else None
        if teacher is not None:
            for key,actual in (('source_sha256',source.sha256),('source_implementation_sha256',source.implementation_sha256)):
                expected=teacher.checkpoint_protocol.get(key)
                if expected and expected!=actual:
                    raise ValueError('Online teacher uses a different source controller: '+key)
        self.teacher = (joint.JointPolicyController(source, teacher,
            device, bank, budget, rounds, risk_mode) if self.mode == 'teacher' else None)

    def act(self, world):
        if self.teacher is not None:
            cfg=self.teacher.model.cfg
            if (world.protocol.history_steps,world.protocol.prediction_steps)!=(cfg.history,cfg.horizon):
                raise ValueError('Teacher input history/horizon differs from its trained network')
            return self.teacher.act(world)
        if self.mode == 'source':
            learned, boids, gates = self.source.commands(world.env)
            force = (250 + 750 * (gates * learned + (1-gates) * boids)).numpy()
        else:
            force = np.full((world.env.defender_num, 2), 250., np.float32)
        return force, dict(evaluated_unique=int(self.mode == 'source'),
                          forward_batches=int(self.mode == 'source'), reference_mode=self.mode)


class ContinuousController(Controller):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.teacher=ReferenceController(self.source,self.model.cfg,
            self.device,self.bank,self.budget,self.rounds,self.risk_mode)

    @torch.no_grad()
    def act(self,world):
        started=time.perf_counter();reference,trace=self.teacher.act(world)
        history=torch.as_tensor(observed_history(world,self.model.cfg.history),device=self.device)[None]
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
            reference_mode=cfg.get('reference_mode','teacher'),
            teacher_used_online=cfg.get('reference_mode','teacher')=='teacher',
            source_used_online=cfg.get('reference_mode','teacher')!='direct',
            teacher_checkpoint_sha256=(file_hash(cfg['teacher_checkpoint']) if cfg.get('reference_mode','teacher')=='teacher'
                                       else value.get('teacher_sha256')),
            encoder_initialization_sha256=value.get('encoder_initialization_sha256'),
            training_update=value.get('update'),training_environment_steps=value.get('environment_steps'),
            reference_has_learning=cfg.get('reference_mode','teacher')!='direct',expanded_control_domain=True)
        result['runtime_code']=canonical_hash(dict(core=result['runtime_code'],policy=file_hash(__file__),
            encoder=file_hash(dynamics.__file__)))
        return result
    evaluation.runtime_identity=identity
    evaluation.load_behavior=lambda checkpoint,source,world,device:load_policy(checkpoint,device)
    evaluation.Controller=ContinuousController
    evaluation.main()


if __name__=='__main__':main()
