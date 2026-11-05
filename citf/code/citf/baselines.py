"""Ordinary MAPPO and continuous MAAC baselines, shared by all training entries."""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


def initialize(module):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, math.sqrt(2.))
        nn.init.zeros_(module.bias)


class SourceObservationEncoder(nn.Module):
    """Encode the released local observation, including a variable peer set."""
    def __init__(self, width=256,public_state=False):
        if width<4 or width%4:
            raise ValueError('Baseline width must be positive and divisible by four')
        super().__init__();self.public_state=public_state
        self.own_dim=17 if public_state else 14;self.peer_dim=7 if public_state else 2
        self.own=nn.Sequential(nn.Linear(self.own_dim,width//2),nn.Tanh())
        self.peers=nn.Sequential(nn.Linear(self.peer_dim,width//4 if public_state else width//2),nn.Tanh())
        self.output=nn.Sequential(nn.Linear(width,width),nn.Tanh(),nn.Linear(width,width),nn.Tanh())
        self.apply(initialize)

    def forward(self, observation):
        own_dim=getattr(self,'own_dim',14);peer_dim=getattr(self,'peer_dim',2)
        peers=observation[...,own_dim:].reshape(*observation.shape[:-1],-1,peer_dim)
        encoded=self.peers(peers);context=[encoded.mean(-2)]
        if getattr(self,'public_state',False):context.append(encoded.amax(-2))
        return self.output(torch.cat((self.own(observation[...,:own_dim]),*context),-1))


class SourceGaussianPolicy(nn.Module):
    """Shared continuous policy with no forecasts, fusion gate, or credit head."""
    def __init__(self,width=256,state_dependent_std=False,public_state=False):
        super().__init__();self.encoder=SourceObservationEncoder(width,public_state)
        self.mean=nn.Linear(width,2);self.state_dependent_std=state_dependent_std
        if state_dependent_std:self.log_std=nn.Linear(width,2)
        else:self.log_std=nn.Parameter(torch.full((2,),-.5))
        nn.init.orthogonal_(self.mean.weight,.01);nn.init.zeros_(self.mean.bias)
        if state_dependent_std:
            nn.init.zeros_(self.log_std.weight);nn.init.constant_(self.log_std.bias,-.5)

    def distribution(self,observation):
        hidden=self.encoder(observation)
        log_std=self.log_std(hidden) if self.state_dependent_std else self.log_std.expand(*hidden.shape[:-1],2)
        return Normal(self.mean(hidden),log_std.clamp(-5.,2.).exp())

    def forward(self,observation,deterministic=False):
        dist=self.distribution(observation);latent=dist.mean if deterministic else dist.rsample()
        action=latent.tanh()
        logp=(dist.log_prob(latent)-2.*(math.log(2.)-latent-F.softplus(-2.*latent))).sum(-1)
        return action,latent,logp


class SourceCentralValue(nn.Module):
    def __init__(self,width=256,public_state=False):
        super().__init__();self.encoder=SourceObservationEncoder(width,public_state)
        self.value=nn.Sequential(nn.Linear(width*3,width),nn.Tanh(),nn.Linear(width,1))
        self.value.apply(initialize)

    def forward(self,observation):
        hidden=self.encoder(observation);n=hidden.shape[-2]
        context=torch.cat((hidden.mean(-2,keepdim=True),hidden.amax(-2,keepdim=True)),-1)
        return self.value(torch.cat((hidden,context.expand(*hidden.shape[:-2],n,context.shape[-1])),-1)).squeeze(-1)


class SourceAttentionQ(nn.Module):
    """MAAC attention over other agents' observations and executed actions."""
    def __init__(self,width=256,heads=4,public_state=False):
        super().__init__();self.width=width;self.heads=heads
        self.encoder=SourceObservationEncoder(width,public_state)
        self.sa=nn.Sequential(nn.Linear(width+2,width),nn.LeakyReLU(.01))
        self.query=nn.Linear(width,width,bias=False);self.key=nn.Linear(width,width,bias=False)
        self.value=nn.Linear(width,width)
        self.output=nn.Sequential(nn.Linear(width*2,width),nn.LeakyReLU(.01),nn.Linear(width,1))

    def forward(self,observation,action,detach_peers=False,regularize=False):
        hidden=self.encoder(observation);own=self.sa(torch.cat((hidden,action),-1))
        others=own.detach() if detach_peers else own
        batch,n=hidden.shape[:2];dimension=self.width//self.heads
        q=self.query(hidden).reshape(batch,n,self.heads,dimension).transpose(1,2)
        k=self.key(others).reshape(batch,n,self.heads,dimension).transpose(1,2)
        v=self.value(others).reshape(batch,n,self.heads,dimension).transpose(1,2)
        logits=q@k.transpose(-1,-2)/math.sqrt(dimension)
        mask=torch.eye(n,dtype=torch.bool,device=action.device)[None,None]
        weights=logits.masked_fill(mask,-torch.inf).softmax(-1)
        context=(weights@v).transpose(1,2).reshape(batch,n,self.width)
        values=self.output(torch.cat((own,context),-1)).squeeze(-1)
        return (values,1e-3*logits.square().mean()) if regularize else values


class SourceMAPPO:
    """Feedforward MAPPO baseline on the unchanged continuous-control task."""
    def __init__(self,device='cpu',width=256,seed=42,public_state=True):
        torch.manual_seed(seed);self.device=device;self.width=width;self.rng=np.random.default_rng(seed)
        self.public_state=public_state
        self.actor=SourceGaussianPolicy(width,public_state=public_state).to(device)
        self.critic=SourceCentralValue(width,public_state=public_state).to(device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=3e-4,eps=1e-5)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=3e-4,eps=1e-5)
        self.value_mean=0.;self.value_scale=1.;self.normalizer_updates=0

    @torch.no_grad()
    def act(self,observation,deterministic=False):
        obs=torch.as_tensor(observation,dtype=torch.float32,device=self.device)
        action,latent,logp=self.actor(obs,deterministic)
        value=self.critic(obs)*self.value_scale+self.value_mean
        return [x.cpu().numpy() for x in (action,latent,logp,value)]

    @torch.enable_grad()
    def update(self,batch,epochs=10,minibatch=1024):
        obs,latent,old_logp,old_value,reward,done,bootstrap=batch
        advantage=np.zeros_like(reward);following=bootstrap;tail=np.zeros_like(bootstrap)
        for t in reversed(range(len(reward))):
            alive=1.-done[t,:,None]
            delta=reward[t]+.99*following*alive-old_value[t]
            tail=delta+.99*.95*alive*tail;advantage[t]=tail;following=old_value[t]
        returns=advantage+old_value
        mean=float(returns.mean());scale=max(1.,float(returns.std()))
        if self.normalizer_updates:
            mean=.9*self.value_mean+.1*mean
            scale=(.9*self.value_scale**2+.1*scale**2)**.5
        # PopArt preserves unnormalized values while updating their scale.
        with torch.no_grad():
            last=self.critic.value[-1]
            last.weight.mul_(self.value_scale/scale)
            last.bias.mul_(self.value_scale).add_(self.value_mean-mean).div_(scale)
        self.value_mean=mean;self.value_scale=scale;self.normalizer_updates+=1
        advantage=(advantage-advantage.mean())/(advantage.std()+1e-8)
        def tensor(x):
            return torch.as_tensor(x.reshape(-1,*x.shape[2:]),dtype=torch.float32,device=self.device)
        obs,latent,old_logp,old_value,returns,advantage=map(tensor,(obs,latent,old_logp,old_value,returns,advantage))
        old_value=(old_value-mean)/scale;returns=(returns-mean)/scale
        records=[]
        for epoch in range(epochs):
            order=self.rng.permutation(len(obs))
            for start in range(0,len(order),minibatch):
                indices=order[start:start+minibatch];dist=self.actor.distribution(obs[indices])
                z=latent[indices]
                logp=(dist.log_prob(z)-2.*(np.log(2.)-z-F.softplus(-2.*z))).sum(-1)
                ratio=(logp-old_logp[indices]).clamp(-20,20).exp();adv=advantage[indices]
                actor_loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()-.01*dist.entropy().sum(-1).mean()
                value=self.critic(obs[indices]);clipped=old_value[indices]+(value-old_value[indices]).clamp(-.2,.2)
                value_loss=.5*torch.maximum((value-returns[indices]).square(),(clipped-returns[indices]).square()).mean()
                if not torch.isfinite(actor_loss) or not torch.isfinite(value_loss):
                    raise FloatingPointError('Nonfinite MAPPO training objective')
                self.actor_optimizer.zero_grad(set_to_none=True);actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),.5);self.actor_optimizer.step()
                self.critic_optimizer.zero_grad(set_to_none=True);value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(),.5);self.critic_optimizer.step()
                records.append((float(actor_loss.detach()),float(value_loss.detach())))
        return dict(actor_loss=float(np.mean(records,axis=0)[0]),value_loss=float(np.mean(records,axis=0)[1]))


class SourceMAAC:
    """Continuous-action MAAC adaptation with decentralized shared actors."""
    def __init__(self,device='cpu',width=256,seed=42,temperature=.1,public_state=True):
        torch.manual_seed(seed);self.device=device;self.width=width;self.temperature=temperature
        self.public_state=public_state
        self.actor=SourceGaussianPolicy(width,state_dependent_std=True,public_state=public_state).to(device)
        self.critic=SourceAttentionQ(width,public_state=public_state).to(device)
        self.target_actor=copy.deepcopy(self.actor).requires_grad_(False)
        self.target_critic=copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=1e-4,eps=1e-5)
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=1e-4,eps=1e-5,weight_decay=1e-3)

    @torch.no_grad()
    def act(self,observation,deterministic=False):
        obs=torch.as_tensor(observation,dtype=torch.float32,device=self.device)
        return self.actor(obs,deterministic)[0].cpu().numpy()

    @torch.enable_grad()
    def update(self,batch):
        obs,action,reward,following,done=[torch.as_tensor(x,dtype=torch.float32,device=self.device) for x in batch]
        with torch.no_grad():
            future,_,logp=self.target_actor(following)
            target=reward+.99*(1.-done[:,None])*(self.target_critic(following,future)-self.temperature*logp)
        value,regularizer=self.critic(obs,action,regularize=True)
        loss=F.mse_loss(value,target)+regularizer
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite MAAC critic objective')
        self.critic_optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(),10.);self.critic_optimizer.step()
        self.critic.requires_grad_(False)
        candidate,_,logp=self.actor(obs)
        value=self.critic(obs,candidate,detach_peers=True)
        policy_loss=(self.temperature*logp-value).mean()
        if not torch.isfinite(policy_loss):raise FloatingPointError('Nonfinite MAAC actor objective')
        self.actor_optimizer.zero_grad(set_to_none=True);policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(),.5);self.actor_optimizer.step()
        self.critic.requires_grad_(True)
        with torch.no_grad():
            for source,target in zip(self.actor.parameters(),self.target_actor.parameters()):target.lerp_(source,.01)
            for source,target in zip(self.critic.parameters(),self.target_critic.parameters()):target.lerp_(source,.01)
        return dict(actor_loss=float(policy_loss.detach()),value_loss=float(loss.detach()))


