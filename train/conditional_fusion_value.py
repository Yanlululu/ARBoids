"""Candidate-conditioned values with anchored, single-vessel credit learning."""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def local_fusion_context(packet, proposals, left, right, peer=True):
    """Permutation and rotation invariant context of a conditional gate factor.

    All quantities are current public measurements. Other vessels enter through
    pooled relative geometry; neither a scene identifier nor a future outcome is
    an input. Candidate commands retain their physical left/right meaning.
    """
    motion = packet['motion']
    pos, vel, heading = motion[..., :2], motion[..., 3:5], motion[..., 2]
    node = packet['node']
    target_body = node[..., 2:4] * 30.
    c, s = heading.cos(), heading.sin()
    target = pos + torch.stack((c*target_body[..., 0]-s*target_body[..., 1],
                               s*target_body[..., 0]+c*target_body[..., 1]), -1)
    center = (pos[:, left] + pos[:, right]) * .5
    direction = target[:, left] - center
    direction = direction / direction.norm(dim=-1, keepdim=True).clamp_min(1.)
    def rotate(vector):
        return torch.stack(((vector*direction).sum(-1),
            direction[..., 0]*vector[..., 1]-direction[..., 1]*vector[..., 0]), -1)
    relative = rotate(pos[:, right]-pos[:, left]) / 10.
    hi = torch.stack((c[:, left], s[:, left]), -1)
    hj = torch.stack((c[:, right], s[:, right]), -1)
    distances = (pos[:, :, None]-pos[:, None]).norm(dim=-1)
    n = pos.shape[1]
    mask = ~torch.eye(n, dtype=torch.bool, device=pos.device)
    proximity = (torch.exp(-distances.square()/100.)*mask).sum(-1)
    own = torch.cat((proposals[:, left], node[:, left, 11:13]), -1)
    other = torch.cat((proposals[:, right], node[:, right, 11:13]), -1)
    if not peer: other = torch.zeros_like(other)
    return torch.cat((relative, rotate(vel[:, left])/3., rotate(vel[:, right])/3.,
        rotate(hi), rotate(hj), motion[:, left, 5:6], motion[:, right, 5:6],
        target_body[:, left].norm(dim=-1, keepdim=True)/30.,
        target_body[:, right].norm(dim=-1, keepdim=True)/30.,
        pos[:, left].norm(dim=-1, keepdim=True)/30.,
        pos[:, right].norm(dim=-1, keepdim=True)/30.,
        proximity[:, left, None], proximity[:, right, None], own, other), -1)


def response_fusion_context(packet, proposals, risk, left, right, peer=True):
    """Share return effects by candidate response, instead of encounter pose.

    Pairwise changes across the entire fusion grid preserve which partner
    candidate makes an own intervention useful. The no-peer representation
    retains the same dimensions using its own constant-neighbor prediction.
    """
    state = local_fusion_context(packet, proposals, left, right, peer)
    if peer:
        table = risk[:, left, right]
    else:
        own = risk[:, left]
        table = own[..., None].expand(-1, -1, own.shape[-1])
    table = table.clamp_max(36.) / 9.
    interaction = table-table.mean(-1, keepdim=True)-table.mean(-2, keepdim=True)+table.mean((-1,-2), keepdim=True)
    return torch.cat((state[:, 12:14], state[:, 16:26], table.flatten(1), interaction.flatten(1)), -1)


class LocalConditionalValue(nn.Module):
    """Locally supported joint values fitted from single-vessel return edges.

    The graph incidence solve removes the state-value nuisance exactly. Kernel
    interpolation then shares conditional effects between nearby encounters,
    instead of asking a global network to extrapolate terminal differences.
    """
    def __init__(self, centers, values, support, bandwidth=1., strength=16., peer=True, prior=None, encoder='state'):
        super().__init__()
        self.register_buffer('centers', torch.as_tensor(centers, dtype=torch.float32))
        self.register_buffer('values', torch.as_tensor(values, dtype=torch.float32))
        self.register_buffer('support', torch.as_tensor(support, dtype=torch.float32))
        self.bandwidth = bandwidth
        self.strength = strength
        self.peer = peer
        self.encoder = encoder
        self.prior = LocalConditionalValue(**prior) if prior is not None else None

    def forward(self, context):
        distance = (context[..., None, :]-self.centers).square().sum(-1)
        count = min(8, len(self.centers))
        nearest, index = distance.topk(count, largest=False)
        weights = torch.exp(-nearest/(2.*self.bandwidth**2))*self.support[index]
        # A zero-effect prior prevents remote examples from acquiring authority
        # solely because their normalized kernel weights sum to one.
        value = (weights[..., None, None]*self.values[index]).sum(-3)
        result = self.strength*value/(.25+weights.sum(-1))[..., None, None]
        return result if self.prior is None else result+self.prior(context)

    def record(self):
        return dict(centers=self.centers.cpu(), values=self.values.cpu(),
            support=self.support.cpu(), bandwidth=self.bandwidth,
            strength=self.strength, peer=self.peer,
            prior=None if self.prior is None else self.prior.record(), encoder=self.encoder)


@torch.no_grad()
def coupled_pair_responses(actor,packet,gates,baseline,neural_cache,span):
    """Propagate each pair's mutual Boids feedback under both candidates.

    The frozen neural predictions are reused from the individual rollouts;
    each active pair recomputes its original Boids candidate in its own joint
    predicted geometry. Remaining vessels retain public velocity forecasts.
    This is a representation inside the Adapter, never an executed thrust.
    """
    motion=packet['motion'];b,n=motion.shape[:2];device=motion.device
    pairs=[(i,j) for i in range(n) for j in range(i+1,n)]
    options=torch.arange(16,device=device);selectors=[];active=[];first=[];second=[]
    for i,j in pairs:
        selection=torch.zeros((16,n),dtype=torch.long,device=device)
        selection[:,i]=options//4;selection[:,j]=options%4
        mask=torch.zeros((16,n),dtype=torch.bool,device=device);mask[:,i]=True;mask[:,j]=True
        selectors.append(selection);active.append(mask);first.extend([i]*16);second.extend([j]*16)
    selector=torch.cat(selectors);active=torch.cat(active)[None,:,:,None]
    first=torch.tensor(first,device=device);second=torch.tensor(second,device=device)
    m=len(selector);boat=torch.arange(n,device=device)[None].expand(m,-1);world=torch.arange(m,device=device)
    selected=gates[:,boat,selector]
    state=motion[:,None].expand(-1,m,-1,-1).clone();minimum=None;checkpoints=[];dt=.1
    for step in range(32):
        if step%2==0:
            if step:_,boids=actor.feedback_observation(packet,state,step*dt,joint=True)
            else:boids=packet['node'][:,None,:,11:13].expand(-1,m,-1,-1)
            current=selected if step<2*span else baseline[:,None]
            neural=neural_cache[step][:,boat,selector]
            force=750.*(current[...,None]*neural+(1.-current[...,None])*boids)+250.
        x,y,h,vx,vy,r=state.unbind(-1)
        x=x+dt*vx;y=y+dt*vy;h=h+dt*r
        c,s=h.cos(),h.sin();u=c*vx+s*vy;v=-s*vx+c*vy
        du=(400.*r*v-(100.+150.*u.abs())*u+force.sum(-1))/380.
        dv=(-380.*r*u-(100.+150.*v.abs())*v)/400.
        dr=(-20.*u*v-(980.+950.*r.abs())*r+1.25*(force[...,0]-force[...,1]))/1430.
        u=u+dt*du;v=v+dt*dv
        following=torch.stack((x,y,h,c*u-s*v,s*u+c*v,r+dt*dr),-1)
        constant=motion[:,None].expand(-1,m,-1,-1).clone()
        constant[...,:2]+=((step+1)*dt)*motion[:,None,:,3:5]
        state=torch.where(active,following,constant)
        distance=(state[:,world,first,:2]-state[:,world,second,:2]).norm(dim=-1)
        minimum=distance if minimum is None else torch.minimum(minimum,distance)
        if step+1 in (4,8,16,32):checkpoints.append(minimum)
    return torch.stack(checkpoints,-1).reshape(b,len(pairs),4,4,4)


@torch.no_grad()
def fusion_response_factors(actor, packet, intent, span):
    """Mode-aware nominal response factors for finite fusion commitments.

    Option zero uses the current adaptive fusion; the other options apply a
    scalar mixture for ``span`` source steps and then restore that fusion.
    Mode flags distinguish replanning from commitment even when their initial
    mixtures coincide; subsequent feedback effects are learned from paired
    continuations. Training and deployment use the same encoding.
    No simulator state, realized current, scene key, or outcome is consumed.
    """
    motion=packet['motion'];b,n=motion.shape[:2];device=motion.device
    proposals=actor.proposal_distribution(packet['obs'])[0].tanh()
    baseline=actor.fast_action(packet,intent)[...,2] if actor.intent_strength else intent[...,2]
    gates=torch.cat((baseline[...,None],motion.new_tensor([0.,.5,1.]).expand(b,n,-1)),-1)
    state=motion[:,:,None].expand(-1,-1,4,-1).clone()
    paths=[];neural_cache=[];dt=.1
    neural=proposals[:,:,None].expand(-1,-1,4,-1)
    force=750.*(gates[...,None]*proposals[:,:,None]+(1.-gates[...,None])*packet['node'][:,:,None,11:13])+250.
    for step in range(32 if getattr(actor,'interaction_enabled',True) else 0):
        if step and step%2==0:
            obs,boids=actor.feedback_observation(packet,state.transpose(1,2),step*dt)
            future=actor.proposal_distribution(obs.reshape(b*4,n,-1))[0].tanh().reshape(b,4,n,2).transpose(1,2)
            neural=future
            current=gates if step<2*span else baseline[...,None].expand_as(gates)
            force=750.*(current[...,None]*future+(1.-current[...,None])*boids.transpose(1,2))+250.
        x,y,h,vx,vy,r=state.unbind(-1)
        x=x+dt*vx;y=y+dt*vy;h=h+dt*r
        c,s=h.cos(),h.sin();u=c*vx+s*vy;v=-s*vx+c*vy
        du=(400.*r*v-(100.+150.*u.abs())*u+force.sum(-1))/380.
        dv=(-380.*r*u-(100.+150.*v.abs())*v)/400.
        dr=(-20.*u*v-(980.+950.*r.abs())*r+1.25*(force[...,0]-force[...,1]))/1430.
        u=u+dt*du;v=v+dt*dv
        state=torch.stack((x,y,h,c*u-s*v,s*u+c*v,r+dt*dr),-1)
        paths.append(state[...,:2])
        neural_cache.append(neural)
    times=torch.arange(1,33,device=device)*dt
    constant=motion[:,:,:2][:,:,None]+times[None,None,:,None]*motion[:,:,3:5][:,:,None]
    paths=(torch.stack(paths,-2) if paths else constant[:,:,None].expand(-1,-1,4,-1,-1))
    feature_version=getattr(actor,'response_feature_version',1)
    coupled=(coupled_pair_responses(actor,packet,gates,baseline,neural_cache,span)
        if actor.peer and feature_version==3 else None)
    encoded=[];pair_index=0
    for i in range(n):
        for j in range(i+1,n):
            directions=[]
            for left,right in ((i,j),(j,i)):
                own=paths[:,left,:,None]
                other=(paths[:,right,None] if actor.peer else constant[:,right,None,None].expand(-1,1,4,-1,-1))
                distance=(other-own).norm(dim=-1)
                features=[local_fusion_context(packet,proposals,left,right,actor.peer)[:,None,None].expand(-1,4,4,-1)]
                # The entire response profile distinguishes a short safe pulse
                # from a fusion that only postpones a collision.
                for checkpoint,end in enumerate((4,8,16,32)):
                    minimum=distance[...,:end].amin(-1)
                    features.extend(((minimum/10.).clamp_max(4.)[...,None],
                        ((7.-minimum).clamp_min(0.)/5.)[...,None]))
                    if getattr(actor,'response_feature_version',1)>1:
                        change=minimum-minimum[:,:1,:1]
                        interaction=minimum-minimum[:,:,:1]-minimum[:,:1,:]+minimum[:,:1,:1]
                        features.extend(((change/.5)[...,None],(interaction/.5)[...,None]))
                    if feature_version==3:
                        response=(coupled[:,pair_index,:,:,checkpoint] if coupled is not None else torch.zeros_like(minimum))
                        if left!=i:response=response.transpose(1,2)
                        correction=(response-minimum)/.5 if actor.peer else torch.zeros_like(minimum)
                        features.extend(((response/10.)[...,None],correction[...,None]))
                direction=motion[:,right,:2]-motion[:,left,:2]
                direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1.)
                def components(delta):
                    along=(delta*direction[:,None]).sum(-1)/5.
                    across=(delta[...,1]*direction[:,None,0]-delta[...,0]*direction[:,None,1])/5.
                    return torch.stack((along,across),-1)
                own_delta=components(paths[:,left,:,-1]-paths[:,left,:1,-1])
                peer_delta=components(paths[:,right,:,-1]-paths[:,right,:1,-1]) if actor.peer else torch.zeros_like(own_delta)
                features.extend((own_delta[:,:,None].expand(-1,-1,4,-1),peer_delta[:,None].expand(-1,4,-1,-1),
                    gates[:,left,:,None,None].expand(-1,-1,4,-1),gates[:,right,None,:,None].expand(-1,4,-1,-1),
                    motion.new_full((b,4,4,1),span/10.)))
                if feature_version>=4:
                    # Equal initial mixtures are different interventions:
                    # option zero replans, whereas a fixed option commits.
                    # Omitting this distinction aliases nonzero return edges
                    # into identical critic inputs and makes credit unlearnable.
                    fixed=(torch.arange(4,device=device)>0).to(motion.dtype)
                    own_fixed=fixed[None,:,None,None].expand(b,-1,4,-1)
                    peer_fixed=fixed[None,None,:,None].expand(b,4,-1,-1)
                    features.extend((own_fixed,peer_fixed,
                        own_fixed*(span/10.),peer_fixed*(span/10.)))
                if feature_version>=5:
                    # Equal defender poses can have different interception
                    # credit when target velocity or remaining time differs.
                    # Expose those public variables directly, rather than
                    # requiring the critic to invert a nominal motion rollout.
                    heading=motion[:,left,2];c,s=heading.cos(),heading.sin()
                    def world(vector):
                        return torch.stack((c*vector[:,0]-s*vector[:,1],s*vector[:,0]+c*vector[:,1]),-1)
                    target=motion[:,left,:2]+world(packet['node'][:,left,2:4]*30.)
                    axis=target-.5*(motion[:,left,:2]+motion[:,right,:2])
                    axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1.)
                    velocity=world(packet['node'][:,left,4:6])
                    target_velocity=torch.stack(((velocity*axis).sum(-1),
                        axis[:,0]*velocity[:,1]-axis[:,1]*velocity[:,0]),-1)
                    elapsed=packet['task'][:,-1:] if 'task' in packet else motion.new_zeros((b,1))
                    global_state=torch.cat((target_velocity,1.-elapsed),-1)
                    features.append(global_state[:,None,None].expand(-1,4,4,-1))
                item=torch.cat(features,-1).clamp(-6.,6.)
                directions.append(item if left==i else item.transpose(1,2))
            encoded.append(torch.stack(directions,-2))
            pair_index+=1
    return torch.stack(encoded,1)


def project_intervention_returns(records, difference_weight=4.):
    """Make noisy paired contrasts integrate to one bounded joint potential.

    Each public context has its own intervention graph. A joint program can
    occur in several edges with different sampled futures; fitting those
    returns independently gives contradictory supervision for the same Q.
    The incidence solve retains paired differences and factual endpoint
    anchors, while enforcing cycle consistency before sharing across states.
    """
    import numpy as np
    from scipy.optimize import minimize
    projected=[];residual=0.;edges=0
    for rows in records:
        result=list(rows)
        for mode in sorted({row[4] for row in rows}):
            selected=[i for i,row in enumerate(rows) if row[4]==mode]
            programs={tuple(a) for i in selected for a in rows[i][:2]}
            index={a:i for i,a in enumerate(sorted(programs))};size=len(index)
            h=np.zeros((size,size));rhs=np.zeros(size)
            for i in selected:
                a,b,ya,yb,_=rows[i];u,v=index[tuple(a)],index[tuple(b)]
                h[u,u]+=1.;h[v,v]+=1.;rhs[u]+=ya;rhs[v]+=yb
                if u!=v:
                    w=difference_weight;delta=yb-ya
                    h[u,u]+=w;h[v,v]+=w;h[u,v]-=w;h[v,u]-=w
                    rhs[u]-=w*delta;rhs[v]+=w*delta
            initial=np.clip(np.linalg.solve(h,rhs),0.,1.)
            fit=minimize(lambda q:(.5*q@h@q-rhs@q,h@q-rhs),initial,
                jac=True,method='L-BFGS-B',bounds=[(0.,1.)]*size,
                options=dict(ftol=1e-12,gtol=1e-8,maxiter=200))
            if not fit.success and np.linalg.norm(fit.jac,np.inf)>1e-4:
                raise RuntimeError('Conditional graph projection did not converge: '+fit.message)
            for i in selected:
                a,b,ya,yb,mode=rows[i];qa,qb=fit.x[index[tuple(a)]],fit.x[index[tuple(b)]]
                residual+=float(((qb-qa)-(yb-ya))**2);edges+=1
                result[i]=(a,b,float(qa),float(qb),mode)
        projected.append(result)
    return projected,dict(edges=edges,contrast_projection_mse=residual/max(1,edges))


class SharedResponseValue(nn.Module):
    """Shared joint value trained on the incidence graph of interventions.

    Every factor is a function of a physical candidate response, not a stored
    encounter's winning program. Summing shared factors preserves joint value
    capacity for the no-peer control as well.
    """
    def __init__(self,features=41,width=48,heads=1,aggregation='bottleneck'):
        super().__init__();self.features=features;self.width=width;self.heads=heads;self.aggregation=aggregation
        channels=min(24,max(8,width//4)) if aggregation in ('joint','graph') else heads
        self.factor=nn.Sequential(nn.Linear(features,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,channels))
        self.state=nn.Sequential(nn.Linear(18,width),nn.SiLU(),nn.Linear(width,heads))
        if aggregation in ('joint','graph'):
            self.joint=nn.Sequential(nn.Linear(4*channels,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,heads))
            nn.init.zeros_(self.joint[-1].weight);nn.init.zeros_(self.joint[-1].bias)
            if aggregation=='graph':
                self.graph_node=nn.Sequential(nn.Linear(2*channels,width),nn.SiLU(),nn.Linear(width,channels),nn.SiLU())
                self.graph_message=nn.Sequential(nn.Linear(3*channels,width),nn.SiLU(),nn.Linear(width,channels),nn.SiLU())
                self.graph_update=nn.Sequential(nn.Linear(3*channels,width),nn.SiLU(),nn.Linear(width,channels),nn.SiLU())
        else:
            nn.init.zeros_(self.factor[-1].weight);nn.init.zeros_(self.factor[-1].bias)

    def forward(self,features):
        value=self.factor(features).mean(-2)
        return value.squeeze(-1) if self.heads==1 and self.aggregation not in ('joint','graph') else value

    def joint_actions(self,embeddings,indices):
        """Nonlinear fleet value preserves interactions between pair factors.

        Independent safe-looking pairs can form an unsafe joint continuation.
        Pool the complete candidate-conditioned graph before predicting return,
        instead of forcing return to be a sum or minimum of scalar pair values.
        """
        if self.aggregation=='graph' and indices.shape[1]>256:
            return torch.cat([self.joint_actions(embeddings,chunk) for chunk in indices.split(256,dim=1)],dim=1)
        table=embeddings.flatten(2,3);channels=table.shape[-1]
        chosen=table.gather(2,indices.transpose(1,2)[...,None].expand(-1,-1,-1,channels)).transpose(1,2)
        if self.aggregation=='graph':
            n=int((1+(1+8*chosen.shape[-2])**.5)/2)
            pairs=torch.triu_indices(n,n,1,device=chosen.device)
            incident=pairs.flatten()
            def pool(messages):
                shape=(*messages.shape[:2],n,channels)
                values=torch.cat((messages,messages),dim=-2)
                index=incident[None,None,:,None].expand_as(values)
                mean=messages.new_zeros(shape).scatter_add(2,index,values)/(n-1)
                maximum=messages.new_full(shape,-torch.inf).scatter_reduce(2,index,values,reduce='amax')
                return torch.cat((mean,maximum),-1)
            nodes=self.graph_node(pool(chosen))
            left,right=nodes[:,:,pairs[0]],nodes[:,:,pairs[1]]
            messages=self.graph_message(torch.cat((left+right,(left-right).abs(),chosen),-1))
            nodes=self.graph_update(torch.cat((nodes,pool(messages)),-1))
            return self.joint(torch.cat((nodes.mean(-2),nodes.amax(-2),chosen.mean(-2),chosen.amax(-2)),-1))
        mean=chosen.mean(-2)
        deviation=(chosen.square().mean(-2)-mean.square()).clamp_min(1e-6).sqrt()
        return self.joint(torch.cat((mean,chosen.amax(-2),chosen.amin(-2),deviation),-1))

    def state_value(self,features):
        value=self.state(features[:,:,0,0,:,:18].mean((1,2)))
        return value.squeeze(-1) if self.heads==1 else value


def intervention_programs(action):
    """Observed-style whole programs obtained by changing up to two vessels.

    Each candidate is a complete continuation, retaining every other current
    decision. The candidate set scales quadratically with the fleet instead of
    composing arbitrary unobserved choices across all vessels simultaneously.
    """
    b,n=action.shape[:2];device=action.device
    baseline=(torch.where(action[...,3]>=0.,1.+2.*action[...,3],0.).long()
        if action.shape[-1]>3 else torch.zeros((b,n),dtype=torch.long,device=device))
    candidates=[baseline,torch.arange(4,device=device)[:,None].expand(-1,n),
        (1.+2.*action[...,2]).round().long().clamp(1,3)]
    for row in baseline:
        for i in range(n):
            changed=row[None].expand(4,-1).clone();changed[:,i]=torch.arange(4,device=device)
            candidates.append(changed)
            for j in range(i+1,n):
                changed=row[None].expand(16,-1).clone()
                changed[:,i]=torch.arange(16,device=device)//4
                changed[:,j]=torch.arange(16,device=device)%4
                candidates.append(changed)
    return torch.cat(candidates).unique(dim=0)


def public_return_context(packet,return_order=False,version=1,reference_action=None):
    """Canonical public fleet state for factual continuation-value support."""
    motion=packet['motion'];heading=motion[...,2]
    c,s=heading[:,0].cos(),heading[:,0].sin();relative=packet['node'][:,0,2:4]*30.
    target=motion[:,0,:2]+torch.stack((c*relative[:,0]-s*relative[:,1],s*relative[:,0]+c*relative[:,1]),-1)
    direction=-target/target.norm(dim=-1,keepdim=True).clamp_min(1.)
    def rotate(vector):
        return torch.stack(((vector*direction[:,None]).sum(-1),
            direction[:,None,0]*vector[...,1]-direction[:,None,1]*vector[...,0]),-1)
    relative=rotate(motion[...,:2]-target[:,None])/30.
    body=rotate(torch.stack((heading.cos(),heading.sin()),-1))
    rows=torch.cat((relative,rotate(motion[...,3:5])/3.,body,motion[...,5:6],
        motion[...,:2].norm(dim=-1,keepdim=True)/30.),-1)
    if version>=3:
        if reference_action is None:
            raise ValueError('Control-conditioned values require the current predecessor action.')
        action=reference_action
        first=action[...,3:4] if action.shape[-1]>3 else torch.full_like(action[...,:1],-1.)
        second=action[...,5:6] if action.shape[-1]>5 else first
        # Source commands and the factual fusion are observable before making
        # the intervention. Exposing them avoids asking the value model to
        # reconstruct the frozen source networks and preceding value stages
        # from kinematics alone; no candidate-motion predictor is used here.
        gate=(torch.where(first>=0.,first,action[...,2:3]) if version>=4 else action[...,2:3])
        # Within an active option, a held coefficient replaces the source's
        # provisional gate. Condition on that actual control, consistently
        # in the sampled suffix and in the live midpoint decision.
        controls=torch.cat((action[...,:2],packet['node'][...,11:13],gate,first,second),-1)
        rows=torch.cat((rows,.25*controls),-1)
    order=torch.atan2(relative[...,1],relative[...,0]).argsort(-1)
    context=rows.gather(1,order[...,None].expand_as(rows)).flatten(1)
    if version>=2:
        # Interception credit depends on target motion and remaining time, even
        # when the defenders occupy the same relative configuration. Recover
        # target velocity from the public body-frame measurement, not water or
        # any realized future disturbance.
        velocity=packet['node'][:,0,4:6]*5.
        velocity=torch.stack((c*velocity[:,0]-s*velocity[:,1],
            s*velocity[:,0]+c*velocity[:,1]),-1)
        context=torch.cat((context,rotate(velocity[:,None])[:,0]/5.,
            packet['node'][:,0,15:16]),-1)
    if version>=3:
        held=(first[...,0]>=0.).any(-1)|(second[...,0]>=0.).any(-1)|(second[...,0]==-2.).all(-1)
        duration=(torch.where(held,reference_action[:,0,4]/20.,torch.full_like(context[:,0],.05))
            if reference_action.shape[-1]>4 else torch.full_like(context[:,0],.05))
        context=torch.cat((context,duration[:,None]),-1)
    if version>=5:
        switch=(reference_action[:,0,8] if reference_action.shape[-1]>=9 else
            torch.floor(reference_action[:,0,4]/2.) if reference_action.shape[-1]>4 else
            torch.zeros_like(context[:,0]))
        context=torch.cat((context,(switch/20.)[:,None]),-1)
    return (context,order) if return_order else context


def fleet_set_cost(context,centers):
    """Compare public fleet states without padding or a fixed vessel count.

    A symmetric set distance retrieves candidate encounters. Exact injective
    matching below binds an observed control program to its physical vessels.
    The target velocity and clock remain global conditioning coordinates.
    """
    extra=context.shape[-1]%8
    if extra not in (0,3) or centers.shape[-1]%8!=extra:
        raise ValueError('Incompatible public fleet context versions.')
    n=(context.shape[-1]-extra)//8;m=(centers.shape[-1]-extra)//8
    own=context[:,:8*n].reshape(-1,n,8)
    other=centers[:,:8*m].reshape(-1,m,8)
    costs=(own[:,None,:,None]-other[None,:,None]).square().sum(-1)
    distance=.5*m*(costs.amin(-1).mean(-1)+costs.amin(-2).mean(-1))
    if extra:distance=distance+(context[:,None,-extra:]-centers[None,:,-extra:]).square().sum(-1)
    return distance,costs


def match_fleet_context(context,centers,count=4):
    """Transport whole intervention programs through one-to-one role matches."""
    from scipy.optimize import linear_sum_assignment
    coarse,costs=fleet_set_cost(context,centers)
    shortlist=coarse.topk(min(max(3*count,count),len(centers)),largest=False).indices
    b,_,n,m=costs.shape;extra=context.shape[-1]%8
    distances=context.new_empty(shortlist.shape)
    mappings=torch.full((*shortlist.shape,m),-1,dtype=torch.long,device=context.device)
    for row in range(b):
        for j,index in enumerate(shortlist[row]):
            cost=costs[row,index]
            left,right=linear_sum_assignment(cost.detach().cpu().numpy())
            left=torch.as_tensor(left,device=context.device)
            right=torch.as_tensor(right,device=context.device)
            distance=cost[left,right].sum()*m/min(n,m)
            # Unmatched spectators remain part of the encounter uncertainty.
            if n!=m:distance=distance+.25*abs(n-m)*cost.amin(-1 if n>m else -2).mean()
            if extra:distance=distance+(context[row,-extra:]-centers[index,-extra:]).square().sum()
            distances[row,j]=distance;mappings[row,j,right]=left
    distance,chosen=distances.topk(min(count,shortlist.shape[-1]),largest=False)
    return distance,shortlist.gather(1,chosen),mappings.gather(1,chosen[...,None].expand(-1,-1,m))


class FactualReturnSupport(nn.Module):
    """Trajectory-balanced value bounds for conservative policy improvement.

    A long rollout contributes one neighbor, irrespective of its number of
    states. Outcomes supervise the factual continuation value only; no action
    labels or winning-program lookup are stored here.
    """
    def __init__(self,contexts,trajectories,returns,bandwidth=.5,root_contexts=None,
                 root_successes=None,root_trials=None,root_bandwidth=.03):
        super().__init__()
        self.register_buffer('contexts',torch.as_tensor(contexts,dtype=torch.float32))
        self.register_buffer('trajectories',torch.as_tensor(trajectories,dtype=torch.long))
        self.register_buffer('returns',torch.as_tensor(returns,dtype=torch.float32))
        self.bandwidth=bandwidth
        self.register_buffer('root_contexts',None if root_contexts is None else
            torch.as_tensor(root_contexts,dtype=torch.float32))
        self.register_buffer('root_successes',None if root_successes is None else
            torch.as_tensor(root_successes,dtype=torch.float32))
        self.register_buffer('root_trials',None if root_trials is None else
            torch.as_tensor(root_trials,dtype=torch.float32))
        self.root_bandwidth=root_bandwidth

    def forward(self,context,set_context=False):
        distances=(torch.cat([fleet_set_cost(context,part)[0]
            for part in self.contexts.split(2048)],-1) if set_context else
            (context[:,None]-self.contexts[None]).square().sum(-1))
        per_trajectory=distances.new_full((len(context),len(self.returns)),float('inf'))
        per_trajectory.scatter_reduce_(1,self.trajectories[None].expand(len(context),-1),distances,reduce='amin')
        nearest,index=per_trajectory.topk(min(4,len(self.returns)),largest=False)
        weights=torch.exp(-nearest/(2.*self.bandwidth**2))
        value=(weights*self.returns[index]).sum(-1)/weights.sum(-1).clamp_min(1e-8)
        if self.root_contexts is not None and not set_context:
            distance=(context[:,None]-self.root_contexts[None]).square().sum(-1)
            nearest,roots=distance.topk(min(4,len(self.root_contexts)),largest=False)
            mass=torch.exp(-nearest/(2.*self.root_bandwidth**2))
            # Each distinct paired future supplies one factual Bernoulli
            # outcome, irrespective of how many graph edges reuse it. Two
            # trajectory-prior observations stabilize sparse root estimates.
            # Only public context and aggregated outcomes enter deployment.
            successes=(mass*self.root_successes[roots]).sum(-1)
            trials=(mass*self.root_trials[roots]).sum(-1)
            value=(successes+2.*value)/(trials+2.)
        return value,weights[:,0]


class ConditionalEffectSupport(nn.Module):
    """Locally constrain optimistic gains where observed contrasts are flat."""
    def __init__(self,contexts,limits,bandwidth=.3):
        super().__init__()
        self.register_buffer('contexts',torch.as_tensor(contexts,dtype=torch.float32))
        self.register_buffer('limits',torch.as_tensor(limits,dtype=torch.float32))
        self.bandwidth=bandwidth

    def forward(self,context):
        distance=(context[:,None]-self.contexts[None]).square().sum(-1)
        nearest,index=distance.min(-1)
        confidence=torch.exp(-nearest/(2.*self.bandwidth**2))
        return 1.-confidence*(1.-self.limits[index])


class ResponseActionSupport(nn.Module):
    """Trust joint values on the observed intervention graph's action support.

    A shared critic can assign a large value to an unobserved recombination of
    individually familiar gates. Support is therefore measured for the entire
    joint program, with the same canonical vessel order as the public state.
    """
    def __init__(self,contexts,programs,bandwidth=.3,transport=False):
        super().__init__()
        self.register_buffer('contexts',torch.as_tensor(contexts,dtype=torch.float32))
        self.register_buffer('programs',torch.as_tensor(programs,dtype=torch.bool))
        self.bandwidth=bandwidth
        self.transport=transport

    def transported_support(self,context,span):
        """Transfer complete observed programs, including unchanged roles."""
        n=context.shape[-1]//8;m=(self.contexts.shape[-1]-1)//8
        valid=torch.nonzero((self.contexts[:,-1]-span/10.).abs()<1e-5).flatten()
        empty=(torch.empty(0,dtype=torch.long,device=context.device),context.new_empty(0))
        if not len(valid):return [empty for _ in context]
        distances,indices,mapping=match_fleet_context(context,self.contexts[valid,:-1])
        indices=valid[indices];weights=torch.exp(-distances/(2.*self.bandwidth**2))
        identifiers=torch.arange(self.programs.shape[-1],device=context.device)
        stored=(identifiers[:,None]//(4**torch.arange(m,device=context.device)))%4
        powers=4**torch.arange(n,device=context.device);result=[]
        for row in range(len(context)):
            codes=[];mass=[]
            for j,source in enumerate(indices[row]):
                if weights[row,j]<1e-6:continue
                roles=mapping[row,j];missing=roles<0;available=self.programs[source].clone()
                if missing.any():available&=(stored[:,missing]==0).all(-1)
                choices=stored[available]
                assigned=torch.zeros((len(choices),n),dtype=torch.long,device=context.device)
                assigned[:,roles[~missing]]=choices[:,~missing]
                codes.append((assigned*powers).sum(-1));mass.append(weights[row,j].expand(len(choices)))
            if not codes:result.append(empty);continue
            unique,inverse=torch.cat(codes).unique(return_inverse=True)
            total=context.new_zeros(len(unique)).scatter_add_(0,inverse,torch.cat(mass)).clamp_max(1.)
            result.append((unique,total))
        return result

    def candidate_programs(self,packet,span):
        """Propose supported joint choices without attaching outcome labels."""
        if not self.transport:return None
        context,order=public_return_context(packet,True);n=order.shape[-1]
        programs=[];powers=4**torch.arange(n,device=context.device)
        for row,(codes,mass) in enumerate(self.transported_support(context,span)):
            if len(codes):programs.append(((codes[:,None]//powers)%4)[:,order[row].argsort()])
        return torch.cat(programs).unique(dim=0) if programs else order.new_empty((0,n))

    def support_upper_bound(self,context,span):
        return torch.stack([mass.max() if len(mass) else context.new_zeros(())
            for _,mass in self.transported_support(context,span)])

    def forward(self,packet,options,span):
        context,order=public_return_context(packet,True)
        if self.transport:
            n=options.shape[-1];powers=4**torch.arange(n,device=context.device)
            result=context.new_zeros((len(context),len(options)))
            for row,(codes,mass) in enumerate(self.transported_support(context,span)):
                available=context.new_zeros(4**n).scatter_(0,codes,mass)
                query=(options[:,order[row]]*powers).sum(-1)
                result[row]=available[query]
            return result
        query=torch.cat((context,context.new_full((len(context),1),span/10.)),-1)
        distances=(query[:,None]-self.contexts[None]).square().sum(-1)
        distances=distances.masked_fill((query[:,-1,None]-self.contexts[None,:,-1]).abs()>1e-5,float('inf'))
        nearest,index=distances.min(-1)
        canonical=options[None].expand(len(context),-1,-1).gather(2,order[:,None].expand(-1,len(options),-1))
        codes=(canonical*(4**torch.arange(options.shape[-1],device=options.device))).sum(-1)
        available=self.programs[index].gather(1,codes)
        confidence=torch.exp(-nearest/(2.*self.bandwidth**2))
        return confidence[:,None]*available


def temporal_program_options(blending_levels=(0.,.5,1.)):
    """Retain constant mixtures and reversal schedules in one plan alphabet."""
    import numpy as np
    levels=np.asarray(blending_levels,dtype=float)
    if len(levels)<2 or (np.diff(levels)<=0.).any() or levels[0]!=0. or levels[-1]!=1.:
        raise ValueError('Temporal blending levels must increase from zero to one.')
    return np.asarray([[-1.,-1.],*[[level,level] for level in levels],[0.,1.],[1.,0.]])


def temporal_program_graph(reference,subjects,span,contexts,rng,seed_actions=(),blending_levels=(0.,.5,1.)):
    """Local interventions on complete, vessel-specific two-phase programs."""
    import numpy as np
    options=temporal_program_options(blending_levels)
    n=len(reference);arms=[];known={};edges=[]
    factual=np.stack((reference[:,3],reference[:,5]),-1).copy()
    factual[reference[:,3]<0.]=-1.
    matches=(np.abs(factual[:,None]-options[None]).max(-1)<1e-6)
    if not matches.any(-1).all():raise ValueError('The predecessor program must belong to the temporal intervention alphabet.')
    baseline=matches.argmax(-1)
    def add(choices):
        key=tuple(choices)
        if key not in known:
            action=reference.copy();action[:,3]=options[choices,0];action[:,4]=span;action[:,5]=options[choices,1]
            known[key]=len(arms);arms.append(action)
        return known[key]
    reference_index=add(baseline)
    pair={}
    for a in range(len(options)):
        for b in range(len(options)):
            choices=baseline.copy();choices[subjects[0]]=a;choices[subjects[1]]=b
            pair[a,b]=add(choices)
    for fixed in range(len(options)):
        for a in range(len(options)):
            for b in range(a+1,len(options)):
                edges.extend(((pair[a,fixed],pair[b,fixed]),(pair[fixed,a],pair[fixed,b])))
    for action in seed_actions:
        if action.shape[-1]<6:continue
        phases=action[:,[3,5]].copy();phases[action[:,3]<0.]=-1.
        matches=np.abs(phases[:,None]-options[None]).max(-1)<1e-6
        if not matches.any(-1).all():continue
        seed=matches.argmax(-1);left=add(seed)
        for subject in range(n):
            for choice in range(len(options)):
                if choice==seed[subject]:continue
                alternate=seed.copy();alternate[subject]=choice
                edges.append((left,add(alternate)))
    probability=np.asarray([.1,*[.2 if level in (0.,1.) else .1 for level in blending_levels],.2,.2])
    probability=probability/probability.sum()
    for iteration in range(contexts):
        if iteration<len(options)-1:choices=np.full(n,iteration+1,dtype=int)
        else:choices=rng.choice(len(options),size=n,p=probability)
        changed=np.flatnonzero(choices!=baseline)
        if not len(changed):continue
        left=add(choices)
        # Retain a local comparison for each component of a complete program.
        # These deletion arms retain every other vessel's two-phase decision.
        for subject in changed:
            deletion=choices.copy();deletion[subject]=baseline[subject]
            edges.append((left,add(deletion)))
    edges=list(dict.fromkeys(tuple(sorted(edge)) for edge in edges if edge[0]!=edge[1]))
    return arms,edges,reference_index


def prefix_intervention_graph(reference,subjects,contexts,rng,seed_actions=(),whole_remainder=False):
    """Change an option's first phase while preserving its planned suffix."""
    import numpy as np
    options=np.asarray([-2.,-1.,0.,.5,1.]);n=len(reference)
    arms=[];known={};edges=[]
    def add(choices):
        first=np.where(choices==0,reference[:,3],options[choices])
        key=tuple(first)
        if key not in known:
            action=reference.copy();action[:,3]=first
            if whole_remainder:action[:,5]=np.where(choices==0,reference[:,5],options[choices])
            known[key]=len(arms);arms.append(action)
        return known[key]
    baseline=np.zeros(n,dtype=int);reference_index=add(baseline);pair={}
    for a in range(5):
        for b in range(5):
            choices=baseline.copy();choices[subjects[0]]=a;choices[subjects[1]]=b
            pair[a,b]=add(choices)
    for held in range(5):
        for a in range(5):
            for b in range(a+1,5):
                edges.extend(((pair[a,held],pair[b,held]),(pair[held,a],pair[held,b])))
    seeds=[baseline]
    for action in seed_actions:
        if action.shape[-1]<6:continue
        first=action[:,3]
        seeds.append(np.where(np.isclose(first,reference[:,3]),0,np.where(first<0.,1,2+2*first)).astype(int))
    for iteration in range(contexts):
        seeds.append(np.full(n,iteration+1,dtype=int) if iteration<4 else rng.integers(0,5,size=n))
    for seed in seeds:
        left=add(seed)
        for subject in range(n):
            for choice in range(5):
                alternate=seed.copy();alternate[subject]=choice
                edges.append((left,add(alternate)))
    edges=list(dict.fromkeys(tuple(sorted(edge)) for edge in edges if edge[0]!=edge[1]))
    return arms,edges,reference_index


@lru_cache(maxsize=2)
def program_factor_basis(n,radix):
    """A shared value basis over complete, canonically ordered fleet plans."""
    import numpy as np
    codes=np.arange(radix**n)
    digits=(codes[:,None]//(radix**np.arange(n)))%radix
    unary=(digits[:,:,None]==np.arange(1,radix)[None,None]).astype(np.float32)
    pairs=[(i,j) for i in range(n) for j in range(i+1,n)]
    blocks=[np.ones((len(codes),1),np.float32),unary.reshape(len(codes),-1)]
    blocks.extend((unary[:,i,:,None]*unary[:,j,None,:]).reshape(len(codes),-1)
        for i,j in pairs)
    return digits,np.concatenate(blocks,axis=1),pairs


def factor_program_support(programs,n,radix,pairwise=True):
    """Admit compositions only when their component assignments were sampled."""
    import numpy as np
    digits,_,pairs=program_factor_basis(n,radix)
    observed=digits[np.asarray(programs,dtype=int)]
    valid=np.ones(len(digits),dtype=bool)
    if not len(observed):return np.zeros_like(valid)
    for i in range(n):valid&=np.isin(digits[:,i],observed[:,i])
    if pairwise:
        for i,j in pairs:
            valid&=np.isin(radix*digits[:,i]+digits[:,j],radix*observed[:,i]+observed[:,j])
    return valid


def fit_factor_program_values(observations,contrasts,null,n,radix,difference_weight=4.,ridge=.25):
    """Identify whole-plan values through shared vessel and pair coefficients.

    Node outcomes anchor task value; single-vessel contrasts constrain changes
    in the SAME coefficients. The ordinary-return control uses the identical
    basis, regularization and compositional support with zero contrast weight.
    """
    import numpy as np
    programs=np.asarray(sorted(observations),dtype=int)
    _,basis,_=program_factor_basis(n,radix)
    design=basis[programs].astype(np.float64)
    targets=np.asarray([np.mean(observations[int(q)]) for q in programs])
    penalty=np.full(design.shape[1],ridge,dtype=float)
    penalty[1+n*(radix-1):]*=4.
    normal=design.T@design+np.diag(penalty)
    rhs=design.T@targets;rhs[0]+=penalty[0]*null
    if difference_weight and contrasts:
        edges=list(contrasts)
        delta=np.asarray([np.mean(contrasts[edge]) for edge in edges])
        weight=difference_weight*(1.+4.*np.abs(delta))
        difference=basis[[b for a,b in edges]].astype(np.float64)-basis[[a for a,b in edges]]
        normal+=difference.T@(weight[:,None]*difference)
        rhs+=difference.T@(weight*delta)
    coefficients=np.linalg.solve(normal,rhs)
    return basis@coefficients,factor_program_support(programs,n,radix)


def fit_unary_program_values(observations,contrasts,null,n,radix,difference_weight=4.,ridge=.25):
    """Fit local decision effects without conditioning on teammates' choices.

    Public context, complete candidate plans and paired outcomes are unchanged.
    The value is restricted to an intercept plus one function of each vessel's
    own decision. Consequently a local intervention has the same estimated
    effect at every teammate assignment. Only actually sampled plans execute.
    """
    import numpy as np
    programs=np.asarray(sorted(observations),dtype=int)
    codes=np.arange(radix**n)
    digits=(codes[:,None]//(radix**np.arange(n)))%radix
    basis=np.concatenate((np.ones((len(codes),1)),
        *(np.eye(radix)[digits[:,i],1:] for i in range(n))),axis=1)
    design=basis[programs]
    target=np.asarray([np.mean(observations[int(q)]) for q in programs])
    normal=design.T@design+ridge*np.eye(basis.shape[1])
    rhs=design.T@target;rhs[0]+=ridge*null
    if difference_weight and contrasts:
        edges=list(contrasts)
        delta=np.asarray([np.mean(contrasts[edge]) for edge in edges])
        weight=difference_weight*(1.+4.*np.abs(delta))
        difference=basis[[b for a,b in edges]]-basis[[a for a,b in edges]]
        normal+=difference.T@(weight[:,None]*difference)
        rhs+=difference.T@(weight*delta)
    support=np.zeros(len(codes),dtype=bool);support[programs]=True
    return basis@np.linalg.solve(normal,rhs),support


def fit_kernel_program_values(observations,contrasts,null,n,radix,difference_weight=4.,ridge=.25):
    """Share complete-plan values without truncating higher-order interactions.

    The kernel acts on complete fusion assignments. Local intervention edges
    constrain the same function through differences of kernel evaluations.
    Both estimators may interpolate sampled components one vessel away from a
    measured complete plan; only the conditional estimator uses paired edges.
    """
    import numpy as np
    programs=np.asarray(sorted(observations),dtype=int)
    digits,_,_=program_factor_basis(n,radix)
    distance=(digits[:,None]!=digits[programs][None]).sum(-1)
    cross=np.exp(-distance.astype(float))
    kernel=cross[programs]+1e-6*np.eye(len(programs))
    index={int(q):i for i,q in enumerate(programs)}
    normal=np.eye(len(programs))
    target=np.asarray([np.mean(observations[int(q)])-null for q in programs])
    if difference_weight:
        for (a,b),samples in contrasts.items():
            delta=float(np.mean(samples));weight=difference_weight*(1.+4.*abs(delta))
            i,j=index[a],index[b]
            normal[i,i]+=weight;normal[j,j]+=weight;normal[i,j]-=weight;normal[j,i]-=weight
            target[i]-=weight*delta;target[j]+=weight*delta
    coefficients=np.linalg.solve(normal@kernel+ridge*np.eye(len(programs)),target)
    support=factor_program_support(programs,n,radix)&(distance.min(-1)<=1)
    return null+cross@coefficients,support


def fit_paired_nuisance_values(observations,contrasts,factual,difference_weight=4.,nuisance_ridge=.25):
    """Estimate program values and shared continuation effects jointly.

    A continuation identifier is only a training block. Its nuisance offset
    absorbs correlated outcome variation within that block; deployment retains
    only the complete-program potentials. Single-vessel contrasts constrain
    these same potentials without the shared offset. The zero-mean nuisance
    constraint anchors them to marginal task success rather than to a selected
    favorable continuation.
    """
    import numpy as np
    from scipy.sparse import coo_matrix,diags,bmat,csc_matrix
    from scipy.sparse.linalg import spsolve
    programs=sorted(observations);index={q:i for i,q in enumerate(programs)}
    futures={f for samples in observations.values() for f in samples}
    future_index={f:i for i,f in enumerate(sorted(futures,key=repr))}
    k=len(programs);m=len(futures);rows=[];cols=[];weights=[];targets=[]
    block_precision=np.zeros(m)
    for q in programs:
        samples=observations[q];weight=1./len(samples)
        for f,values in samples.items():
            row=len(targets);targets.append(float(np.mean(values)))
            rows.extend((row,row));cols.extend((index[q],k+future_index[f]))
            weights.extend((np.sqrt(weight),np.sqrt(weight)))
            targets[-1]*=np.sqrt(weight)
            block_precision[future_index[f]]+=weight
    for (a,b),samples in contrasts.items():
        if not difference_weight:break
        outcomes=np.asarray([np.mean(values) for values in samples.values()])
        # A shared disturbance can alter both outcomes while preserving their
        # difference. Retain that precision instead of weighting large effects
        # more heavily merely because they were observed to be favorable.
        precision=difference_weight/(.25+float(outcomes.var()))
        row=len(targets);targets.append(np.sqrt(precision)*float(outcomes.mean()))
        rows.extend((row,row));cols.extend((index[a],index[b]))
        weights.extend((-np.sqrt(precision),np.sqrt(precision)))
    design=coo_matrix((weights,(rows,cols)),shape=(len(targets),k+m)).tocsr()
    # Node losses carry unit total weight per program. A sparse paired future
    # therefore has much less weight than a future shared by the whole library.
    # Shrink its common disturbance relative to that actual information mass;
    # a fixed penalty would suppress precisely the sparse paired comparisons.
    prior=np.r_[np.full(k,.25),nuisance_ridge*block_precision]
    normal=design.T@design+diags(prior)
    target=design.T@np.asarray(targets)+np.r_[np.full(k,.25*factual),np.zeros(m)]
    constraint=csc_matrix(np.r_[np.zeros(k),np.full(m,1./m)][:,None])
    system=bmat([[normal,constraint],[constraint.T,None]],format='csc')
    solution=spsolve(system,np.r_[target,0.])
    return solution[:k].clip(0.,1.)


def fit_calibrated_program_values(observations,contrasts,factual_outcomes,factual,difference_weight=4.):
    """Calibrate local program comparisons with their shared factual outcomes.

    Different programs need not respond equally to a difficult continuation.
    A cross-fitted control variate therefore estimates a separate response for
    each complete program and each local contrast. Only the resulting marginal
    potentials, rather than the realized continuation outcomes, are deployed.
    """
    import numpy as np
    programs=sorted(observations);index={q:i for i,q in enumerate(programs)}
    futures=sorted(factual_outcomes,key=repr)
    fold={future:i%2 for i,future in enumerate(futures)}
    reference=float(np.mean(list(factual_outcomes.values()))) if futures else factual
    def estimate(samples):
        keys=list(samples)
        y=np.asarray([np.mean(samples[key]) for key in keys],dtype=float)
        x=np.asarray([factual_outcomes.get(key,reference) for key in keys],dtype=float)
        split=np.asarray([fold.get(key,0) for key in keys])
        adjusted=y.copy()
        for part in (0,1):
            train=split!=part;held=~train
            if train.sum()<2 or not held.any():continue
            centered=x[train]-x[train].mean()
            covariance=np.mean(centered*(y[train]-y[train].mean()))
            slope=np.clip(covariance/(np.mean(centered**2)+.1/train.sum()),-1.,1.)
            adjusted[held]+=slope*(reference-x[held])
        return float(adjusted.mean()),float(adjusted.var())
    node=np.asarray([estimate(observations[q])[0] for q in programs])
    normal=1.25*np.eye(len(programs));target=node+.25*factual
    for (a,b),samples in contrasts.items():
        if not difference_weight:break
        delta,variance=estimate(samples)
        weight=difference_weight/(.25+variance)
        i,j=index[a],index[b]
        normal[i,i]+=weight;normal[j,j]+=weight
        normal[i,j]-=weight;normal[j,i]-=weight
        target[i]-=weight*delta;target[j]+=weight*delta
    return np.linalg.solve(normal,target).clip(0.,1.)


def fit_bounded_graph_values(matrix,target,lower=0.,upper=1.):
    """Fit node outcomes and local contrasts within the task-value range.

    Clipping an unconstrained graph solution changes its edge differences
    without re-optimizing neighboring potentials. The box-constrained solve
    keeps those relationships in the estimation objective itself.
    """
    import numpy as np
    from scipy.optimize import minimize
    unconstrained=np.linalg.solve(matrix,target)
    initial=unconstrained.clip(lower,upper)
    if np.array_equal(initial,unconstrained):return initial
    def objective(value):
        product=matrix@value
        return .5*float(value@product)-float(target@value),product-target
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',
        bounds=[(lower,upper)]*len(initial),
        options=dict(maxiter=1000,ftol=1e-12,gtol=1e-8))
    if not result.success:
        raise RuntimeError(f'Bounded conditional-value optimization failed: {result.message}')
    return result.x


def fit_binomial_program_values(observations,contrasts,factual,difference_weight=4.):
    """Calibrate complete-program success and same-future conditional odds.

    A discordant paired outcome identifies which program succeeded under the
    shared future. Concordant pairs supply absolute outcomes but no conditional
    odds observation. One potential per complete program ties both likelihoods
    together; no independent pair score is used at execution.
    """
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    programs=sorted(observations);index={code:i for i,code in enumerate(programs)}
    count=np.asarray([len(observations[code]) for code in programs],dtype=float)
    success=np.asarray([np.sum(observations[code]) for code in programs],dtype=float)
    prior=float(np.clip(factual,.01,.99))
    count=count+.25;success=success+.25*prior
    edge=[]
    if difference_weight:
        for (a,b),samples in contrasts.items():
            samples=np.asarray(samples)
            wins=float((samples>0.).sum());losses=float((samples<0.).sum())
            if wins+losses:edge.append((index[a],index[b],wins,losses))
    if edge:
        a,b,win,loss=np.asarray(edge).T;a=a.astype(int);b=b.astype(int)
        total=win+loss
    def objective(logits):
        value=np.sum(count*np.logaddexp(0.,logits)-success*logits)+.0005*np.square(logits).sum()
        gradient=count*expit(logits)-success+.001*logits
        if edge:
            difference=logits[b]-logits[a]
            value+=difference_weight*np.sum(total*np.logaddexp(0.,difference)-win*difference)
            residual=difference_weight*(total*expit(difference)-win)
            np.add.at(gradient,b,residual);np.add.at(gradient,a,-residual)
        return value,gradient
    mean=np.clip(success/count,.001,.999)
    result=minimize(objective,np.log(mean/(1.-mean)),jac=True,method='L-BFGS-B',
        options=dict(maxiter=160,ftol=1e-10,gtol=1e-6))
    return expit(result.x)


class SharedProgramPotential(nn.Module):
    """A fleet value shared across public contexts and complete programs.

    Program choices enter the vessel messages before pooling. Single-vessel
    comparisons consequently constrain a common nonlinear value function,
    rather than separate unconstrained entries of a local lookup table.
    """
    def __init__(self,width=48,paired=True,state=None,node_width=8,global_width=3,program_options=None,probabilistic=False):
        super().__init__();self.width=width;self.paired=paired
        self.probabilistic=probabilistic
        self.node_width=node_width;self.global_width=global_width
        self.register_buffer('program_options',None if program_options is None else
            torch.as_tensor(program_options,dtype=torch.float32))
        self.radix=4 if program_options is None else len(program_options)
        control_width=(4 if program_options is None else 8) if node_width>=15 else 0
        self.node=nn.Sequential(nn.Linear(node_width+self.radix+control_width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.message=nn.Sequential(nn.Linear(2*width+node_width,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.update=nn.Sequential(nn.Linear(3*width,width),nn.SiLU())
        self.readout=nn.Sequential(nn.Linear(2*width+global_width+1,width),nn.SiLU(),nn.Linear(width,1))
        if state is not None:self.load_state_dict(state)

    def logits(self,context,codes):
        extra=context.shape[-1]%self.node_width;n=(context.shape[-1]-extra)//self.node_width
        rows=context[:,:self.node_width*n].reshape(-1,n,self.node_width)
        programs=(codes[:,None]//(self.radix**torch.arange(n,device=context.device)))%self.radix
        features=[rows,F.one_hot(programs,self.radix).to(rows.dtype)]
        if self.node_width>=15:
            source=4.*rows[...,8:10];rule=4.*rows[...,10:12]
            feedback=4.*rows[...,12];held=4.*rows[...,13]
            if self.program_options is None:
                selected=torch.where(programs>0,(programs.to(rows.dtype)-1.)*.5,feedback)
                factual=torch.where(held>=0.,held,feedback)
                command=selected[...,None]*source+(1.-selected[...,None])*rule
                change=(selected-factual)[...,None]*(source-rule)
            else:
                phases=self.program_options[programs]
                factual=torch.stack((held,4.*rows[...,14]),-1)
                factual=torch.where(factual>=0.,factual,feedback[...,None])
                selected=torch.where(phases==-2.,factual,
                    torch.where(phases>=0.,phases,feedback[...,None]))
                command=(selected[...,None]*source[:,:,None]+(1.-selected[...,None])*rule[:,:,None]).flatten(-2)
                change=((selected-factual)[...,None]*(source-rule)[:,:,None]).flatten(-2)
            features.extend((command,change))
        node=self.node(torch.cat(features,-1))
        difference=rows[:,None]-rows[:,:,None]
        messages=self.message(torch.cat((node[:,:,None].expand(-1,-1,n,-1),
            node[:,None].expand(-1,n,-1,-1),difference),-1))
        mask=~torch.eye(n,dtype=torch.bool,device=context.device)[None,:,:,None]
        average=(messages*mask).sum(2)/max(1,n-1)
        maximum=messages.masked_fill(~mask,-1e6).amax(2)
        encoded=self.update(torch.cat((node,average,maximum),-1))
        global_state=context[:,-extra:] if extra==self.global_width else context.new_zeros((len(context),self.global_width))
        pooled=torch.cat((encoded.mean(1),encoded.amax(1),global_state,
            context.new_full((len(context),1),n/6.)),-1)
        raw=self.readout(pooled).squeeze(-1)
        return raw

    def forward(self,context,codes):
        raw=self.logits(context,codes)
        return raw.sigmoid() if self.probabilistic else raw


def fit_shared_program_potential(examples,width=48,updates=1500,paired=True,conditional=True,
                                 difference_weight=4.,seed=42,node_width=8,global_width=3,
                                 program_options=None,device='cpu',paired_likelihood=False):
    """Fit the same node observations with or without local contrast loss."""
    import numpy as np
    contexts=[];codes=[];targets=[];left=[];right=[];differences=[];precision=[];wins=[];losses=[]
    if paired_likelihood and paired:
        raise ValueError('Paired likelihood estimates absolute success probabilities.')
    for example in examples:
        offset=len(codes);programs=example['programs']
        contexts.extend([example['context']]*len(programs))
        codes.extend(programs);targets.extend(example['targets'])
        for edge in example['edges']:
            a,b,delta=edge[:3]
            left.append(offset+a);right.append(offset+b);differences.append(delta)
            precision.append(edge[3] if len(edge)>3 else 1.+4.*abs(delta))
            if paired_likelihood:
                if len(edge)<6:raise ValueError('Paired likelihood needs discordant paired-outcome counts.')
                wins.append(edge[4]);losses.append(edge[5])
    context=torch.as_tensor(np.asarray(contexts),dtype=torch.float32,device=device)
    code=torch.as_tensor(codes,dtype=torch.long,device=device);target=torch.as_tensor(targets,dtype=torch.float32,device=device)
    left=torch.as_tensor(left,dtype=torch.long,device=device);right=torch.as_tensor(right,dtype=torch.long,device=device)
    delta=torch.as_tensor(differences,dtype=torch.float32,device=device)
    precision=torch.as_tensor(precision,dtype=torch.float32,device=device)
    wins=torch.as_tensor(wins,dtype=torch.float32,device=device)
    losses=torch.as_tensor(losses,dtype=torch.float32,device=device)
    degree=torch.bincount(torch.cat((left,right)),minlength=len(code)).to(torch.float32)
    # The batch mixes uniform nodes and the endpoints of uniform edges.
    # Use all their absolute outcomes, correcting the resulting degree bias.
    node_weight=3./(1.+len(code)*degree/max(1,len(left)))
    torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed+104729)
    net=SharedProgramPotential(width,paired,node_width=node_width,global_width=global_width,
        program_options=program_options,probabilistic=paired_likelihood).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=.005)
    last={}
    with torch.enable_grad():
        for step in range(updates):
            nodes=torch.randint(len(code),(128,),generator=generator).to(device)
            edges=torch.randint(len(left),(128,),generator=generator).to(device)
            selected=torch.cat((nodes,left[edges],right[edges]))
            prediction=net.logits(context[selected],code[selected])
            if paired_likelihood:
                node_loss=(F.binary_cross_entropy_with_logits(prediction,target[selected],reduction='none')
                    *node_weight[selected]).mean()
                log_odds=prediction[256:]-prediction[128:256]
                # Conditioning on discordance removes the common outcome of a
                # paired continuation. Concordant pairs supply node evidence.
                edge_loss=(wins[edges]*F.softplus(-log_odds)+losses[edges]*F.softplus(log_odds)).mean()
            else:
                node_loss=((prediction-target[selected]).square()*node_weight[selected]).mean()
                contrast=(prediction[256:]-prediction[128:256]-delta[edges]).square()
                edge_loss=(contrast*precision[edges]).mean()
            loss=node_loss+(difference_weight*edge_loss if conditional else 0.)
            optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            if (step+1)%250==0 or step+1==updates:
                last=dict(phase='shared_program_value_fit',update=step+1,conditional=conditional,
                    node_loss=float(node_loss.detach()),node_loss_kind='bernoulli' if paired_likelihood else 'squared',
                    contrast_loss=float(edge_loss.detach()))
                if not paired_likelihood:last['node_mse']=last['node_loss']
                print(last,flush=True)
    record=dict(width=width,paired=paired,node_width=node_width,global_width=global_width,
        program_options=program_options,probabilistic=paired_likelihood,
        state={k:v.detach().cpu() for k,v in net.state_dict().items()})
    return net.cpu().eval(),record,dict(nodes=len(code),edges=len(left),updates=updates,seed=seed,last=last)


def fit_intervention_context_metric(examples,conditional=True,difference_weight=4.,
                                    bandwidth=.15,updates=400,seed=42):
    """Learn which public-state differences preserve intervention consequences.

    Each training prediction excludes its own context. Node outcomes identify
    task-value similarity; paired contrasts additionally identify similarity of
    local decision effects. Only a positive diagonal metric is deployed.
    """
    import numpy as np
    contexts=torch.as_tensor(np.asarray([x['context'] for x in examples]),dtype=torch.float32)
    programs=sorted({q for x in examples for q in x['programs']});lookup={q:i for i,q in enumerate(programs)}
    values=torch.zeros((len(examples),len(programs)));known=torch.zeros_like(values,dtype=torch.bool)
    root=[];code=[];target=[];edges=[]
    for i,x in enumerate(examples):
        indices=[lookup[q] for q in x['programs']]
        values[i,indices]=torch.as_tensor(x['targets'],dtype=torch.float32);known[i,indices]=True
        root.extend([i]*len(indices));code.extend(indices);target.extend(x['targets'])
        edges.extend((i,indices[a],indices[b],delta) for a,b,delta,*_ in x['edges'])
    root=torch.tensor(root);code=torch.tensor(code);target=torch.tensor(target,dtype=torch.float32)
    valid=(known.sum(0)[code]>1);root,code,target=root[valid],code[valid],target[valid]
    edges=[e for e in edges if known[:,e[1]].sum()>1 and known[:,e[2]].sum()>1]
    if len(contexts)<2 or not len(root) or not edges:
        return np.ones(contexts.shape[-1],dtype=np.float32),dict(updates=0)
    er,ea,eb,ed=zip(*edges);er=torch.tensor(er);ea=torch.tensor(ea);eb=torch.tensor(eb);ed=torch.tensor(ed)
    squared=(contexts[:,None]-contexts[None]).square()
    active=contexts.var(0,unbiased=False)>1e-8
    parameter=nn.Parameter(torch.zeros(contexts.shape[-1]))
    optimizer=torch.optim.Adam([parameter],lr=.025)
    generator=torch.Generator().manual_seed(seed+2750159)
    last={}
    with torch.enable_grad():
        for step in range(updates):
            ni=torch.randint(len(root),(128,),generator=generator)
            ei=torch.randint(len(er),(128,),generator=generator)
            rr=torch.cat((root[ni],er[ei],er[ei]));cc=torch.cat((code[ni],ea[ei],eb[ei]))
            precision=parameter.clamp(-4.,4.).exp()
            logits=-(squared[rr]*precision).sum(-1)/(2.*bandwidth**2)
            eligible=known[:,cc].T.clone();eligible[torch.arange(len(rr)),rr]=False
            probability=logits.masked_fill(~eligible,-torch.inf).softmax(-1)
            prediction=(probability*values[:,cc].T).sum(-1)
            node_loss=(prediction[:128]-target[ni]).square().mean()
            edge_loss=(prediction[256:]-prediction[128:256]-ed[ei]).square().mean()
            penalty=.002*parameter[active].square().mean()
            loss=node_loss+(difference_weight*edge_loss if conditional else 0.)+penalty
            optimizer.zero_grad();loss.backward();optimizer.step()
            with torch.no_grad():parameter[~active]=0.
            if step+1==updates:
                last=dict(node_mse=float(node_loss.detach()),contrast_mse=float(edge_loss.detach()))
    return parameter.detach().clamp(-4.,4.).exp().numpy(),dict(updates=updates,contexts=len(contexts),**last)


class JointInterventionOptions(nn.Module):
    """A frozen predecessor that selects among several learned windows."""
    def __init__(self,records):
        super().__init__()
        common=records[0].get('prior') if records else None
        self.shared_prior=(common is not None and len(records)>1
            and all(record.get('prior') is common for record in records))
        self.values=nn.ModuleList()
        for record in records:
            shared=self.shared_prior and len(self.values)>0
            value=JointInterventionValue(**(dict(record,prior=None) if shared else record))
            if shared:value.prior=self.values[0].prior
            self.values.append(value)

    def forward(self,packet,action):
        if not len(self.values):return action
        if self.shared_prior:action=self.values[0].prior(packet,action)
        alternatives=[value(packet,action,return_gain=True,apply_prior=not self.shared_prior) for value in self.values]
        gains=torch.stack([gain.masked_fill(gain<=value.margin,float('-inf'))
            for value,(_,gain) in zip(self.values,alternatives)],1)
        choice=gains.argmax(-1)
        return torch.stack([candidate for candidate,_ in alternatives],1)[torch.arange(len(action),device=action.device),choice]


class JointInterventionValue(nn.Module):
    """Local joint potentials reconstructed from single-vessel graph edges.

    Whole observed programs retain higher-order coordination that a sum of
    pair factors cannot represent. Kernel transfer uses canonical public
    motion only; absent programs carry no improvement authority.
    """
    def __init__(self,contexts,values,observed,span,anchor,bandwidth=.15,margin=.15,
                 conditional_strata=False,prior=None,paired_advantage=False,context_version=1,
                 wait_values=None,transport=False,reversal=False,effect_locality=False,
                 gain_variance=None,confidence=0.,support_uncertainty=False,initial_only=False,
                 decision_tolerance=0.,blending_penalty=0.,fixed_intervention_clock=False,
                 effect_values=None,effect_margin=0.,program_model=None,program_residuals=None,program_options=None,
                 prefix_refinement=False,context_precision=None,learning_objective=None,
                 measured=None,counterfactual_credit=False,program_offsets=None):
        super().__init__()
        self.register_buffer('contexts',torch.as_tensor(contexts,dtype=torch.float32))
        self.register_buffer('context_precision',torch.ones(self.contexts.shape[-1]) if context_precision is None
            else torch.as_tensor(context_precision,dtype=torch.float32))
        self.register_buffer('values',torch.as_tensor(values,dtype=torch.float32))
        self.register_buffer('observed',torch.as_tensor(observed,dtype=torch.bool))
        self.register_buffer('measured',torch.as_tensor(observed if measured is None else measured,dtype=torch.bool))
        self.counterfactual_credit=counterfactual_credit
        if counterfactual_credit and (conditional_strata or paired_advantage or transport or effect_locality):
            raise ValueError('Critic-based counterfactual credit uses ordinary complete-program success values.')
        self.register_buffer('supported_program_codes',torch.nonzero(self.observed.any(0)).flatten())
        self.register_buffer('wait_values',torch.zeros(len(contexts)) if wait_values is None
            else torch.as_tensor(wait_values,dtype=torch.float32))
        self.anchor=FactualReturnSupport(**anchor)
        self.span=span;self.bandwidth=bandwidth;self.margin=margin
        self.conditional_strata=conditional_strata
        self.paired_advantage=paired_advantage
        self.context_version=context_version
        if context_version>=3 and (transport or effect_locality):
            raise ValueError('Control-conditioned values require matching all predecessor-control coordinates.')
        self.program_model=None if program_model is None else SharedProgramPotential(**program_model).eval()
        self.register_buffer('program_options',None if program_options is None else
            torch.as_tensor(program_options,dtype=torch.float32))
        self.program_radix=4 if program_options is None else len(program_options)
        self.prefix_refinement=prefix_refinement
        if prefix_refinement and (program_options is None or decision_tolerance or blending_penalty):
            raise ValueError('Prefix refinements use relative whole-plan options and their paired task values.')
        if program_options is not None and (transport or effect_locality):
            raise ValueError('Temporal program values use directly matched complete-fleet contexts.')
        if self.program_model is not None:
            learned=self.program_model.program_options
            if self.program_model.radix!=self.program_radix or ((learned is None)!=(self.program_options is None)):
                raise ValueError('The learned value and executed plans must use the same program alphabet.')
            if learned is not None and not torch.equal(learned,self.program_options):
                raise ValueError('The learned value and executed plans must preserve both program phases.')
        self.register_buffer('program_residuals',None if program_residuals is None else
            torch.as_tensor(program_residuals,dtype=torch.float32))
        self.register_buffer('program_offsets',None if program_offsets is None else
            torch.as_tensor(program_offsets,dtype=torch.float32))
        if program_residuals is not None and program_model is None:
            raise ValueError('Local intervention residuals require their shared program-value model.')
        if program_model is not None and (transport or effect_locality or conditional_strata or confidence):
            raise ValueError('Shared program potentials currently use direct, unstratified context support.')
        self.transport=transport
        self.reversal=reversal
        self.effect_locality=effect_locality
        self.confidence=confidence
        self.support_uncertainty=support_uncertainty
        self.initial_only=initial_only
        self.decision_tolerance=decision_tolerance
        self.blending_penalty=blending_penalty
        self.fixed_intervention_clock=fixed_intervention_clock
        self.effect_margin=effect_margin
        self.register_buffer('effect_values',None if effect_values is None else
            torch.as_tensor(effect_values,dtype=torch.float32))
        if effect_values is not None and (transport or effect_locality or conditional_strata):
            raise ValueError('The paired-effect constraint requires the matching whole-program context estimator.')
        self.register_buffer('gain_variance',torch.zeros_like(self.values) if gain_variance is None
            else torch.as_tensor(gain_variance,dtype=torch.float32))
        if confidence and (not paired_advantage or conditional_strata):
            raise ValueError('Confidence bounds require unstratified paired improvement values.')
        if support_uncertainty and (not confidence or not paired_advantage):
            raise ValueError('Support uncertainty requires paired posterior improvement values.')
        self.prior=(JointInterventionOptions(prior) if isinstance(prior,list) else
            JointInterventionValue(**prior) if prior is not None else None)
        if effect_locality:
            m=(self.contexts.shape[-1]-self.contexts.shape[-1]%8)//8
            codes=torch.arange(self.values.shape[1]);programs=(codes[:,None]//(4**torch.arange(m)))%4
            centers=[];weights=[];masks=[];indices=[];offsets=[0]
            for center in range(len(self.contexts)):
                available=torch.nonzero(self.observed[center]).squeeze(-1)
                changed=programs[available]>0
                if not len(available):continue
                kinds,membership=changed.unique(dim=0,return_inverse=True)
                position=self.contexts[center,:8*m].reshape(m,8)[:,:2]*30.
                separation=(position[:,None]-position[None]).norm(dim=-1)
                for group,mask in enumerate(kinds):
                    influence=(torch.exp(-separation[:,mask].amin(-1).square()/200.).clamp_min(.02)
                        if mask.any() else torch.ones(m,device=position.device))
                    selected=available[membership==group]
                    centers.append(center);weights.append(influence);masks.append(mask);indices.append(selected)
                    offsets.append(offsets[-1]+len(selected))
            self.register_buffer('effect_centers',torch.tensor(centers,dtype=torch.long))
            self.register_buffer('effect_weights',torch.stack(weights) if weights else self.contexts.new_empty((0,m)))
            self.register_buffer('effect_masks',torch.stack(masks) if masks else torch.empty((0,m),dtype=torch.bool))
            self.register_buffer('effect_programs',torch.cat(indices) if indices else torch.empty(0,dtype=torch.long))
            self.register_buffer('effect_offsets',torch.tensor(offsets,dtype=torch.long))

    def supported_gain(self,mean,variance,mass):
        """Mix local credit with its zero-mean prior outside measured support.

        The symmetric rescue/neutral/harm prior has mean uncertainty 1/6.
        Weak retrieval must increase uncertainty rather than merely shrink a
        positive lower estimate toward zero. This permits a zero improvement
        threshold without granting authority to unsupported interventions.
        """
        support=mass.clamp(0.,1.)
        mixed_variance=support*(variance+(1.-support)*mean.square())+(1.-support)/6.
        return support*mean-self.confidence*mixed_variance.clamp_min(0.).sqrt()

    def forward(self,packet,action,return_gain=False,apply_prior=True):
        if not self.initial_only:return self.select_action(packet,action,return_gain,apply_prior)
        # Restrict only this update to its sampled decision boundary. The
        # frozen predecessor remains active at every later decision.
        if apply_prior and self.prior is not None:action=self.prior(packet,action)
        initial=packet['node'][:,0,15].abs()<1e-7
        if not initial.any():
            return (action,action.new_full((len(action),),-1.)) if return_gain else action
        result=self.select_action(packet,action,return_gain,False)
        if return_gain:result,gain=result
        if action.shape[-1]!=result.shape[-1]:
            free=torch.full_like(action[...,:1],-1.)
            action=torch.cat((action[...,:3],free,torch.full_like(free,self.span),free),-1)
        result=torch.where(initial[:,None,None],result,action)
        return (result,torch.where(initial,gain,torch.full_like(gain,-1.))) if return_gain else result

    def select_action(self,packet,action,return_gain=False,apply_prior=True):
        if apply_prior and self.prior is not None:action=self.prior(packet,action)
        if not self.supported_program_codes.numel():
            # An empty support set is an identity update, including when this
            # record is an ancestor of a later, useful policy update.
            if action.shape[-1]==3:
                free=torch.full_like(action[...,:1],-1.)
                action=torch.cat((action,free,torch.full_like(free,self.span),free),-1)
            return (action,action.new_full((len(action),),-1.)) if return_gain else action
        if self.prefix_refinement:
            if action.shape[-1]<6:
                return (action,action.new_full((len(action),),-1.)) if return_gain else action
            eligible=(action[:,0,4]==self.span)&(action[...,[3,5]]>=0.).any(-1).any(-1)
            if not bool(eligible.all()):
                result=action.clone();gain=action.new_full((len(action),),-1.)
                if bool(eligible.any()):
                    subset={key:value[eligible] for key,value in packet.items()}
                    result[eligible],gain[eligible]=self.select_action(subset,action[eligible],True,False)
                return (result,gain) if return_gain else result
        context,order=public_return_context(packet,True,self.context_version,reference_action=action)
        if self.effect_locality:
            return self.local_effect_action(context,order,action,return_gain)
        if self.transport:
            return self.transported_action(context,order,action,return_gain)
        b,n=action.shape[:2]
        distance=((context[:,None]-self.contexts[None]).square()*self.context_precision).sum(-1)
        nearest,index=distance.topk(min(4,len(self.contexts)),largest=False)
        weights=torch.exp(-nearest/(2.*self.bandwidth**2))
        # Each program's support mass is bounded by the total context mass.
        # Below the existing admissibility threshold every program receives
        # gain -1, so no complete-plan expansion is needed for these states.
        # Apply the same bound row-wise when only part of a batch has support.
        supported=weights.sum(-1)>=1e-6
        if not bool(supported.all()) and not self.blending_penalty and not self.decision_tolerance:
            result=action.clone()
            if result.shape[-1]==3:
                free=torch.full_like(result[...,:1],-1.)
                result=torch.cat((result,free,torch.full_like(free,self.span),free),-1)
            gain=action.new_full((len(action),),-1.)
            if bool(supported.any()):
                subset={key:value[supported] for key,value in packet.items()}
                chosen,score=self.select_action(subset,action[supported],True,False)
                result[supported]=chosen;gain[supported]=score
            return (result,gain) if return_gain else result
        # Success values are bounded by one. Outside the conditional support,
        # even that upper bound cannot clear the intervention margin.
        if not return_gain and self.margin>=0. and (weights.sum(-1)<=self.margin).all():
            if action.shape[-1]==3:
                column=torch.full_like(action[...,:1],-1.)
                action=torch.cat((action,column,torch.full_like(column,self.span),column),-1)
            return action
        baseline=(torch.where(action[...,3]>=0.,1.+2.*action[...,3],0.).long()
            if action.shape[-1]>3 else torch.zeros((b,n),dtype=torch.long,device=action.device))
        if self.program_options is not None and action.shape[-1]>5:
            pair=action[...,[3,5]]
            pair=torch.where((action[...,3]<0.)[...,None],torch.full_like(pair,-1.),pair)
            baseline=(pair[:,:,None]-self.program_options[None,None]).square().sum(-1).argmin(-1)
        if self.prefix_refinement:baseline=torch.zeros_like(baseline)
        canonical=baseline.gather(1,order)
        powers=self.program_radix**torch.arange(n,device=action.device)
        if self.decision_tolerance or self.blending_penalty:
            indices=torch.arange(self.program_radix**n,device=action.device)
        else:
            # An unmeasured plan has gain -1 before the edit penalty, so it
            # cannot beat the unchanged plan. Search supported complete plans
            # plus that reference instead of expanding the whole alphabet.
            reference_codes=(canonical*powers).sum(-1)
            indices=torch.unique(torch.cat((self.supported_program_codes,reference_codes)))
        available=self.observed[index[...,None],indices]
        predictions=self.values[index[...,None],indices]
        factual=(context.new_zeros(len(context)) if self.paired_advantage else self.anchor(context)[0])
        waiting=(weights*self.wait_values[index]).sum(-1)/weights.sum(-1).clamp_min(1e-8)
        if self.conditional_strata:
            # Learn rescue and harm on the SAME factual future separately.
            # The factual value mixes these effects at execution; the realized
            # future label is never an input to the deployed Adapter.
            predictions=(1.-factual[:,None,None])*predictions[...,0]+factual[:,None,None]*predictions[...,1]
        mass=(weights[...,None]*available).sum(1)
        value=(weights[...,None]*available*predictions).sum(1)/mass.clamp_min(1e-8)
        if self.program_model is not None:
            row,code=torch.nonzero(mass>=1e-6,as_tuple=True)
            value=torch.zeros_like(mass)
            if len(code):
                estimates=[self.program_model(context[r],q) for r,q in
                    zip(row.split(256),indices[code].split(256))]
                value[row,code]=torch.cat(estimates).clamp(-1. if self.paired_advantage else 0.,1.)
            if self.program_offsets is not None:
                # A context intercept calibrates the task-value level without
                # replacing the shared model's learned program differences.
                value+=(weights[...,None]*available*self.program_offsets[index][...,None]).sum(1)/mass.clamp_min(1e-8)
                value=value.clamp(-1. if self.paired_advantage else 0.,1.)
            if self.program_residuals is not None:
                # The shared function transfers conditional response patterns;
                # local paired estimates calibrate it at measured programs.
                # At an isolated observed context this recovers its fitted
                # potential, while retaining learned changes between contexts.
                value+=(weights[...,None]*available*self.program_residuals[index[...,None],indices]).sum(1)/mass.clamp_min(1e-8)
                value=value.clamp(-1. if self.paired_advantage else 0.,1.)
        if self.confidence:
            # Nearby roots on the same trajectory are correlated. Pool their
            # posterior uncertainty without treating retrieval duplicates as
            # new, independent continuation samples.
            variance=(weights[...,None]*available*(self.gain_variance[index[...,None],indices]
                +(predictions-value[:,None]).square())).sum(1)/mass.clamp_min(1e-8)
            if not self.support_uncertainty:value=value-self.confidence*variance.clamp_min(0.).sqrt()
        mean=value-factual[:,None]-waiting[:,None]
        gain=(self.supported_gain(mean,variance,mass) if self.support_uncertainty
            else mean*mass.clamp_max(1.))
        gain=gain.masked_fill(mass<1e-6,-1.)
        if self.effect_values is not None:
            effect=(weights[...,None]*available*self.effect_values[index[...,None],indices]).sum(1)/mass.clamp_min(1e-8)
            # Rank complete programs by their absolute task value, but admit
            # a correction only when paired outcomes support a positive
            # effect against the frozen predecessor at the same context.
            gain=gain.masked_fill(effect*mass.clamp_max(1.)<=self.effect_margin,-1.)
        programs=((indices[:,None]//powers)%self.program_radix).long()
        gain=gain-.005*(programs[None]!=canonical[:,None]).sum(-1)
        # Resolve nearly equivalent value estimates by the physical size of
        # the policy change. Counting altered vessels alone cannot distinguish
        # small mixture adjustments from complete reversals on the same boats.
        choice=gain.argmax(-1)
        if self.counterfactual_credit:
            # COMA-style candidate projection: hold all teammate decisions
            # fixed and marginalize the ordinary critic over this vessel's
            # source-preference distribution. Actual paired-difference labels
            # are absent. The ordinary task-value gain still decides admission.
            alphabet=torch.arange(self.program_radix,device=action.device)
            alternatives=(indices[:,None,None]+(alphabet[None,None]-programs[...,None])*powers[None,:,None])
            seen=self.measured[index[...,None,None,None],alternatives]
            sample_weights=weights[...,None,None,None]*seen
            alternative_mass=sample_weights.sum(1)
            q=(sample_weights*self.values[index[...,None,None,None],alternatives]).sum(1)/alternative_mass.clamp_min(1e-8)
            feedback=action[...,2].gather(1,order)
            if self.program_options is None:
                blend=torch.where(alphabet>0,(alphabet.to(action.dtype)-1.)*.5,-1.)
                coefficients=torch.where(blend[None,None]>=0.,blend[None,None],feedback[...,None])
                deviation=((coefficients-feedback[...,None])/.4).square()
            else:
                phases=self.program_options[None,None]
                coefficients=torch.where(phases>=0.,phases,feedback[...,None,None])
                if self.prefix_refinement:
                    held=action[...,[3,5]].gather(1,order[...,None].expand(-1,-1,2))
                    reference=torch.where(held>=0.,held,feedback[...,None])
                    coefficients=torch.where(phases==-2.,reference[:,:,None],coefficients)
                deviation=((coefficients-feedback[...,None,None])/.4).square().mean(-1)
            probability=(-deviation).softmax(-1)
            probability=probability[:,None]*(alternative_mass>=1e-6)
            expectation=(probability*q).sum(-1)/probability.sum(-1).clamp_min(1e-8)
            credit=(value[...,None]-expectation).sum(-1)
            credit=credit-.005*(programs[None]!=canonical[:,None]).sum(-1)
            admitted=gain>self.margin
            preferred=credit.masked_fill(~admitted,float('-inf')).argmax(-1)
            choice=torch.where(admitted.any(-1),preferred,choice)
        if self.decision_tolerance or self.blending_penalty:
            feedback=action[...,2].gather(1,order)
            reference=torch.where(canonical>0,(canonical.to(action.dtype)-1.)*.5,feedback)
            coefficients=torch.where(programs[None]>0,(programs.to(action.dtype)[None]-1.)*.5,
                feedback[:,None])
            if self.program_options is None:displacement=(coefficients-reference[:,None]).square().sum(-1)
            else:
                proposed=self.program_options[programs]
                prior=self.program_options[canonical]
                proposed=torch.where(proposed[None]>=0.,proposed[None],feedback[:,None,:,None])
                prior=torch.where(prior>=0.,prior,feedback[...,None])
                displacement=(proposed-prior[:,None]).square().mean(-1).sum(-1)
            gain=gain-self.blending_penalty*displacement
            choice=gain.argmax(-1)
            if self.decision_tolerance:
                near=gain>=gain.amax(-1,keepdim=True)-self.decision_tolerance
                choice=displacement.masked_fill(~near,float('inf')).argmin(-1)
        use=gain.gather(1,choice[:,None]).squeeze(1)>self.margin
        selected=programs[choice].gather(1,order.argsort(-1))
        if self.prefix_refinement:
            # Every teammate retains its factual suffix and the original
            # decision time. A zero digit leaves that vessel's entire plan
            # unchanged; other digits alter only its first-phase blend.
            physical=torch.where(selected==0,action[...,3],self.program_options[selected,0])
            replacement=action.clone();replacement[...,3]=physical
            second=self.program_options[selected,1]
            replacement[...,5]=torch.where(second==-2.,action[...,5],second)
            result=torch.where(use[:,None,None],replacement,action)
            return (result,gain.gather(1,choice[:,None]).squeeze(1)) if return_gain else result
        if self.program_options is None:
            physical=torch.where(selected>0,(selected.to(action.dtype)-1.)*.5,-1.)
            second=torch.where((selected>0)&self.reversal,1.-physical,physical)
        else:physical,second=self.program_options[selected].unbind(-1)
        if self.fixed_intervention_clock:
            # An explicitly selected all-feedback program is a timed wait,
            # distinct from rejecting this update and continuing its predecessor.
            second=torch.where((selected==0).all(-1)[:,None],torch.full_like(second,-2.),second)
        replacement=torch.cat((action[...,:3],physical[...,None],
            torch.full_like(physical[...,None],self.span),second[...,None]),-1)
        if action.shape[-1]==3:
            action=torch.cat((action,torch.full_like(physical[...,None],-1.),
                torch.full_like(physical[...,None],self.span),torch.full_like(physical[...,None],-1.)),-1)
        result=torch.where(use[:,None,None],replacement,action)
        return (result,gain.gather(1,choice[:,None]).squeeze(1)) if return_gain else result

    def transported_action(self,context,order,action,return_gain=False):
        """Transfer a supported joint intervention to a differently sized team.

        Changed vessels are never silently dropped: a missing training vessel
        must have used the free feedback option. Extra deployment vessels keep
        that same free option. Values of identical transported programs are
        pooled only after their vessel assignments agree.
        """
        b,n=action.shape[:2]
        nearest,index,mapping=match_fleet_context(context,self.contexts)
        weights=torch.exp(-nearest/(2.*self.bandwidth**2))
        if action.shape[-1]==3:
            free=torch.full_like(action[...,:1],-1.)
            action=torch.cat((action,free,torch.full_like(free,self.span),free),-1)
        result=action.clone();best_gain=context.new_full((b,),-1.)
        if (weights.sum(-1)<=self.margin).all():
            return (result,best_gain) if return_gain else result
        m=mapping.shape[-1]
        ids=torch.arange(self.values.shape[1],device=action.device)
        programs=((ids[:,None]//(4**torch.arange(m,device=action.device)))%4).long()
        factual=(context.new_zeros(b) if self.paired_advantage else
            self.anchor(context,set_context=True)[0])
        powers=4**torch.arange(n,device=action.device)
        for row in range(b):
            candidate_codes=[];candidate_mass=[];candidate_value=[];candidate_wait=[];candidate_second=[]
            for j,source in enumerate(index[row]):
                if weights[row,j]<1e-8:continue
                roles=mapping[row,j];missing=roles<0
                available=self.observed[source].clone()
                if missing.any():available&=(programs[:,missing]==0).all(-1)
                chosen=programs[available]
                if not len(chosen):continue
                assigned=torch.zeros((len(chosen),n),dtype=torch.long,device=action.device)
                assigned[:,roles[~missing]]=chosen[:,~missing]
                value=self.values[source,available]
                if self.conditional_strata:
                    value=(1.-factual[row])*value[...,0]+factual[row]*value[...,1]
                mass=weights[row,j].expand(len(chosen))
                candidate_codes.append((assigned*powers).sum(-1))
                candidate_mass.append(mass);candidate_value.append(mass*value)
                candidate_wait.append(mass*self.wait_values[source])
                if self.confidence:candidate_second.append(mass*(self.gain_variance[source,available]+value.square()))
            if not candidate_codes:continue
            codes,inverse=torch.cat(candidate_codes).unique(return_inverse=True)
            mass=context.new_zeros(len(codes)).scatter_add_(0,inverse,torch.cat(candidate_mass))
            values=context.new_zeros(len(codes)).scatter_add_(0,inverse,torch.cat(candidate_value))
            waits=context.new_zeros(len(codes)).scatter_add_(0,inverse,torch.cat(candidate_wait))
            transferred=(codes[:,None]//powers)%4
            baseline=torch.where(action[row,:,3]>=0.,1.+2.*action[row,:,3],0.).long()[order[row]]
            mean=values/mass.clamp_min(1e-8)
            if self.confidence:
                second=context.new_zeros(len(codes)).scatter_add_(0,inverse,torch.cat(candidate_second))/mass.clamp_min(1e-8)
                variance=(second-mean.square()).clamp_min(0.)
                if not self.support_uncertainty:mean=mean-self.confidence*variance.sqrt()
            mean=mean-waits/mass.clamp_min(1e-8)-factual[row]
            gain=(self.supported_gain(mean,variance,mass) if self.support_uncertainty
                else mean*mass.clamp_max(1.))
            gain=gain-.005*(transferred!=baseline[None]).sum(-1)
            choice=gain.argmax();best_gain[row]=gain[choice]
            if gain[choice]>self.margin:
                selected=transferred[choice][order[row].argsort()]
                physical=torch.where(selected>0,(selected.to(action.dtype)-1.)*.5,-1.)
                second=torch.where((selected>0)&self.reversal,1.-physical,physical)
                if self.fixed_intervention_clock and not bool((selected>0).any()):second.fill_(-2.)
                result[row,:,3]=physical;result[row,:,4]=self.span;result[row,:,5]=second
        return (result,best_gain) if return_gain else result

    def local_effect_action(self,context,order,action,return_gain=False):
        """Transfer effects by treated vessels and their physical neighbors.

        Far, unaltered vessels should not veto an otherwise supported causal
        comparison solely through nuisance heading/current differences. They
        retain nonzero influence; nearby spectators and all treated vessels
        remain in the one-to-one assignment. All observed signs are pooled.
        """
        from scipy.optimize import linear_sum_assignment
        b,n=action.shape[:2];extra=context.shape[-1]%8;m=(self.contexts.shape[-1]-extra)//8
        if action.shape[-1]==3:
            free=torch.full_like(action[...,:1],-1.)
            action=torch.cat((action,free,torch.full_like(free,self.span),free),-1)
        result=action.clone();best_gain=context.new_full((b,),-1.)
        if not len(self.effect_centers):return (result,best_gain) if return_gain else result
        _,costs=fleet_set_cost(context,self.contexts)
        coarse=(costs.amin(-2)[:,self.effect_centers]*self.effect_weights).sum(-1)/self.effect_weights.sum(-1)
        global_cost=(context[:,None,-extra:]-self.contexts[None,:,-extra:]).square().sum(-1) if extra else costs.new_zeros((b,len(self.contexts)))
        coarse=coarse+global_cost[:,self.effect_centers]
        shortlisted=coarse.topk(min(24,coarse.shape[-1]),largest=False).indices
        powers=4**torch.arange(n,device=context.device);source_powers=4**torch.arange(m,device=context.device)
        factual=context.new_zeros(b) if self.paired_advantage else self.anchor(context,set_context=True)[0]
        for row in range(b):
            records={}
            for group in shortlisted[row]:
                source=int(self.effect_centers[group]);influence=self.effect_weights[group]
                cost=costs[row,source];left,right=linear_sum_assignment((cost*influence[None]).detach().cpu().numpy())
                left=torch.as_tensor(left,device=context.device);right=torch.as_tensor(right,device=context.device)
                missing=torch.ones(m,dtype=torch.bool,device=context.device);missing[right]=False
                if (missing&self.effect_masks[group]).any():continue
                distance=(cost[left,right]*influence[right]).sum()/influence.sum()+global_cost[row,source]
                if missing.any():distance=distance+(cost.amin(0)[missing]*influence[missing]).sum()/influence.sum()
                if n>m:
                    unmatched=torch.ones(n,dtype=torch.bool,device=context.device);unmatched[left]=False
                    treated=left[self.effect_masks[group,right]]
                    positions=context[row,:8*n].reshape(n,8)[:,:2]*30.
                    near=(torch.exp(-(positions[:,None]-positions[treated][None]).square().sum(-1).amin(-1)/200.)
                        if len(treated) else context.new_ones(n))
                    distance=distance+.25*(cost.amin(-1)[unmatched]*near[unmatched]).sum()/n
                mass=torch.exp(-distance/(2.*self.bandwidth**2))
                if mass<1e-8:continue
                selected=self.effect_programs[self.effect_offsets[group]:self.effect_offsets[group+1]]
                program=(selected[:,None]//source_powers)%4
                transferred=torch.zeros((len(program),n),dtype=torch.long,device=context.device)
                transferred[:,left]=program[:,right];codes=(transferred*powers).sum(-1)
                values=self.values[source,selected]
                if self.conditional_strata:values=(1.-factual[row])*values[...,0]+factual[row]*values[...,1]
                for code,value,variance in zip(codes,values,self.gain_variance[source,selected]):
                    key=(source,int(code))
                    if key not in records or mass>records[key][0]:
                        records[key]=(mass,value,self.wait_values[source],variance)
            if not records:continue
            codes=torch.tensor([key[1] for key in records],device=context.device)
            unique,inverse=codes.unique(return_inverse=True)
            entries=list(records.values());weight=torch.stack([entry[0] for entry in entries])
            mass=context.new_zeros(len(unique)).scatter_add_(0,inverse,weight)
            delta=torch.stack([entry[1]-entry[2] for entry in entries])
            value=context.new_zeros(len(unique)).scatter_add_(0,inverse,weight*delta)/mass.clamp_min(1e-8)
            if self.confidence:
                second=torch.stack([entry[3] for entry in entries])+delta.square()
                second=context.new_zeros(len(unique)).scatter_add_(0,inverse,weight*second)/mass.clamp_min(1e-8)
                variance=(second-value.square()).clamp_min(0.)
                if not self.support_uncertainty:value=value-self.confidence*variance.sqrt()
            programs=(unique[:,None]//powers)%4
            baseline=torch.where(action[row,:,3]>=0.,1.+2.*action[row,:,3],0.).long()[order[row]]
            mean=value-factual[row]
            gain=(self.supported_gain(mean,variance,mass) if self.support_uncertainty
                else mean*mass.clamp_max(1.))
            gain=gain-.005*(programs!=baseline[None]).sum(-1)
            choice=gain.argmax();best_gain[row]=gain[choice]
            if gain[choice]>self.margin:
                selected=programs[choice][order[row].argsort()]
                physical=torch.where(selected>0,(selected.to(action.dtype)-1.)*.5,-1.)
                second=torch.where((selected>0)&self.reversal,1.-physical,physical)
                if self.fixed_intervention_clock and not bool((selected>0).any()):second.fill_(-2.)
                result[row,:,3]=physical;result[row,:,4]=self.span;result[row,:,5]=second
        return (result,best_gain) if return_gain else result


class InitialRoleNetwork(nn.Module):
    """Shared conditional return factors for single-vessel temporal roles.

    Inputs are current public geometry and the preceding policy's commands.
    No predicted candidate trajectories, scene identities or future outcomes
    are available to the network. The same factors apply to any fleet size.
    """
    def __init__(self,width=32):
        super().__init__()
        self.message=nn.Sequential(nn.Linear(39,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.node=nn.Sequential(nn.Linear(15+2*width,width),nn.SiLU())
        self.effect=nn.Sequential(nn.Linear(3*width+3,width),nn.SiLU(),nn.Linear(width,1))

    def forward(self,packet,action):
        node=torch.cat((packet['node'][...,:9],action[...,:3],
            packet['node'][...,11:13],packet['node'][...,14:15]),-1)
        b,n=node.shape[:2]
        edge=packet['edge'][...,:9]
        messages=self.message(torch.cat((node[:,:,None].expand(-1,-1,n,-1),
            node[:,None].expand(-1,n,-1,-1),edge),-1))
        mask=~torch.eye(n,dtype=torch.bool,device=node.device)[None]
        weights=torch.exp(-.75*edge[...,7])*mask
        average=(messages*weights[...,None]).sum(2)/weights.sum(2).clamp_min(1e-6)[...,None]
        maximum=messages.masked_fill(~mask[...,None],-1e6).amax(2)
        encoded=self.node(torch.cat((node,average,maximum),-1))
        context=torch.cat((encoded,encoded.mean(1,keepdim=True).expand(-1,n,-1),
            encoded.amax(1,keepdim=True).expand(-1,n,-1)),-1)
        gate=node.new_tensor([0.,.5,1.]).expand(b,n,-1)
        difference=gate-action[...,2,None]
        options=torch.stack((gate,difference,difference.square()),-1)
        return self.effect(torch.cat((context[:,:,None].expand(-1,-1,3,-1),options),-1)).squeeze(-1).tanh()


class InitialRoleCredit(nn.Module):
    """Select a persistent role using learned, permutation-equivariant credit."""
    def __init__(self,states,width=32,span=80,confidence=.5,margin=0.):
        super().__init__()
        self.models=nn.ModuleList()
        for state in states:
            net=InitialRoleNetwork(width);net.load_state_dict(state);self.models.append(net.eval())
        self.span=span;self.confidence=confidence;self.margin=margin

    def forward(self,packet,action):
        initial=packet['node'][:,0,15].abs()<1e-7
        if not initial.any():return action
        values=torch.stack([model(packet,action) for model in self.models])
        score=values.mean(0)-self.confidence*values.std(0,unbiased=False)-.005
        gain,index=score.flatten(1).max(-1)
        n=action.shape[1]
        physical=torch.full_like(action[...,:1],-1.)
        physical[torch.arange(len(action),device=action.device),index//3,0]=(index%3).to(action.dtype)*.5
        replacement=torch.cat((action[...,:3],physical,torch.full_like(physical,self.span),physical),-1)
        if action.shape[-1]==3:
            free=torch.full_like(physical,-1.)
            action=torch.cat((action,free,torch.full_like(free,self.span),free),-1)
        return torch.where((initial&(gain>self.margin))[:,None,None],replacement,action)


def fit_initial_role_credit(dataset,width=32,heads=5,updates=500,seed=42):
    """Fit paired terminal credit; bootstrap whole layouts, not graph edges."""
    import numpy as np
    packets=[];actions=[];targets=[];precisions=[]
    for rollout in dataset['rollouts']:
        branches=[b for b in rollout['branches'] if not b.get('observed_future',False)]
        if not branches:continue
        if any(b['root'][1]!=0 for b in branches):
            raise ValueError('Initial-role credit requires initial-state interventions.')
        first=branches[0];n=len(first['reference_action']);samples={}
        for branch in branches:
            reference=branch['factual_outcome']['success']
            if reference is None:continue
            future=branch.get('continuation_seed',branch['future_repetition'])
            for side,outcome in (('left','before'),('right','after')):
                value=branch[outcome]['success'];gates=branch[side][:,3]
                changed=np.flatnonzero(gates>=0.)
                if value is None or len(changed)!=1:continue
                subject=int(changed[0]);option=int(round(2.*gates[subject]))
                samples.setdefault((subject,option),{})[future]=float(value-reference)
        if len(samples)!=3*n:raise ValueError('All single-vessel role alternatives are required.')
        target=np.zeros((n,3));precision=np.zeros((n,3))
        for (subject,option),observations in samples.items():
            values=np.asarray(list(observations.values()))
            target[subject,option]=values.mean()
            precision[subject,option]=1./(.1+values.var()/len(values))
        packets.append(first['packet']);actions.append(first['reference_action'][:,:3])
        targets.append(target);precisions.append(precision)
    packet={key:torch.tensor(np.stack([p[key] for p in packets]),dtype=torch.float32)
        for key in ('node','edge')}
    action=torch.tensor(np.stack(actions),dtype=torch.float32)
    target=torch.tensor(np.stack(targets),dtype=torch.float32)
    weight=torch.tensor(np.stack(precisions),dtype=torch.float32);weight=weight/weight.mean()
    states=[];losses=[]
    with torch.enable_grad():
        for head in range(heads):
            torch.manual_seed(seed+head*7919)
            net=InitialRoleNetwork(width)
            index=torch.randint(len(action),(len(action),))
            inputs={key:value[index] for key,value in packet.items()}
            optimizer=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=.02)
            for _ in range(updates):
                prediction=net(inputs,action[index])
                loss=((prediction-target[index]).square()*weight[index]).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
            states.append({key:value.detach().cpu() for key,value in net.state_dict().items()})
            losses.append(float(loss.detach()))
    return dict(states=states,width=width,span=dataset['model']['physical_span'],confidence=.5,margin=0.),dict(
        layouts=len(actions),heads=heads,updates=updates,losses=losses,seed=seed,
        supervision='independent paired terminal-return differences; layout bootstrap')


class FeedbackRoleNetwork(nn.Module):
    """Current-state conditional credit shared across boats and encounters."""
    def __init__(self,width=48):
        super().__init__()
        self.message=nn.Sequential(nn.Linear(41,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.node=nn.Sequential(nn.Linear(16+2*width,width),nn.SiLU())
        self.effect=nn.Sequential(nn.Linear(3*width+5,width),nn.SiLU(),nn.Linear(width,1))

    def forward(self,packet,action):
        public=packet['node']
        node=torch.cat((public[...,:9],action[...,:3],public[...,11:13],public[...,14:16]),-1)
        node=node.clamp(-20.,20.)
        b,n=node.shape[:2];edge=packet['edge'][...,:9].clamp(-20.,20.)
        messages=self.message(torch.cat((node[:,:,None].expand(-1,-1,n,-1),
            node[:,None].expand(-1,n,-1,-1),edge),-1))
        mask=~torch.eye(n,dtype=torch.bool,device=node.device)[None]
        weights=torch.exp(-.75*edge[...,7])*mask
        average=(messages*weights[...,None]).sum(2)/weights.sum(2).clamp_min(1e-6)[...,None]
        maximum=messages.masked_fill(~mask[...,None],-1e6).amax(2)
        encoded=self.node(torch.cat((node,average,maximum),-1))
        context=torch.cat((encoded,encoded.mean(1,keepdim=True).expand(-1,n,-1),
            encoded.amax(1,keepdim=True).expand(-1,n,-1)),-1)
        gates=node.new_tensor([0.,.5,1.]).expand(b,n,-1)
        difference=gates-action[...,2,None]
        response=difference[...,None]*(action[...,:2]-public[...,11:13])[:,:,None]
        options=torch.cat((torch.stack((gates,difference,difference.square()),-1),response),-1)
        return self.effect(torch.cat((context[:,:,None].expand(-1,-1,3,-1),options),-1)).squeeze(-1).tanh()


class FeedbackRoleCredit(nn.Module):
    """Renew a short role from public feedback, with layout-bootstrap credit."""
    def __init__(self,states,width=48,span=20,confidence=.5,margin=.02,
                 support_contexts=None,support_bandwidth=.3):
        super().__init__();self.models=nn.ModuleList()
        for state in states:
            model=FeedbackRoleNetwork(width);model.load_state_dict(state);self.models.append(model.eval())
        self.span=span;self.confidence=confidence;self.margin=margin
        self.register_buffer('support_contexts',None if support_contexts is None else
            torch.as_tensor(support_contexts,dtype=torch.float32))
        self.support_bandwidth=support_bandwidth

    def forward(self,packet,action):
        values=torch.stack([model(packet,action) for model in self.models])
        score=values.mean(0)-self.confidence*values.std(0,unbiased=False)
        if self.support_contexts is not None:
            context=public_return_context(packet,version=2)
            distance,_,_=match_fleet_context(context,self.support_contexts,count=1)
            support=torch.exp(-distance[:,0]/(2.*self.support_bandwidth**2))
            # A shared network can extrapolate the same positive credit far
            # before its training encounters. Only supported state feedback
            # can authorize the learned intervention; no outcome is queried.
            score=score*support[:,None,None]
        gain,index=score.flatten(1).max(-1)
        physical=torch.full_like(action[...,:1],-1.)
        physical[torch.arange(len(action),device=action.device),index//3,0]=(index%3).to(action.dtype)*.5
        physical=torch.where((gain>self.margin)[:,None,None],physical,-torch.ones_like(physical))
        return torch.cat((action[...,:3],physical,torch.full_like(physical,self.span),physical),-1)


def fit_feedback_role_credit(dataset,width=48,heads=5,updates=1200,seed=42,device='cpu'):
    """Fit all paired options; stop on held-out whole layouts, not selected maxima."""
    import numpy as np
    packets=[];actions=[];targets=[];precisions=[];layouts=[];counts=[]
    for rollout in dataset['rollouts']:
        branches=[b for b in rollout['branches'] if not b.get('observed_future',False)]
        if not branches:continue
        first=branches[0];n=len(first['reference_action']);samples={}
        for branch in branches:
            reference=branch['factual_outcome']['success']
            future=branch['continuation_seed']
            for side,outcome in (('left','before'),('right','after')):
                value=branch[outcome]['success'];gates=branch[side][:,3]
                changed=np.flatnonzero(gates>=0.)
                if reference is None or value is None or len(changed)!=1:continue
                subject=int(changed[0]);option=int(round(2.*gates[subject]))
                samples.setdefault((subject,option),{})[future]=float(value-reference)
        if len(samples)!=3*n:raise ValueError('Feedback credit requires every single-vessel alternative.')
        target=np.zeros((n,3));precision=np.zeros((n,3))
        for (subject,option),observations in samples.items():
            values=np.asarray(list(observations.values()))
            target[subject,option]=values.mean()
            precision[subject,option]=1./(.25+values.var()/len(values))
        packets.append(first['packet']);actions.append(first['reference_action'][:,:3])
        targets.append(target);precisions.append(precision);layouts.append(rollout['row']['seed'])
        counts.append(min(len(values) for values in samples.values()))
    packet={key:torch.as_tensor(np.stack([p[key] for p in packets]),dtype=torch.float32,device=device)
        for key in ('node','edge')}
    action=torch.as_tensor(np.stack(actions),dtype=torch.float32,device=device)
    target=torch.as_tensor(np.stack(targets),dtype=torch.float32,device=device)
    weight=torch.as_tensor(np.stack(precisions),dtype=torch.float32,device=device);weight=weight/weight.mean()
    unique=np.unique(layouts);layouts=np.asarray(layouts);states=[];fits=[]
    if len(unique)<8:raise ValueError('Cross-layout credit needs at least eight independent layouts.')
    with torch.enable_grad():
        for head in range(heads):
            torch.manual_seed(seed+head*7919);rng=np.random.default_rng(seed+head*7919)
            sampled=rng.choice(unique,size=len(unique),replace=True)
            holdout=np.flatnonzero(~np.isin(layouts,sampled))
            if not len(holdout):raise ValueError('A bootstrap head has no independent validation layout.')
            train=np.concatenate([np.flatnonzero(layouts==layout) for layout in sampled])
            index=torch.as_tensor(train,device=device);validation=torch.as_tensor(holdout,device=device)
            inputs={key:value[index] for key,value in packet.items()}
            valid_inputs={key:value[validation] for key,value in packet.items()}
            net=FeedbackRoleNetwork(width).to(device)
            optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=.03)
            best=float('inf');best_state=None;best_update=0
            for update in range(updates):
                prediction=net(inputs,action[index])
                loss=((prediction-target[index]).square()*weight[index]).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
                if (update+1)%20==0:
                    with torch.no_grad():
                        validation_loss=float(((net(valid_inputs,action[validation])-target[validation]).square()
                            *weight[validation]).mean())
                    if validation_loss<best:
                        best=validation_loss;best_update=update+1
                        best_state={key:value.detach().cpu().clone() for key,value in net.state_dict().items()}
                    elif update+1-best_update>=240:break
            states.append(best_state)
            fits.append(dict(validation_loss=best,selected_update=best_update,
                training_layouts=len(np.unique(sampled)),validation_layouts=len(np.unique(layouts[holdout]))))
    return dict(states=states,width=width,span=dataset['model']['physical_intervention_span'],
        confidence=.5,margin=.02),dict(roots=len(actions),layouts=len(unique),heads=heads,
        independent_futures=min(counts),fits=fits,seed=seed,
        supervision='All paired terminal differences; bootstrap and early stopping on whole layouts.')


def cooperative_role_programs(n,device,all_pairs=False):
    programs=[[-1.]*n]
    for i in range(n):
        for gate in (0.,.5,1.):
            role=[-1.]*n;role[i]=gate;programs.append(role)
    if all_pairs:
        for i in range(n):
            for j in range(i+1,n):
                for left in (0.,.5,1.):
                    for right in (0.,.5,1.):
                        role=[-1.]*n;role[i]=left;role[j]=right;programs.append(role)
    else:
        for i in range(n):
            for j in range(n):
                if i==j:continue
                role=[-1.]*n;role[i]=0.;role[j]=1.;programs.append(role)
    return torch.tensor(programs,dtype=torch.float32,device=device)


def conditional_action_programs(n,device,proposal=False,all_pairs=False):
    gates=cooperative_role_programs(n,device,all_pairs)
    offsets=torch.zeros((len(gates),n,2),dtype=torch.float32,device=device)
    if proposal:
        extra=[];corrections=[]
        for subject in range(n):
            for change in ((.5,.5),(-.5,-.5),(.5,-.5),(-.5,.5)):
                role=torch.full((n,),-1.,device=device);role[subject]=1.
                correction=torch.zeros((n,2),device=device);correction[subject]=torch.tensor(change,device=device)
                extra.append(role);corrections.append(correction)
        gates=torch.cat((gates,torch.stack(extra)));offsets=torch.cat((offsets,torch.stack(corrections)))
    return gates,offsets


def finite_credit_context(packet,action):
    """Own controls and measured local geometry, without candidate forecasts."""
    edge=packet['edge'][...,:9];n=edge.shape[1]
    distance=edge[...,7].masked_fill(torch.eye(n,device=edge.device,dtype=torch.bool)[None],float('inf'))
    neighbor=distance.argmin(-1)
    nearest=edge.gather(2,neighbor[...,None,None].expand(-1,-1,1,9)).squeeze(2)
    return torch.cat((packet['node'][...,:9],packet['node'][...,11:13],action[...,:3],nearest),-1)


def public_credit_packet(packet):
    """Build decision inputs from measured motion and relative observations."""
    motion=packet['motion'];c=motion[:,0,2].cos();s=motion[:,0,2].sin()
    def world(vector):
        return torch.stack((c*vector[:,0]-s*vector[:,1],s*vector[:,0]+c*vector[:,1]),-1)
    target=motion[:,0,:2]+world(packet['node'][:,0,2:4]*30.)
    velocity=world(packet['node'][:,0,4:6]*5.);direction=velocity/velocity.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    task=torch.cat((target/60.,direction[:,1:2],direction[:,:1],velocity/5.,
        target.new_zeros((len(target),2)),packet['task'][:,8:9]),-1)
    return dict(packet,water=motion[...,3:6]/5.,task=task)


class PairedTerminalOutcome(nn.Module):
    """Subject-centered probabilities for the four paired terminal outcomes."""
    def __init__(self,width=64,public_observations=True):
        super().__init__();self.public_observations=public_observations
        self.node=nn.Sequential(nn.Linear(24,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.pair=nn.Sequential(nn.Linear(19,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.head=nn.Sequential(nn.Linear(width*5+9,192),nn.SiLU(),nn.Linear(192,4))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,packet,left,right,subjects,duration):
        if self.public_observations:packet=public_credit_packet(packet)
        batch,n=left.shape[:2];rows=torch.arange(batch,device=left.device)
        marker=F.one_hot(subjects,num_classes=n).to(left.dtype)[...,None]
        water=packet['water'];heading=packet['motion'][...,2];c,s=heading.cos(),heading.sin()
        water=torch.stack((c*water[...,0]+s*water[...,1],-s*water[...,0]+c*water[...,1],water[...,2]),-1)
        span=duration[:,None,None].expand(-1,n,1)/10.
        node=torch.cat((packet['node'][...,:9],packet['node'][...,14:16],water,left,
            packet['node'][...,11:13],right,span,marker),-1)
        lf=left[...,2:3]*left[...,:2]+(1.-left[...,2:3])*packet['node'][...,11:13]
        rf=right[...,2:3]*right[...,:2]+(1.-right[...,2:3])*packet['node'][...,11:13]
        own=lambda x:x[:,:,None].expand(-1,-1,n,-1)
        peer=lambda x:x[:,None].expand(-1,n,-1,-1)
        edge=torch.cat((packet['edge'][...,:9],own(lf),peer(lf),own(rf),peer(rf),own(marker),peer(marker)),-1)
        h=self.node(node);e=self.pair(edge)[rows,subjects]
        mask=~marker.squeeze(-1).bool()
        task=packet['task'];radial=task[:,:2];radius=radial.norm(dim=-1).clamp_min(1e-8);radial=radial/radius[:,None]
        velocity=task[:,4:6];direction=torch.stack((task[:,3],task[:,2]),-1)
        cross=lambda a,b:a[:,0]*b[:,1]-a[:,1]*b[:,0]
        closest=(packet['motion'][...,:2]/60.-task[:,None,:2]).norm(dim=-1).amin(1)
        context=torch.stack((radius,(radial*velocity).sum(-1),cross(radial,velocity),
            (radial*direction).sum(-1),cross(radial,direction),task[:,6],task[:,7],task[:,8],closest),-1)
        pooled=torch.cat((h[rows,subjects],h.mean(1),h.amax(1),
            (e*mask[...,None]).sum(1)/(n-1),e.masked_fill(~mask[...,None],-1e5).amax(1),context),-1)
        return self.head(pooled).log_softmax(-1)


class FiniteTemporalCredit(nn.Module):
    """Single-vessel physical interventions with learned duration and waiting.

    Candidate values are trained on actual terminal continuations. The gate
    policy is evaluated first; this module changes one physical coefficient.
    Neither candidate trajectory prediction nor M1's joint gate search is
    implemented here.
    """
    def __init__(self,states,spans,width,contexts,center,scale,bandwidth,
            confidence=1.,margin=.01,max_interventions=1,wait_states=(),use_wait=True,conditional=True,fit=None,
            outcome_states=(),public_observations=False):
        super().__init__()
        import develop_source_fusion as d
        self.models=nn.ModuleList()
        for span,ensemble in zip(spans,states):
            nets=nn.ModuleList()
            for state in ensemble:
                net=d.JointLearningValue(width=width,relational=True,invariant=True,options=span>1,temporal_option=span>1)
                net.load_state_dict(state);nets.append(net.eval())
            self.models.append(nets)
        self.outcomes=nn.ModuleList();self.public_observations=public_observations
        for state in outcome_states:
            net=PairedTerminalOutcome(width,public_observations);net.load_state_dict(state);self.outcomes.append(net.eval())
        self.waiters=nn.ModuleList()
        for state in wait_states:
            net=d.JointLearningValue(width=width,relational=True,invariant=True)
            net.load_state_dict(state);self.waiters.append(net.eval())
        self.spans=tuple(spans);self.width=width;self.confidence=confidence;self.margin=margin
        self.max_interventions=max_interventions;self.use_wait=use_wait;self.conditional=conditional
        self.bandwidth=bandwidth
        self.register_buffer('contexts',torch.as_tensor(contexts,dtype=torch.float32))
        self.register_buffer('center',torch.as_tensor(center,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))

    def eligible(self,packet):
        count=packet.get('_conditional_count',packet['node'].new_zeros((len(packet['node']),1)))[:,0]
        return count<self.max_interventions if self.max_interventions else torch.ones_like(count,dtype=torch.bool)

    def estimates(self,packet,action):
        batch,n=action.shape[:2]
        choices=action[:,None].expand(-1,n,-1,-1).clone()
        for subject in range(n):choices[:,subject,subject,2]=(action[:,subject,2]<=.5).to(action.dtype)
        expanded={key:value[:,None].expand(-1,n,*value.shape[1:]).reshape(batch*n,*value.shape[1:])
            for key,value in packet.items() if key in ('node','edge','motion','water','task','obs')}
        controls=choices.reshape(batch*n,n,3)
        estimates=[]
        if len(self.outcomes):
            reference=action[:,None].expand(-1,n,-1,-1).reshape(batch*n,n,3)
            subjects=torch.arange(n,device=action.device).repeat(batch)
            for span in self.spans:
                predictions=torch.stack([net(expanded,reference,controls,subjects,
                    action.new_full((batch*n,),span)).exp() for net in self.outcomes])
                estimates.append((predictions[...,1]-predictions[...,2]).reshape(len(self.outcomes),batch,n))
        else:
            for span,ensemble in zip(self.spans,self.models):
                alternatives=dict(expanded,_intervened=(torch.eye(n,device=action.device).repeat(batch,1)
                    if span>1 else action.new_zeros((batch*n,n))))
                estimates.append(torch.stack([net(alternatives,controls).reshape(batch,n)-net(packet,action)[:,None]
                    for net in ensemble]))
        values=torch.stack(estimates,2)
        scores=values.mean(0)-self.confidence*values.std(0,unbiased=False)
        if self.bandwidth>0.:
            query=(finite_credit_context(packet,action)-self.center)/self.scale
            distance=torch.cdist(query.flatten(0,1),self.contexts).square().amin(-1)/query.shape[-1]
            support=torch.exp(-.5*distance.reshape(batch,n)/self.bandwidth)
            scores=scores*support[:,None]
        return scores,choices

    def forward(self,packet,action):
        batch,n=action.shape[:2];eligible=self.eligible(packet)
        roles=action.new_full((batch,n),-1.);duration=action.new_ones(batch)
        if bool(eligible.any()):
            indices=eligible.nonzero().flatten()
            current={key:value[indices] for key,value in packet.items()}
            if self.public_observations:current=public_credit_packet(current)
            scores,choices=self.estimates(current,action[indices,:,:3])
            gain,choice=scores.flatten(1).max(-1)
            wait=gain.new_zeros(len(gain))
            if self.use_wait and len(self.waiters):
                estimates=torch.stack([net(current).clamp(0.,1.) for net in self.waiters])
                wait=(estimates.mean(0)-estimates.std(0,unbiased=False)).clamp_min(0.)
            use=gain>wait+self.margin
            subject=choice%n;local=torch.arange(len(indices),device=action.device)
            roles[indices[use],subject[use]]=choices[local[use],subject[use],subject[use],2]
            duration[indices[use]]=action.new_tensor(self.spans)[choice[use]//n]
        return torch.cat((action[...,:3],roles[...,None],duration[:,None,None].expand(-1,n,1),roles[...,None]),-1)


@torch.enable_grad()
def fit_finite_temporal_credit(dataset,conditional=True,width=64,heads=3,updates=1500,seed=42,device='cpu',wait_updates=1500,initial_states=()):
    import copy
    import numpy as np
    import develop_source_fusion as d
    keys=('node','edge','motion','water','task','obs');rng=np.random.default_rng(seed)
    def packet(records):
        return {key:torch.as_tensor(np.stack([record[key] for record in records]),dtype=torch.float32,device=device) for key in keys}
    states=[];spans=[];fit=[];root_records={}
    for block in dataset['paired_durations']:
        groups={}
        for pair in block['pairs']:
            key=(pair['scene'][0],pair['root_id'],pair['subject'])
            groups.setdefault(key,[]).append(pair)
            root_records.setdefault(key[:2],pair)
        rows=[]
        for group in groups.values():
            rows.append(dict(group[0],before_mean=float(np.mean([p['before']['success'] for p in group])),
                after_mean=float(np.mean([p['after']['success'] for p in group]))))
        inputs=packet([row['packet'] for row in rows])
        left=torch.as_tensor(np.stack([row['left'] for row in rows]),device=device)
        right=torch.as_tensor(np.stack([row['right'] for row in rows]),device=device)
        before=torch.tensor([row['before_mean'] for row in rows],device=device)
        after=torch.tensor([row['after_mean'] for row in rows],device=device)
        scenes=np.array([row['scene'][0] for row in rows]);ensemble=[]
        for head in range(heads):
            torch.manual_seed(seed+head*103+int(block['duration'])*19)
            temporal=block['duration']>1
            net=d.JointLearningValue(width=width,relational=True,invariant=True,options=temporal,temporal_option=temporal).to(device)
            if initial_states:
                source=dict(initial_states[head%len(initial_states)])
                for key in ('control_node.0.weight','control_pair.0.weight'):
                    source[key]=F.pad(source[key],(0,net.state_dict()[key].shape[1]-source[key].shape[1]))
                net.load_state_dict(source)
                # Start both alternatives from the same factual-only teacher.
                # Its terminal-return scale is converted to a probability
                # scale; actual terminal observations then calibrate each span.
                with torch.no_grad():
                    net.value[-1].weight.mul_(.5);net.value[-1].bias.mul_(.5).add_(.5)
                    net.advantage[-1].weight.mul_(.5);net.advantage[-1].bias.mul_(.5)
            optimizer=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=.01)
            held=np.flatnonzero(scenes%heads==head);train=np.flatnonzero(scenes%heads!=head)
            subjects=torch.tensor([row['subject'] for row in rows],device=device)
            intervention=F.one_hot(subjects,num_classes=left.shape[1]).to(left.dtype)
            best=float('inf');best_state=None;best_step=0
            for update in range(updates):
                indices=rng.choice(train,128,replace=True);p={k:v[indices] for k,v in inputs.items()}
                changed=dict(p,_intervened=intervention[indices])
                a=net(p,left[indices]);b=net(changed,right[indices])
                loss=F.mse_loss(a,before[indices])+F.mse_loss(b,after[indices])
                if conditional:loss=loss+8.*F.mse_loss(b-a,(after-before)[indices])
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),2.);optimizer.step()
                if (update+1)%50==0 or update+1==updates:
                    with torch.no_grad():
                        p={k:v[held] for k,v in inputs.items()};a=net(p,left[held])
                        b=net(dict(p,_intervened=intervention[held]),right[held])
                        error=F.mse_loss(b-a,(after-before)[held])+.1*(F.mse_loss(a,before[held])+F.mse_loss(b,after[held]))
                    if float(error)<best:
                        best=float(error);best_state=copy.deepcopy(net.cpu().state_dict());net.to(device);best_step=update+1
                    if update+1-best_step>=350:break
            ensemble.append(best_state)
            record=dict(duration=block['duration'],head=head,validation_loss=best,updates=best_step,conditional=conditional)
            fit.append(record);print(dict(phase='finite_temporal_credit_fit',**record),flush=True)
        states.append(ensemble);spans.append(block['duration'])
    roots=list(root_records.values());p=packet([row['packet'] for row in roots])
    actions=torch.as_tensor(np.stack([row['left'] for row in roots]),device=device)
    context=finite_credit_context(p,actions).flatten(0,1)
    center=context.mean(0);scale=context.std(0).clamp_min(.1);context=(context-center)/scale
    scenes=torch.tensor([row['scene'][0] for row in roots],device=device).repeat_interleave(actions.shape[1])
    distance=torch.cdist(context,context).square()/context.shape[-1]
    distance=distance.masked_fill(scenes[:,None]==scenes[None],float('inf'))
    bandwidth=float(torch.quantile(distance.amin(-1),.9).clamp_min(.05))
    result=dict(states=states,spans=spans,width=width,contexts=context.cpu(),center=center.cpu(),scale=scale.cpu(),
        bandwidth=bandwidth,confidence=1.,margin=.01,max_interventions=1,wait_states=[],use_wait=True,
        conditional=conditional,fit=fit)
    result['wait_states']=fit_finite_wait_value(dataset,result,updates=wait_updates,seed=seed,device=device)
    return result


@torch.enable_grad()
def fit_paired_terminal_outcomes(dataset,conditional=True,width=64,heads=3,updates=2500,seed=42,
        device='cpu',wait_updates=1500,use_wait=True,rank_credit=False):
    import copy
    import numpy as np
    keys=('node','edge','motion','water','task','obs');rng=np.random.default_rng(seed)
    groups={};roots={};spans=[]
    for block in dataset['paired_durations']:
        spans.append(block['duration'])
        for row in block['pairs']:
            key=(row['scene'][0],row['root_id'],row['subject'],block['duration'])
            groups.setdefault(key,[]).append(row);roots.setdefault(key[:2],row)
    rows=[]
    for key,values in groups.items():
        values=sorted(values,key=lambda row:row['future_repetition'])
        outcomes=[2*row['before']['success']+row['after']['success'] for row in values]
        effects=np.array([row['after']['success']-row['before']['success'] for row in values],dtype=np.float32)
        rows.append(dict(values[0],duration=key[3],probabilities=np.bincount(outcomes,minlength=4)/len(values),effects=effects))
    def packet(records):
        return {key:torch.as_tensor(np.stack([row[key] for row in records]),dtype=torch.float32,device=device) for key in keys}
    inputs=packet([row['packet'] for row in rows])
    left=torch.as_tensor(np.stack([row['left'] for row in rows]),device=device)
    right=torch.as_tensor(np.stack([row['right'] for row in rows]),device=device)
    subjects=torch.tensor([row['subject'] for row in rows],device=device)
    durations=torch.tensor([row['duration'] for row in rows],device=device,dtype=torch.float32)
    truth=torch.as_tensor(np.stack([row['probabilities'] for row in rows]),device=device,dtype=torch.float32)
    effects=torch.as_tensor(np.stack([row['effects'] for row in rows]),device=device)
    root_indices={}
    for i,row in enumerate(rows):root_indices.setdefault((row['scene'][0],row['root_id']),[]).append(i)
    partners=[root_indices[row['scene'][0],row['root_id']] for row in rows]
    scenes=np.array([row['scene'][0] for row in rows]);states=[];fit=[]
    for head in range(heads):
        torch.manual_seed(seed+head*103);net=PairedTerminalOutcome(width).to(device)
        held=np.flatnonzero(scenes%heads==head);train=np.flatnonzero(scenes%heads!=head)
        held_positions={int(index):position for position,index in enumerate(held)}
        held_roots=[np.array([held_positions[index] for index in group]) for group in root_indices.values() if group[0] in held_positions]
        with torch.no_grad():net.head[-1].bias.copy_(truth[train].mean(0).clamp_min(.001).log())
        optimizer=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.01)
        best=float('inf');best_state=None;best_step=0;selected_gain=None;independent_gain=None
        for update in range(updates):
            indices=rng.choice(train,128,replace=True);p={key:value[indices] for key,value in inputs.items()}
            logp=net(p,left[indices],right[indices],subjects[indices],durations[indices])
            if conditional:loss=-(truth[indices]*logp).sum(-1).mean()
            else:
                predicted=logp.exp();before=predicted[:,2]+predicted[:,3];after=predicted[:,1]+predicted[:,3]
                loss=F.binary_cross_entropy(before,truth[indices,2]+truth[indices,3])+F.binary_cross_entropy(after,truth[indices,1]+truth[indices,3])
            if conditional and rank_credit:
                other=np.array([rng.choice(partners[int(index)]) for index in indices])
                alternative=net({key:value[other] for key,value in inputs.items()},left[other],right[other],subjects[other],durations[other]).exp()
                prediction=logp.exp();first_effect=prediction[:,1]-prediction[:,2]
                second_effect=alternative[:,1]-alternative[:,2]
                null=torch.as_tensor(rng.random(len(indices))<.25,device=device)
                second_effect=torch.where(null,torch.zeros_like(second_effect),second_effect)
                difference=effects[indices]-torch.where(null[:,None],torch.zeros_like(effects[other]),effects[other])
                mean=difference.mean(-1);variance=difference.var(-1,unbiased=True)/difference.shape[-1]
                reliability=mean.square()/(mean.square()+variance+.001)
                preference=(mean/(2.*variance.sqrt()+.025)).sigmoid()
                ranking=F.binary_cross_entropy_with_logits((first_effect-second_effect)/.1,preference,reduction='none')
                loss=loss+.5*(reliability*ranking).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),2.);optimizer.step()
            if (update+1)%50==0 or update+1==updates:
                with torch.no_grad():
                    predicted=torch.cat([net({key:value[part] for key,value in inputs.items()},
                        left[part],right[part],subjects[part],durations[part]).exp() for part in np.array_split(held,max(1,len(held)//256))])
                    actual=truth[held];effect=predicted[:,1]-predicted[:,2];target=actual[:,1]-actual[:,2]
                    error=F.mse_loss(effect,target)+.1*(F.mse_loss(predicted[:,2]+predicted[:,3],actual[:,2]+actual[:,3])+
                        F.mse_loss(predicted[:,1]+predicted[:,3],actual[:,1]+actual[:,3]))
                    criterion=float(error);gain=check_gain=None
                    if rank_credit:
                        chosen=[]
                        for group in held_roots:
                            best_action=int(group[int(effect[group].argmax().item())])
                            chosen.append(best_action if effect[best_action]>.01 else -1)
                        # Checkpoint selection uses one half of independent
                        # future realizations; the other half is only reported.
                        half=effects.shape[1]//2;actual_effects=effects[held]
                        gain=float(torch.stack([actual_effects[index,:half].mean() if index>=0 else effect.new_tensor(0.) for index in chosen]).mean())
                        check_gain=float(torch.stack([actual_effects[index,half:].mean() if index>=0 else effect.new_tensor(0.) for index in chosen]).mean())
                        criterion=-gain+.05*float(error)
                if criterion<best:
                    best=criterion;best_state=copy.deepcopy(net.cpu().state_dict());net.to(device);best_step=update+1
                    selected_gain=gain;independent_gain=check_gain;selected_error=float(error)
                if update+1-best_step>=500:break
        states.append(best_state)
        record=dict(head=head,validation_loss=selected_error,updates=best_step,conditional=conditional,spans=spans,
            credit_objective='paired terminal ranking' if rank_credit and conditional else 'paired terminal likelihood' if conditional else 'ordinary marginal returns',
            selection_gain=selected_gain,independent_future_gain=independent_gain)
        fit.append(record);print(dict(phase='paired_terminal_outcomes_fit',**record),flush=True)
    root_rows=list(roots.values());root_packet=packet([row['packet'] for row in root_rows])
    actions=torch.as_tensor(np.stack([row['left'] for row in root_rows]),device=device)
    context=finite_credit_context(root_packet,actions).flatten(0,1)
    center=context.mean(0);scale=context.std(0).clamp_min(.1);context=(context-center)/scale
    scene_ids=torch.tensor([row['scene'][0] for row in root_rows],device=device).repeat_interleave(actions.shape[1])
    distance=torch.cdist(context,context).square()/context.shape[-1]
    distance=distance.masked_fill(scene_ids[:,None]==scene_ids[None],float('inf'))
    result=dict(states=[],spans=spans,width=width,contexts=context.cpu(),center=center.cpu(),scale=scale.cpu(),
        bandwidth=float(torch.quantile(distance.amin(-1),.9).clamp_min(.05)),outcome_states=states,
        confidence=1.,margin=.01,max_interventions=1,wait_states=[],use_wait=use_wait,conditional=conditional,fit=fit,
        public_observations=True)
    if use_wait:result['wait_states']=fit_finite_wait_value(dataset,result,updates=wait_updates,seed=seed,device=device)
    return result


@torch.enable_grad()
def fit_finite_wait_value(dataset,credit,updates=1500,seed=42,device='cpu'):
    import copy
    import numpy as np
    import develop_source_fusion as d
    rng=np.random.default_rng(seed);width=credit['width']
    heads=len(credit['outcome_states']) if credit.get('outcome_states') else len(credit['states'][0])
    keys=('node','edge','motion','water','task','obs')
    def packet(records):
        return {key:torch.as_tensor(np.stack([record[key] for record in records]),dtype=torch.float32,device=device) for key in keys}
    controller=FiniteTemporalCredit(**dict(credit,wait_states=[])).to(device).requires_grad_(False)
    factual=[]
    for rollout in dataset['rollouts']:
        for index,row in enumerate(rollout['steps']):
            factual.append(dict(packet=row['packet'],action=row['action'][:,:3],
                following=None if index+1==len(rollout['steps']) else len(factual)+1))
    inputs=packet([row['packet'] for row in factual]);actions=torch.as_tensor(np.stack([row['action'] for row in factual]),device=device)
    if credit.get('public_observations'):inputs=public_credit_packet(inputs)
    with torch.no_grad():
        gains=torch.cat([controller.estimates({k:v[start:start+32] for k,v in inputs.items()},actions[start:start+32])[0].flatten(1).amax(1)
            for start in range(0,len(factual),32)])
        # The same intervention cost is charged now and at a future stopping
        # time. Valuing a future intervention before this cost makes waiting
        # artificially preferable even when the opportunity stays constant.
        gains=(gains-controller.margin).clamp(0.,1.)
    live=torch.tensor([row['following'] is not None for row in factual],device=device)
    following=torch.tensor([row['following'] or 0 for row in factual],device=device)
    torch.manual_seed(seed+713)
    waiters=nn.ModuleList([d.JointLearningValue(width=width,relational=True,invariant=True).to(device) for _ in range(heads)])
    for net in waiters:
        nn.init.zeros_(net.value[-1].weight);nn.init.zeros_(net.value[-1].bias)
    target=copy.deepcopy(waiters).requires_grad_(False)
    optimizer=torch.optim.AdamW(waiters.parameters(),lr=3e-4,weight_decay=.01)
    for update in range(updates):
        indices=rng.integers(len(factual),size=128);next_ids=following[indices]
        p={k:v[indices] for k,v in inputs.items()};q={k:v[next_ids] for k,v in inputs.items()}
        with torch.no_grad():
            future=torch.stack([net(q) for net in target]).amin(0).clamp(0.,1.)
            expected=.995*torch.maximum(gains[next_ids],future)*live[indices]
        loss=sum(F.mse_loss(net(p),expected) for net in waiters)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        nn.utils.clip_grad_norm_(waiters.parameters(),2.);optimizer.step()
        with torch.no_grad():
            for destination,source in zip(target.parameters(),waiters.parameters()):destination.lerp_(source,.01)
    return [{key:value.cpu() for key,value in net.state_dict().items()} for net in waiters]


class ConditionalFusionLearner:
    """Update the actual fusion policy from paired physical-action credit.

    Both controls fit the same real transitions and branch endpoints. The
    conditional arm additionally calibrates single-vessel value differences.
    A separate factual critic supplies branch tails, so auxiliary contrasts
    cannot manufacture their own bootstrap targets.
    """
    def __init__(self,policy,conditional=True,device='cpu',seed=42):
        import copy
        import numpy as np
        import develop_source_fusion as d
        torch.manual_seed(seed);self.rng=np.random.default_rng(seed)
        self.device=device;self.conditional=conditional;self.gamma=.995
        self.model=torch.load(policy,map_location='cpu',weights_only=False)
        self.actor=copy.deepcopy(d.joint_load(str(policy),Path(policy).stat().st_mtime_ns)).to(device)
        self.actor.base.requires_grad_(False)
        self.actor.base.a1.requires_grad_(True);self.actor.base.adap_layer.requires_grad_(True)
        self.reference=copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.values=nn.ModuleList([d.JointLearningValue(width=64,relational=True,invariant=True).to(device) for _ in range(2)])
        self.factual=copy.deepcopy(self.values)
        self.target=copy.deepcopy(self.factual).requires_grad_(False)
        self.actor_optimizer=torch.optim.Adam([p for p in self.actor.parameters() if p.requires_grad],lr=3e-5)
        self.value_optimizer=torch.optim.Adam(self.values.parameters(),lr=3e-4)
        self.factual_optimizer=torch.optim.Adam(self.factual.parameters(),lr=3e-4)
        self.replay=[];self.pairs=[];self.updates=0
        self.keys=('node','edge','motion','water','task','obs')

    def packet(self,records):
        import numpy as np
        packet={key:torch.as_tensor(np.stack([record[key] for record in records]),dtype=torch.float32,device=self.device)
            for key in self.keys}
        if self.actor.interaction_enabled:
            missing=[i for i,record in enumerate(records) if '_candidate_risk' not in record]
            if missing:
                with torch.no_grad():
                    for begin in range(0,len(missing),64):
                        ids=missing[begin:begin+64];current={k:v[ids] for k,v in packet.items()}
                        proposal=self.actor.proposal_distribution(current['obs'])[0].tanh()
                        risk=self.actor.candidate_risk(current,proposal).cpu().numpy()
                        for i,value in zip(ids,risk):records[i]['_candidate_risk']=value
            packet['_candidate_risk']=torch.as_tensor(np.stack([record['_candidate_risk'] for record in records]),device=self.device)
        return packet

    def remember(self,rollouts,pairs=()):
        import numpy as np
        for rollout in rollouts:
            steps=rollout['steps'];remaining=0.
            for index in reversed(range(len(steps))):
                record=steps[index];following=steps[index+1]['packet'] if index+1<len(steps) else None
                potential=-float(np.linalg.norm(record['packet']['node'][:,2:4]*30.,axis=-1).min())/60.
                next_potential=0. if following is None else -float(np.linalg.norm(following['node'][:,2:4]*30.,axis=-1).min())/60.
                reward=-.001+(1. if record['done']>2 else -1. if record['done'] else 0.)+.2*(self.gamma*next_potential-potential)
                remaining=reward+self.gamma*remaining
                self.replay.append(dict(packet=record['packet'],action=record['action'][:,:3],reward=reward,
                    next=following,mc=remaining,done=bool(record['done'])))
        self.replay=self.replay[-60000:]
        if pairs:self.pairs=list(pairs)

    @torch.no_grad()
    def tails(self,entries):
        result=torch.tensor([entry['reward'] for entry in entries],device=self.device)
        live=[i for i,entry in enumerate(entries) if entry['next'] is not None]
        if live:
            packet=self.packet([entries[i]['next'] for i in live])
            behavior=getattr(self,'behavior',self.actor)
            action=behavior(packet,deterministic=True,gate_only=True)[0]
            value=torch.stack([critic(packet,action) for critic in self.target]).mean(0)
            discount=torch.tensor([entries[i].get('discount',self.gamma) for i in live],device=self.device)
            result[live]+=discount*value
        return result

    @torch.no_grad()
    def credit_targets(self):
        """Estimate a conditional effect across independent paired futures.

        Each group changes one vessel at one factual state. Repetitions are
        uncertainty samples, not independent labels for memorizing noise.
        """
        import numpy as np
        groups={}
        for pair in self.pairs:
            subject=int(np.abs(pair['right'][:,2]-pair['left'][:,2]).argmax())
            key=(tuple(pair['scene']),pair['root_id'],subject)
            groups.setdefault(key,[]).append(pair)
        entries=list(groups.values());records=[];before=[];after=[];variance=[]
        for begin in range(0,len(entries),16):
            batch=entries[begin:begin+16]
            flat=[pair for group in batch for pair in group]
            left=self.tails([pair['before'] for pair in flat])
            right=self.tails([pair['after'] for pair in flat])
            offset=0
            for group in batch:
                count=len(group);a=left[offset:offset+count];b=right[offset:offset+count]
                records.append(group[0]);before.append(a.mean());after.append(b.mean())
                variance.append((b-a).var(unbiased=count>1)/count if count>1 else a.new_tensor(.04))
                offset+=count
        return records,torch.stack(before),torch.stack(after),torch.stack(variance)

    def supported_projection(self,records,before,after,variance,indices):
        """Project measured single-vessel credit onto the executing Adapter.

        The ordinary control uses fitted joint returns on exactly the same
        supported action pairs. Conditional learning uses their paired effect,
        shrunk by its empirical uncertainty; no unseen-action maximization is
        needed during this policy update.
        """
        import numpy as np
        import develop_source_fusion as d
        rows=[records[int(i)] for i in indices]
        inputs=self.packet([row['packet'] for row in rows])
        left=torch.as_tensor(np.stack([row['left'] for row in rows]),device=self.device)
        right=torch.as_tensor(np.stack([row['right'] for row in rows]),device=self.device)
        subject=(right[...,2]-left[...,2]).abs().argmax(-1)
        batch=torch.arange(len(rows),device=self.device)
        with torch.no_grad():
            reference=self.reference(inputs,deterministic=True,gate_only=True)
            if self.conditional:
                effect=(after-before)[indices]
                uncertainty=variance[indices]
                advantage=effect*effect.square()/(effect.square()+uncertainty+1e-4)
            else:
                advantage=torch.stack([critic(inputs,right)-critic(inputs,left) for critic in self.values]).mean(0)
        decision,_,_,_,scores,_=self.actor(inputs,deterministic=True,gate_only=True)
        if self.actor.interaction_enabled:
            powers=3**torch.arange(left.shape[1],device=self.device)
            li=((left[...,2]*2).round().long()*powers).sum(-1)
            ri=((right[...,2]*2).round().long()*powers).sum(-1)
            prior=reference[4][batch,ri]-reference[4][batch,li]
            target=(prior+advantage/.05).sigmoid().detach()
            logit=scores[batch,ri]-scores[batch,li]
            policy_loss=F.binary_cross_entropy_with_logits(logit,target)
            trust=F.binary_cross_entropy_with_logits(logit,prior.sigmoid())
        else:
            old=reference[0][batch,subject,2]
            change=right[batch,subject,2]-left[batch,subject,2]
            prior=-2.*change.square()
            adjustment=((prior+advantage/.05).sigmoid()-prior.sigmoid())*change
            desired=(old+adjustment).clamp(0.,1.).detach()
            net=self.actor.base;_,_,hidden=self.actor.proposal_distribution(inputs['obs'])
            own=net.activation(net.a1(torch.cat((decision[...,:2],inputs['node'][...,11:13]),-1)))
            native=2.*net.adap_layer(torch.cat((own,hidden),-1)).squeeze(-1)
            policy_loss=F.binary_cross_entropy_with_logits(native[batch,subject],desired)
            trust=F.binary_cross_entropy_with_logits(native[batch,subject],old)
        return policy_loss+.02*trust

    def train(self,updates=500,warmup=False):
        import copy
        import numpy as np
        # Every branch resumed this behavior policy. Keep its bootstrap action
        # fixed throughout the fit, then make a bounded policy improvement.
        self.behavior=copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.reference=self.behavior
        credits=None if warmup or not self.pairs else self.credit_targets()
        metrics=[]
        with torch.enable_grad():
            for update in range(updates):
                selected=[self.replay[i] for i in self.rng.integers(len(self.replay),size=128)]
                packet=self.packet([row['packet'] for row in selected])
                action=torch.as_tensor(np.stack([row['action'] for row in selected]),device=self.device)
                mc=torch.tensor([row['mc'] for row in selected],device=self.device)
                target=mc if warmup else self.tails(selected)
                factual_loss=sum(F.smooth_l1_loss(critic(packet,action),target)+.25*F.smooth_l1_loss(critic(packet,action),mc)
                    for critic in self.factual)
                self.factual_optimizer.zero_grad(set_to_none=True);factual_loss.backward()
                nn.utils.clip_grad_norm_(self.factual.parameters(),2.);self.factual_optimizer.step()
                value_loss=sum(F.smooth_l1_loss(critic(packet,action),target)+.25*F.smooth_l1_loss(critic(packet,action),mc)
                    for critic in self.values)
                contrast=action.new_zeros(())
                if self.pairs and not warmup:
                    if update==updates//2:credits=self.credit_targets()
                    records,expected_before,expected_after,variance=credits
                    indices=self.rng.integers(len(records),size=64)
                    pairs=[records[int(i)] for i in indices]
                    inputs=self.packet([pair['packet'] for pair in pairs])
                    left=torch.as_tensor(np.stack([pair['left'] for pair in pairs]),device=self.device)
                    right=torch.as_tensor(np.stack([pair['right'] for pair in pairs]),device=self.device)
                    before=expected_before[indices];after=expected_after[indices]
                    for critic in self.values:
                        ql=critic(inputs,left);qr=critic(inputs,right)
                        value_loss=value_loss+.25*(F.smooth_l1_loss(ql,before)+F.smooth_l1_loss(qr,after))
                        contrast=contrast+F.mse_loss(qr-ql,after-before)
                    if self.conditional:value_loss=value_loss+4.*contrast
                self.value_optimizer.zero_grad(set_to_none=True);value_loss.backward()
                nn.utils.clip_grad_norm_(self.values.parameters(),2.);self.value_optimizer.step()
                with torch.no_grad():
                    tau=1. if warmup else .01
                    for destination,source in zip(self.target.parameters(),self.factual.parameters()):
                        destination.lerp_(source,tau)
                policy_loss=action.new_zeros(())
                if not warmup and credits is not None and update>=max(0,updates-200) and (update+1)%2==0:
                    indices=self.rng.integers(len(credits[0]),size=32)
                    policy_loss=self.supported_projection(*credits,indices)
                    self.actor_optimizer.zero_grad(set_to_none=True);policy_loss.backward()
                    nn.utils.clip_grad_norm_(self.actor.parameters(),.5);self.actor_optimizer.step()
                self.updates+=1
                if (update+1)%100==0:
                    row=dict(update=self.updates,factual_loss=float(factual_loss.detach()),value_loss=float(value_loss.detach()),
                        contrast_loss=float(contrast.detach()),policy_loss=float(policy_loss.detach()))
                    metrics.append(row);print(dict(phase='conditional_fusion_policy_learning',conditional=self.conditional,**row),flush=True)
        return metrics

    def policy(self):
        return dict(self.model,actor={k:v.detach().cpu() for k,v in self.actor.state_dict().items()},
            conditional_training=self.conditional,fusion_learning='paired-conditional-soft-policy-v1',
            fusion_optimizer_updates=self.updates,fusion_improvement='supported-conditional-credit-projection')


class FactualContinuationValue(nn.Module):
    """Terminal outcome value of a public state under one frozen policy.

    This critic supplies training targets, never an online trajectory search.
    Its target/asset frame and symmetric pooling support different fleet sizes.
    """
    def __init__(self,width=64):
        super().__init__()
        self.node=nn.Sequential(nn.Linear(19,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.pair=nn.Sequential(nn.Linear(10,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.value=nn.Sequential(nn.Linear(4*width+8,width),nn.SiLU(),nn.Linear(width,4))

    def forward(self,packet,action):
        public=packet['node'];motion=packet['motion'];b,n=motion.shape[:2]
        c=motion[:,0,2].cos();s=motion[:,0,2].sin()
        def world(vector):
            return torch.stack((c*vector[:,0]-s*vector[:,1],s*vector[:,0]+c*vector[:,1]),-1)
        target=motion[:,0,:2]+world(public[:,0,2:4]*30.)
        velocity=world(public[:,0,4:6]*5.)
        axis=-target/target.norm(dim=-1,keepdim=True).clamp_min(1e-6)
        def common(vector):
            basis=axis.reshape(b,*([1]*(vector.ndim-2)),2)
            return torch.stack(((vector*basis).sum(-1),basis[...,0]*vector[...,1]-basis[...,1]*vector[...,0]),-1)
        relative=motion[...,:2]-target[:,None];relative_velocity=motion[...,3:5]-velocity[:,None]
        distance=relative.norm(dim=-1).clamp_min(1e-4)
        approach=(relative*relative_velocity).sum(-1)/distance
        heading=torch.stack((motion[...,2].cos(),motion[...,2].sin()),-1)
        controls=action[...,2:3]*action[...,:2]+(1.-action[...,2:3])*public[...,11:13]
        state=torch.cat((common(relative)/30.,common(relative_velocity)/5.,common(heading),motion[...,5:6],
            action[...,:3],public[...,11:13],controls,distance[...,None]/30.,approach[...,None]/5.,
            public[...,14:16],motion.new_full((b,n,1),n/6.)),-1).clamp(-20.,20.)
        displacement=motion[:,None,:,:2]-motion[:,:,None,:2]
        dv=motion[:,None,:,3:5]-motion[:,:,None,3:5]
        separation=displacement.norm(dim=-1).clamp_min(1e-4)
        closing=(displacement*dv).sum(-1)/separation
        collision_time=(-((displacement*dv).sum(-1))/(dv.square().sum(-1)+.1)).clamp(0.,6.)
        closest=(displacement+collision_time[...,None]*dv).norm(dim=-1)
        pair=torch.cat((common(displacement)/15.,common(dv)/5.,separation[...,None]/15.,
            closing[...,None]/5.,collision_time[...,None]/6.,closest[...,None]/15.,
            (action[:,:,None,2]-action[:,None,:,2])[...,None],
            (heading[:,:,None]*heading[:,None]).sum(-1,keepdim=True)),-1).clamp(-20.,20.)
        mask=~torch.eye(n,dtype=torch.bool,device=motion.device)[None]
        nodes=self.node(state);pairs=self.pair(pair)
        global_state=torch.cat((target.norm(dim=-1,keepdim=True)/30.,common(velocity)/5.,
            distance.amin(-1,keepdim=True)/30.,separation.masked_fill(~mask,1e6).amin((1,2))[:,None]/15.,
            closest.masked_fill(~mask,1e6).amin((1,2))[:,None]/15.,
            public[:,0,15:16],motion.new_full((b,1),n/6.)),-1)
        pooled=torch.cat((nodes.mean(1),nodes.amax(1),(pairs*mask[...,None]).sum((1,2))/max(1,n*(n-1)),
            pairs.masked_fill(~mask[...,None],-1e6).amax((1,2)),global_state),-1)
        return self.value(pooled)


def fit_factual_continuation_value(datasets,seed=42,width=64,folds=5,updates=2400,device='cpu'):
    """Cross-fit dense factual terminal labels by whole initial layout."""
    import numpy as np
    records=[];seen=set()
    for dataset in datasets:
        for rollout in dataset['rollouts']:
            row=rollout['row'];steps=rollout['steps']
            identity=(row['seed'],row.get('future_seed'),row.get('root_id'),row.get('root_step'))
            if identity in seen:continue
            seen.add(identity)
            terminal=0 if row['capture'] else 1 if row['collision'] else 3 if row['timeout'] else 2
            for step in steps[::2]:
                records.append((row['seed'],step['packet'],step['action'][:,:3],terminal))
    packet={key:torch.as_tensor(np.stack([row[1][key] for row in records]),dtype=torch.float32,device=device)
        for key in ('node','motion')}
    action=torch.as_tensor(np.stack([row[2] for row in records]),dtype=torch.float32,device=device)
    target=torch.tensor([row[3] for row in records],device=device)
    layouts=np.asarray([row[0] for row in records]);unique=np.unique(layouts)
    rng=np.random.default_rng(seed);ordered=rng.permutation(unique)
    layout_fold={int(layout):i%folds for i,layout in enumerate(ordered)}
    assignment=np.asarray([layout_fold[int(layout)] for layout in layouts])
    states=[];fits=[]
    with torch.enable_grad():
        for fold in range(folds):
            torch.manual_seed(seed+7919*fold);rng=np.random.default_rng(seed+7919*fold)
            training=np.flatnonzero(assignment!=fold);validation=torch.tensor(np.flatnonzero(assignment==fold),device=device)
            net=FactualContinuationValue(width).to(device)
            opt=torch.optim.AdamW(net.parameters(),lr=5e-4,weight_decay=.01)
            best=float('inf');best_state=None;selected=0
            for update in range(updates):
                ids=torch.tensor(training[rng.integers(len(training),size=256)],device=device)
                loss=nn.functional.cross_entropy(net({k:v[ids] for k,v in packet.items()},action[ids]),target[ids])
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);opt.step()
                if (update+1)%100:continue
                with torch.no_grad():
                    logits=torch.cat([net({k:v[ids] for k,v in packet.items()},action[ids]) for ids in validation.split(1024)])
                    probability=logits.softmax(-1);success=probability[:,0]+probability[:,3]
                    observed=((target[validation]==0)|(target[validation]==3)).float()
                    score=float((success-observed).square().mean())
                if score<best:
                    best=score;selected=update+1;best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                if update+1-selected>=700:break
            states.append(best_state);fits.append(dict(fold=fold,states=len(validation),success_brier=best,selected_update=selected))
            print(dict(phase='factual_continuation_fit',**fits[-1]),flush=True)
    return dict(states=states,width=width,layout_fold=layout_fold,fits=fits,training_states=len(records),
        use='Cross-fitted factual terminal outcome value; each branch uses a model trained without its initial layout.')


def local_role_features(packet, action, programs, subjects, spans, physical_features=False,
                        threat_features=False,prediction_features=False):
    """Public encounter geometry centered on the boat whose role changes."""
    motion=packet['motion'];node=packet['node'];batch,n=action.shape[:2]
    rows=torch.arange(batch,device=action.device);heading=motion[...,2]
    c,s=heading.cos(),heading.sin()
    local=node[:,0,2:4]*30.
    target=motion[:,0,:2]+torch.stack((c[:,0]*local[:,0]-s[:,0]*local[:,1],
        s[:,0]*local[:,0]+c[:,0]*local[:,1]),-1)
    velocity=node[:,0,4:6]*5.
    velocity=torch.stack((c[:,0]*velocity[:,0]-s[:,0]*velocity[:,1],
        s[:,0]*velocity[:,0]+c[:,0]*velocity[:,1]),-1)
    axis=-target/target.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    def rotate(vector):
        a=axis[:,None] if vector.ndim==3 else axis
        return torch.stack(((a*vector).sum(-1),a[...,0]*vector[...,1]-a[...,1]*vector[...,0]),-1)
    relative=rotate(motion[...,:2]-target[:,None])/30.
    speed=rotate(motion[...,3:5]-velocity[:,None])/5.
    direction=rotate(torch.stack((c,s),-1))
    gate=torch.where(programs>=0.,programs,action[...,2])
    physical=gate[...,None]*action[...,:2]+(1.-gate[...,None])*node[...,11:13]
    vessels=torch.cat((relative,speed,direction,motion[...,5:6],action[...,:3],
        node[...,11:13],(programs>=0.).to(action.dtype)[...,None],gate[...,None],physical),-1)
    own=vessels[rows,subjects]
    displacement=motion[...,:2]-motion[rows,subjects,None,:2]
    dv=motion[...,3:5]-motion[rows,subjects,None,3:5]
    distance=displacement.norm(dim=-1)
    time=(-(displacement*dv).sum(-1)/dv.square().sum(-1).clamp_min(.01)).clamp(0.,10.)
    miss=(displacement+time[...,None]*dv).norm(dim=-1)
    distance=distance.masked_fill(F.one_hot(subjects,num_classes=n).bool(),1e5)
    nearest=distance.argsort(-1)[:,:2]
    neighbors=torch.cat((rotate(displacement)/15.,rotate(dv)/5.,direction,
        action[...,2:3],(programs>=0.).to(action.dtype)[...,None],gate[...,None],physical,
        distance[...,None]/15.,time[...,None]/10.,miss[...,None]/15.),-1)
    if threat_features:
        # Euclidean proximity alone loses a fast closing partner behind a
        # nearer, separating boat. Retain both neighborhoods in the critic.
        # These are measured relative-motion features, not candidate rollouts.
        separation=displacement.norm(dim=-1).clamp_min(1.)
        closing=-(displacement*dv).sum(-1)/separation
        urgency=(miss-5.).clamp_min(0.)+.2*time+.1*separation
        urgency=urgency.masked_fill(F.one_hot(subjects,num_classes=n).bool(),1e5)
        threatening=urgency.argsort(-1)[:,:2]
        threat_nodes=torch.cat((neighbors,closing[...,None]/5.,
            ((displacement[...,0]*dv[...,1]-displacement[...,1]*dv[...,0])/separation)[...,None]/5.),-1)
        threat_nodes=threat_nodes.gather(1,threatening[...,None].expand(-1,-1,threat_nodes.shape[-1])).flatten(1)
    neighbors=neighbors.gather(1,nearest[...,None].expand(-1,-1,neighbors.shape[-1])).flatten(1)
    close=relative.norm(dim=-1).sort(-1).values[:,:2]
    rank=(relative.norm(dim=-1)<relative[rows,subjects].norm(dim=-1)[:,None]).sum(-1)/max(n-1,1)
    global_state=torch.cat((target.norm(dim=-1,keepdim=True)/60.,rotate(velocity)/5.,
        node[:,0,15:16],spans[:,None]/50.,rank[:,None],close,
        action.new_full((batch,1),n/6.)),1)
    features=torch.cat((own,neighbors,vessels.mean(1),vessels.amax(1),global_state),1)
    if threat_features:
        features=torch.cat((features,threat_nodes),1)
    if physical_features:
        # Instantaneous actuator effects, not M1 candidate trajectories. Ground
        # motion is measured; hidden water-relative velocity is not an input.
        vx,vy,omega=motion[...,3],motion[...,4],motion[...,5]
        surge=c*vx+s*vy;sway=-s*vx+c*vy
        force=750.*physical+250.
        du=(400.*omega*sway-(100.+150.*surge.abs())*surge+force.sum(-1))/380.
        dv=(-380.*omega*surge-(100.+150.*sway.abs())*sway)/400.
        dr=(-20.*surge*sway-(980.+950.*omega.abs())*omega+1.25*(force[...,0]-force[...,1]))/1430.
        acceleration=torch.stack((c*du-s*dv-omega*vy,s*du+c*dv+omega*vx),-1)
        radial=relative/relative.norm(dim=-1,keepdim=True).clamp_min(1e-4)
        oriented=rotate(acceleration)/10.
        dynamics=torch.cat((oriented,dr[...,None]/5.,(oriented*radial).sum(-1,keepdim=True),
            (dr*(direction[...,0]*radial[...,1]-direction[...,1]*radial[...,0]))[...,None]/5.),-1)
        unit=displacement/displacement.norm(dim=-1,keepdim=True).clamp_min(1.)
        relative_acceleration=(acceleration-acceleration[rows,subjects,None])/10.
        effects=torch.stack(((unit*relative_acceleration).sum(-1),
            unit[...,0]*relative_acceleration[...,1]-unit[...,1]*relative_acceleration[...,0],
            (dr-dr[rows,subjects,None])/5.),-1)
        effects=effects.gather(1,nearest[...,None].expand(-1,-1,3)).flatten(1)
        features=torch.cat((features,dynamics[rows,subjects],effects,dynamics.mean(1),dynamics.amax(1)),1)
    if prediction_features:
        forecast=features.new_zeros((batch,10));risk=packet.get('_joint_candidate_risk')
        if risk is not None and risk.ndim==5:
            k=risk.shape[-1];digits=(gate*(k-1)).round().long()
            pair=torch.stack([risk[rows,i,j,digits[:,i],digits[:,j]]
                for i in range(n) for j in range(n)],-1).reshape(batch,n,n)
            pair=pair.masked_fill(torch.eye(n,dtype=torch.bool,device=action.device)[None],0.)
            own_risk=pair[rows,subjects]
            forecast[:,:5]=torch.stack((own_risk.amax(-1),own_risk.mean(-1),
                pair.amax((1,2)),pair.mean((1,2)),own_risk.topk(min(2,n),dim=-1).values[:,-1]),-1).clamp_max(200.)/25.
            interception=packet.get('_candidate_interception')
            if interception is not None:
                reach=interception.gather(2,digits[...,None]).squeeze(-1)/30.
                forecast[:,5:8]=torch.stack((reach[rows,subjects],reach.amin(-1),reach.mean(-1)),-1)
            forecast[:,8:]=torch.stack((gate[rows,subjects],(gate-action[...,2]).abs().sum(-1)),1)
        features=torch.cat((features,forecast),1)
    return features.clamp(-20.,20.)


class LocalRoleEffectCredit(nn.Module):
    """Antisymmetric conditional effects with local empirical support.

    Every learned edge changes one boat while all other commanded roles stay
    fixed. A two-boat program averages both connected paths from free control;
    it is not scored by summing unconditional single-boat effects.
    """
    def __init__(self,forests,contexts,center,scale,bandwidth,spans,confidence=.5,margin=.01,
                 max_changed=2,initial_only=True,temporal=False,physical_features=False,max_interventions=1,
                 wait_span=False,fit=None,estimator='local-conditional-effects',threat_features=False,
                 risk_tolerance=None,full_grid=False,prediction_features=False,support_weights=None):
        super().__init__();self.forests=forests;self.spans=tuple(spans)
        self.confidence=confidence;self.margin=margin;self.bandwidth=bandwidth
        self.max_changed=max_changed;self.initial_only=initial_only
        self.temporal=temporal
        self.physical_features=physical_features
        self.threat_features=threat_features
        self.prediction_features=prediction_features
        self.risk_tolerance=risk_tolerance
        self.full_grid=full_grid
        self.response_features=False;self.wait_span=wait_span;self.max_interventions=max_interventions
        self.register_buffer('contexts',torch.as_tensor(contexts,dtype=torch.float32))
        self.register_buffer('center',torch.as_tensor(center,dtype=torch.float32))
        self.register_buffer('scale',torch.as_tensor(scale,dtype=torch.float32))
        # A constant training coordinate cannot identify a distance penalty.
        # In particular, a fixed training fleet size must not become an
        # artificial rejection criterion for a variable-size local critic.
        self.register_buffer('support_weights',torch.ones_like(self.scale) if support_weights is None else
            torch.as_tensor(support_weights,dtype=torch.float32))

    def eligible(self,packet):
        count=packet.get('_conditional_count',packet['node'].new_zeros((len(packet['node']),1)))[:,0]
        use=count<self.max_interventions if self.max_interventions else torch.ones_like(count,dtype=torch.bool)
        return use&(packet['node'][:,0,15].abs()<1e-7) if self.initial_only else use

    def programs(self,packet,action):
        batch,n=packet['node'].shape[:2];node=packet['node']
        if self.temporal:
            if self.full_grid:
                roles=node.new_full((batch,1+3*n,n),-1.)
                for i in range(n):
                    for j,gate in enumerate((0.,.5,1.)):roles[:,1+3*i+j,i]=gate
                return roles
            roles=node.new_full((batch,n+1,n),-1.)
            for i in range(n):roles[:,i+1,i]=(action[:,i,2]<=.5).to(action.dtype)
            return roles
        roles=node.new_full((batch,1+3*n+9,n),-1.)
        for i in range(n):
            for j,gate in enumerate((0.,.5,1.)):roles[:,1+3*i+j,i]=gate
        leaders=node[:,:,2:4].norm(dim=-1).argsort(-1)[:,:2]
        rows=torch.arange(batch,device=node.device)
        for i in range(3):
            for j in range(3):
                roles[rows,1+3*n+3*i+j,leaders[:,0]]=i*.5
                roles[rows,1+3*n+3*i+j,leaders[:,1]]=j*.5
        return roles

    def edge_values(self,packet,action,left,right,subject,duration):
        before=local_role_features(packet,action,left,subject,duration,self.physical_features,self.threat_features,self.prediction_features)
        after=local_role_features(packet,action,right,subject,duration,self.physical_features,self.threat_features,self.prediction_features)
        forward=torch.cat((before,after-before),1)
        reverse=torch.cat((after,before-after),1)
        import numpy as np
        x=forward.detach().cpu().numpy();y=reverse.detach().cpu().numpy()
        estimates=action.new_tensor(np.stack([.5*(forest.predict(x)-forest.predict(y)) for forest in self.forests]))
        if self.bandwidth>0.:
            query=(torch.cat((forward,reverse))-self.center)/self.scale
            distance=torch.cdist(query*self.support_weights,self.contexts*self.support_weights).square().amin(-1)/query.shape[-1]
            distance=distance.reshape(2,len(forward)).mean(0)
            estimates*=torch.exp(-.5*distance/self.bandwidth)[None]
        return estimates

    def estimates(self,packet,action):
        roles=self.programs(packet,action);batch,options,n=roles.shape;values=[]
        for duration in self.spans:
            edges=[];assignments=[]
            for b in range(batch):
                for option in range(1,options):
                    if int((roles[b,option]>=0.).sum())>self.max_changed:continue
                    changed=(roles[b,option]>=0.).nonzero().flatten().tolist()
                    if len(changed)==1:
                        edges.append((b,roles[b,0],roles[b,option],changed[0]));assignments.append((b,option,1.))
                    else:
                        for first,second in (changed,changed[::-1]):
                            middle=roles[b,0].clone();middle[first]=roles[b,option,first]
                            edges.extend(((b,roles[b,0],middle,first),(b,middle,roles[b,option],second)))
                            assignments.extend(((b,option,.5),(b,option,.5)))
            index=torch.tensor([edge[0] for edge in edges],device=action.device)
            prediction=self.edge_values({key:value[index] for key,value in packet.items()},action[index],
                torch.stack([edge[1] for edge in edges]),torch.stack([edge[2] for edge in edges]),
                torch.tensor([edge[3] for edge in edges],device=action.device),action.new_full((len(edges),),duration))
            value=action.new_zeros((len(self.forests),batch*options))
            locations=torch.tensor([b*options+option for b,option,_ in assignments],device=action.device)
            weights=action.new_tensor([weight for _,_,weight in assignments])
            value.scatter_add_(1,locations[None].expand(len(self.forests),-1),prediction*weights[None])
            values.append(value.reshape(len(self.forests),batch,options))
        return torch.stack(values,2),roles

    def forward(self,packet,action):
        roles=action.new_full(action.shape[:2],-1.);spans=action.new_ones(len(action))
        eligible=self.eligible(packet)
        if bool(eligible.any()):
            index=eligible.nonzero().flatten();current={key:value[index] for key,value in packet.items()}
            values,programs=self.estimates(current,action[index,:,:3])
            score=values.mean(0)-self.confidence*values.std(0,unbiased=False)
            risk=current.get('_joint_candidate_risk')
            if self.risk_tolerance is not None and risk is not None:
                # M2 may change the role assignment only within the encounter
                # support computed by M1. Removing M1 removes this support too;
                # the credit learner does not contain a replacement predictor.
                levels=risk.shape[-1];n=programs.shape[-1]
                gates=torch.where(programs>=0.,programs,action[index,None,:,2])
                digits=(gates*(levels-1)).round().long()
                pair_cost=torch.stack([risk[:,i,j].flatten(1).gather(1,digits[:,:,i]*levels+digits[:,:,j])
                    for i in range(n) for j in range(i+1,n)],-1)
                worst=pair_cost.amax(-1);total=pair_cost.sum(-1)
                permitted=(worst<=worst[:,:1]+self.risk_tolerance)&(
                    total<=total[:,:1]+self.risk_tolerance*max(1,n-1))
                score=score.masked_fill(~permitted[:,None],-torch.inf)
            gain,choice=score.flatten(1).max(-1)
            selected=programs[torch.arange(len(index),device=action.device),choice%programs.shape[1]]
            roles[index]=torch.where((gain>self.margin)[:,None],selected,-torch.ones_like(selected))
            spans[index]=action.new_tensor(self.spans)[choice//programs.shape[1]]
        return torch.cat((action[...,:3],roles[...,None],spans[:,None,None].expand(-1,action.shape[1],1),roles[...,None]),-1)


class DiscordanceEffectForest:
    """Separate intervention relevance from the direction of a changed outcome."""
    def __init__(self,incidence,direction):
        self.incidence=incidence;self.direction=direction

    def predict(self,x):
        return self.incidence.predict(x)*(2.*self.direction.predict(x)-1.)


class PairedKernelEffect:
    """A smooth conditional sensitivity field over observed action changes.

    The midpoint context is unchanged when an intervention is reversed;
    its signed action displacement reverses the prediction exactly. Zero
    intervention has zero value without learning a large factual intercept.
    """
    def __init__(self,bandwidth=1.,ridge=.05):
        self.bandwidth=bandwidth;self.ridge=ridge

    def features(self,x):
        import numpy as np
        half=x.shape[1]//2
        context=(x[:,:half]+.5*x[:,half:]-self.center)/self.scale
        change=x[:,half:][:,self.active]/self.change_scale
        return context.astype(np.float64),change.astype(np.float64)

    def covariance(self,context,change):
        import numpy as np
        distance=(np.square(context).sum(1)[:,None]+self.norm[None]-2.*context@self.context.T).clip(0.)
        similarity=np.exp(-.5*distance/(context.shape[1]*self.bandwidth**2))
        return similarity*(change@self.change.T)/max(1,change.shape[1])

    def fit(self,x,y,sample_weight=None):
        import numpy as np
        from scipy.linalg import cho_factor,cho_solve
        half=x.shape[1]//2;midpoint=x[:,:half]+.5*x[:,half:]
        self.center=midpoint.mean(0);self.scale=midpoint.std(0).clip(.05)
        self.active=np.abs(x[:,half:]).max(0)>1e-6
        self.change_scale=x[:,half:][:,self.active].std(0).clip(.05)
        context,change=self.features(x)
        first=np.abs(change).argmax(1);orientation=np.sign(change[np.arange(len(x)),first])
        orientation[orientation==0.]=1.
        # Both directions and bootstrap duplicates are one observed contrast.
        unique,inverse=np.unique(np.concatenate((context,orientation[:,None]*change),1),axis=0,return_inverse=True)
        weights=np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight)
        mass=np.bincount(inverse,weights=weights)
        target=np.bincount(inverse,weights=weights*y*orientation)/mass
        self.context=unique[:,:half];self.change=unique[:,half:]
        self.norm=np.square(self.context).sum(1)
        kernel=self.covariance(self.context,self.change)
        kernel.flat[::len(kernel)+1]+=self.ridge/(mass/mass.mean()).clip(.25,4.)
        self.coefficient=cho_solve(cho_factor(kernel,lower=True,check_finite=False),target,check_finite=False)
        return self

    def predict(self,x):
        context,change=self.features(x)
        return self.covariance(context,change)@self.coefficient


class IntegrableOutcomeEffect:
    """A shared outcome surface whose differences are fitted on paired futures.

    The same value function evaluates both ends of every intervention. This
    makes conditional effects consistent around action cycles, while observed
    terminal outcomes anchor their scale. Setting conditional_weight to zero
    is the matched ordinary-return control, with identical inputs and budget.
    """
    def __init__(self,conditional_weight=4.,seed=42,width=96,updates=1200):
        self.conditional_weight=conditional_weight;self.seed=seed
        self.width=width;self.updates=updates

    def fit(self,x,y,sample_weight=None,before_target=None,after_target=None):
        import numpy as np
        if before_target is None or after_target is None:
            raise ValueError('Outcome anchors must come from the same paired continuations.')
        half=x.shape[1]//2;left=x[:,:half];right=left+x[:,half:]
        points=np.concatenate((left,right));self.center=points.mean(0)
        self.scale=points.std(0).clip(.05)
        self.active=points.std(0)>1e-5
        # Training only at six vessels leaves fleet-count and some duration
        # coordinates constant. Do not let untrained random coefficients on
        # those coordinates determine frozen unseen-fleet decisions.
        a=torch.as_tensor(((left-self.center)/self.scale)*self.active,dtype=torch.float32)
        b=torch.as_tensor(((right-self.center)/self.scale)*self.active,dtype=torch.float32)
        target=torch.as_tensor(y,dtype=torch.float32)
        qa=torch.as_tensor(before_target,dtype=torch.float32)
        qb=torch.as_tensor(after_target,dtype=torch.float32)
        weights=torch.ones(len(x)) if sample_weight is None else torch.as_tensor(sample_weight,dtype=torch.float32)
        weights=weights/weights.mean();rng=np.random.default_rng(self.seed)
        with torch.random.fork_rng(devices=[]),torch.enable_grad():
            torch.manual_seed(self.seed)
            self.network=nn.Sequential(nn.Linear(half,self.width),nn.SiLU(),
                nn.Linear(self.width,self.width),nn.SiLU(),nn.Linear(self.width,1),nn.Sigmoid())
            optimizer=torch.optim.AdamW(self.network.parameters(),lr=3e-4,weight_decay=1e-3)
            for _ in range(self.updates):
                indices=torch.as_tensor(rng.integers(len(x),size=256))
                pa=self.network(a[indices]).squeeze(-1);pb=self.network(b[indices]).squeeze(-1)
                ordinary=.5*((pa-qa[indices]).square()+(pb-qb[indices]).square())
                paired=((pb-pa)-target[indices]).square()
                loss=((ordinary+self.conditional_weight*paired)*weights[indices]).mean()
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(),2.);optimizer.step()
        self.network.eval().requires_grad_(False)
        return self

    @torch.no_grad()
    def predict(self,x):
        half=x.shape[1]//2;left=x[:,:half];right=left+x[:,half:]
        active=getattr(self,'active',1.)
        a=torch.as_tensor(((left-self.center)/self.scale)*active,dtype=torch.float32)
        b=torch.as_tensor(((right-self.center)/self.scale)*active,dtype=torch.float32)
        return (self.network(b)-self.network(a)).squeeze(-1).numpy()


def fit_local_role_effects(datasets,seed=42,heads=5,leaf=12,trees=96,discordance=False,physical_features=False,
                           threat_features=False,uniform_effect_weight=False,boosted=False,return_horizon=0.,
                           kernel_bandwidth=0.,conditional_outcome_weight=None,failure_horizon=0.,prediction_features=False,
                           failure_deadline=0.,future_bootstrap=False):
    """Bootstrap whole layouts and paired futures, not correlated branches."""
    import numpy as np
    from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
    records=[]
    for data in datasets:
        span=data['model']['physical_intervention_span']
        for roll in data['rollouts']:
            groups={}
            for branch in roll['branches']:
                if branch.get('observed_future',False):continue
                left=tuple(float(x) for x in branch['left'][:,3]);right=tuple(float(x) for x in branch['right'][:,3])
                subject=np.flatnonzero(np.array(left)!=np.array(right))
                if len(subject)!=1:raise ValueError('Conditional role edge must change exactly one boat.')
                key=(left,right,int(subject[0]))
                if key not in groups:groups[key]=dict(branch=branch,values={},success={},before={},after={})
                known=all(branch[side]['success'] is not None for side in ('before','after'))
                if not known and not failure_deadline:
                    raise ValueError('Unobserved terminal outcomes require an explicit finite failure estimand.')
                success=float(branch['after']['success']-branch['before']['success']) if known else np.nan
                groups[key]['success'][branch['future_repetition']]=success
                for side in ('before','after'):
                    outcome=branch[side]['success']
                    groups[key][side][branch['future_repetition']]=float(outcome) if outcome is not None else np.nan
                if return_horizon or failure_horizon or failure_deadline:
                    def terminal_value(outcome):
                        duration=.2*outcome['physical_steps'] if 'physical_steps' in outcome else outcome['duration']
                        if failure_deadline:
                            if not outcome.get('terminal') and duration+1e-6<failure_deadline:
                                raise ValueError('A finite failure target needs observations through its entire deadline.')
                            return -float(outcome.get('terminal') in (1,2) and duration<=failure_deadline+1e-6)
                        if failure_horizon:
                            return -(1.-outcome['success'])*np.exp(-duration/failure_horizon)
                        return (2.*outcome['success']-1.)*np.exp(-duration/return_horizon)
                    value=terminal_value(branch['after'])-terminal_value(branch['before'])
                else:value=success
                groups[key]['values'][branch['future_repetition']]=float(value)
            for (left,right,subject),record in groups.items():
                branch=record['branch'];packet={key:torch.as_tensor(value[None]) for key,value in branch['packet'].items()}
                action=torch.as_tensor(branch['reference_action'][None,:,:3])
                before=local_role_features(packet,action,action.new_tensor([left]),torch.tensor([subject]),action.new_tensor([span]),physical_features,threat_features,prediction_features)
                after=local_role_features(packet,action,action.new_tensor([right]),torch.tensor([subject]),action.new_tensor([span]),physical_features,threat_features,prediction_features)
                records.append(dict(scene=roll['row']['seed'],root=hashlib.sha256(branch['packet']['node'].tobytes()).hexdigest(),
                    factual=all(g<0. for g in left),x=torch.cat((before,after-before),1)[0].numpy(),
                    reverse=torch.cat((after,before-after),1)[0].numpy(),
                    future_ids=tuple(sorted(record['values'])),
                    values=np.array([record['values'][key] for key in sorted(record['values'])]),
                    before=np.array([record['before'][key] for key in sorted(record['values'])]),
                    after=np.array([record['after'][key] for key in sorted(record['values'])]),
                    success=np.array([record['success'][key] for key in sorted(record['values'])])))
    layouts=np.array([row['scene'] for row in records]);unique=np.unique(layouts)
    if len(unique)<8:raise ValueError('Local effect fitting requires at least eight independent layouts.')
    x=np.array([row['x'] for row in records]);reverse=np.array([row['reverse'] for row in records])
    # The effect estimator learns both directions of every conditional edge.
    # Its support distribution must contain those same two directions: a
    # reversed, observed intervention is not an out-of-distribution action.
    support_inputs=np.concatenate((x,reverse));support_layouts=np.tile(layouts,2)
    center=support_inputs.mean(0);scale=support_inputs.std(0).clip(.05)
    normalized=(support_inputs-center)/scale
    # Estimate the radius against another whole layout in either direction.
    support=[]
    for scene in unique:
        query=torch.as_tensor(normalized[support_layouts==scene]);bank=torch.as_tensor(normalized[support_layouts!=scene])
        support.extend((torch.cdist(query,bank).square().amin(-1)/x.shape[1]).tolist())
    bandwidth=max(float(np.median(support)),.02);forests=[];fits=[]
    for head in range(heads):
        rng=np.random.default_rng(seed+head*7919);sampled=rng.choice(unique,len(unique),replace=True)
        training=np.concatenate([np.flatnonzero(layouts==scene) for scene in sampled])
        validation=np.flatnonzero(~np.isin(layouts,sampled))
        if not len(validation):raise ValueError('No held-out layout in local conditional bootstrap.')
        draws={scene:rng.integers(8,size=8) for scene in np.unique(sampled)}
        future_weights={}
        if future_bootstrap:
            # One weight per observed future at the same physical root is
            # shared by every action contrast. Unlike a fixed eight-draw
            # index, this covers all observations after adaptive replication.
            keys=sorted({(records[index]['scene'],records[index]['root'],future)
                for index in training for future in records[index]['future_ids']})
            future_weights=dict(zip(keys,rng.exponential(size=len(keys))))
        target=[];weight=[];incidence=[];direction=[];discordant=[];before_target=[];after_target=[]
        for index in training:
            values=records[index]['values'];indices=draws[layouts[index]]%len(values)
            sample=values[indices];changed=np.count_nonzero(sample)
            if future_bootstrap:
                record=records[index]
                weights=np.array([future_weights[(record['scene'],record['root'],future)] for future in record['future_ids']])
                weights=weights/weights.sum()
                target.append(np.sum(values*weights))
                before_target.append(np.sum(record['before']*weights))
                after_target.append(np.sum(record['after']*weights))
                sample=values
                changed=float(np.sum(weights*(values!=0))*len(values))
            else:
                weights=np.full(len(sample),1./len(sample))
                target.append(sample.mean())
                before_target.append(records[index]['before'][indices].mean())
                after_target.append(records[index]['after'][indices].mean())
            # Outcome-dependent precision weights preferentially suppress
            # interventions that change the terminal event. Equal root weight
            # estimates the population conditional effect; the paired-future
            # bootstrap still measures its sampling uncertainty.
            weight.append(1. if uniform_effect_weight else 1./(.05+values.var(ddof=1)/len(values)))
            incidence.append(changed/len(sample));direction.append((np.sum(weights*(sample>0))*len(sample)+.5)/(changed+1.))
            discordant.append(changed)
        target=np.array(target);weight=np.array(weight)
        def new_forest():
            if conditional_outcome_weight is not None:
                return IntegrableOutcomeEffect(conditional_weight=conditional_outcome_weight,seed=seed+head*7919)
            if kernel_bandwidth:
                return PairedKernelEffect(bandwidth=kernel_bandwidth)
            if boosted:
                return HistGradientBoostingRegressor(max_iter=trees,learning_rate=.05,
                    max_leaf_nodes=15,min_samples_leaf=leaf,l2_regularization=2.,early_stopping=False,
                    random_state=seed+head*7919)
            return ExtraTreesRegressor(n_estimators=trees,min_samples_leaf=leaf,max_features=.8,
                max_depth=16,n_jobs=1,random_state=seed+head*7919)
        inputs=np.concatenate((x[training],reverse[training]))
        if conditional_outcome_weight is not None:
            if return_horizon or failure_horizon or failure_deadline or discordance or kernel_bandwidth:
                raise ValueError('Anchored outcome differences require their matching binary outcome targets.')
            forest=new_forest()
            forest.fit(inputs,np.concatenate((target,-target)),sample_weight=np.concatenate((weight,weight)),
                before_target=np.concatenate((before_target,after_target)),
                after_target=np.concatenate((after_target,before_target)))
        elif discordance:
            occurrence=new_forest();orientation=new_forest()
            occurrence.fit(inputs,np.tile(incidence,2))
            orientation_weight=np.tile(discordant,2);used=orientation_weight>0
            orientation.fit(inputs[used],np.concatenate((direction,1.-np.array(direction)))[used],
                sample_weight=orientation_weight[used])
            forest=DiscordanceEffectForest(occurrence,orientation)
        else:
            forest=new_forest()
            forest.fit(inputs,np.concatenate((target,-target)),sample_weight=np.concatenate((weight,weight)))
        prediction=.5*(forest.predict(x[validation])-forest.predict(reverse[validation]))
        truth=np.array([records[index]['values'].mean() for index in validation])
        success_truth=np.array([records[index]['success'].mean() for index in validation])
        chosen={}
        for row,predicted,actual,value_truth in zip(validation,prediction,success_truth,truth):
            key=(records[row]['scene'],records[row]['root'])
            if records[row]['factual']:
                if key not in chosen:chosen[key]=(0.,0.,0.)
                if predicted>max(.01,chosen[key][0]):chosen[key]=(float(predicted),float(actual),float(value_truth))
        fits.append(dict(training_layouts=len(np.unique(sampled)),validation_layouts=len(np.unique(layouts[validation])),
            effect_rmse=float(np.sqrt(np.mean((prediction-truth)**2))),
            effect_correlation=float(np.corrcoef(prediction,truth)[0,1]) if prediction.std()*truth.std()>0 else 0.,
            held_layout_selected_gain=(float(np.mean([value[1] for value in chosen.values()])) if chosen else 0.)
                if np.isfinite(success_truth).all() else None,
            held_layout_selected_value_gain=float(np.mean([value[2] for value in chosen.values()])) if chosen else 0.))
        forests.append(forest)
    return dict(estimator='local-conditional-effects',forests=forests,contexts=normalized,center=center,scale=scale,
        bandwidth=bandwidth,spans=sorted({data['model']['physical_intervention_span'] for data in datasets}),
        confidence=.5,margin=.01,max_changed=2,initial_only=True,physical_features=physical_features,
        threat_features=threat_features,
        prediction_features=prediction_features,
        fit=dict(layouts=len(unique),edges=len(records),leaf=leaf,discordance=discordance,physical_features=physical_features,
            threat_features=threat_features,uniform_effect_weight=uniform_effect_weight,boosted=boosted,
            prediction_features=prediction_features,
            return_horizon=return_horizon,
            failure_horizon=failure_horizon,
            failure_deadline=failure_deadline,
            future_bootstrap=future_bootstrap,
            conditional_outcome_weight=conditional_outcome_weight,
            kernel_bandwidth=kernel_bandwidth,
            support_geometry='bidirectional-conditional-edges',fits=fits))


class CooperativeRoleNetwork(nn.Module):
    """Joint return potentials conditioned on every treated boat's command.

    The intervention enters message passing, so a two-boat effect need not be
    the sum of two unconditional single-boat effects. The free program anchors
    the nuisance state value exactly. Optional M1 response differences augment
    the public-state representation and vanish when M1 is removed.
    """
    def __init__(self,width=32,proposal=False,conditional=True,probability_difference=False,response_features=False,
                 response_standardized=False,all_pairs=False,task_frame=False):
        super().__init__()
        self.proposal=proposal;self.conditional=conditional
        self.probability_difference=probability_difference
        self.response_features=response_features
        self.response_standardized=response_standardized
        self.all_pairs=all_pairs
        self.task_frame=task_frame
        self.message=nn.Sequential(nn.Linear(63 if task_frame else 53,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.node=nn.Sequential(nn.Linear((27 if task_frame else 22)+2*width,width),nn.SiLU())
        self.value=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,1))
        if response_features:
            self.response=nn.Sequential(nn.Linear(49,width),nn.SiLU(),nn.Linear(width,1))
            nn.init.zeros_(self.response[-1].weight);nn.init.zeros_(self.response[-1].bias)
            if response_standardized:
                self.register_buffer('response_location',torch.zeros(16))
                self.register_buffer('response_scale',torch.ones(16))
                self.register_buffer('response_difference_scale',torch.ones(16))

    def forward(self,packet,action,span,reference_only=False):
        public=packet['node'];b,n=action.shape[:2]
        programs,offsets=conditional_action_programs(n,action.device,self.proposal,self.all_pairs)
        if reference_only:programs=programs[:1];offsets=offsets[:1]
        o=len(programs)
        active=(programs>=0.)[None].expand(b,-1,-1)
        gate=torch.where(active,programs[None],action[:,None,:,2])
        actual_gate=packet.get('_response_gates',gate) if self.task_frame and not reference_only else gate
        delta=actual_gate-action[:,None,:,2]
        corrected=(action[:,None,:,:2]+offsets[None]).clamp(-1.,1.)
        rule=public[:,None,:,11:13]
        reference=action[:,None,:,2:3]*action[:,None,:,:2]+(1.-action[:,None,:,2:3])*rule
        response=actual_gate[...,None]*corrected+(1.-actual_gate[...,None])*rule-reference
        duration=torch.as_tensor(span,dtype=action.dtype,device=action.device).reshape(-1,1,1)
        duration=active.to(action.dtype)*duration/40.
        condition=torch.cat((torch.stack((active.to(action.dtype),gate,delta,duration),-1),response),-1)
        state=torch.cat((public[...,:9],action[...,:3],public[...,11:13],public[...,14:16]),-1)
        edge=packet['edge'][...,:9]
        if self.task_frame:
            motion=packet['motion'];c=motion[...,2].cos();s=motion[...,2].sin()
            local_target=public[:,0,2:4]*30.
            target=motion[:,0,:2]+torch.stack((c[:,0]*local_target[:,0]-s[:,0]*local_target[:,1],
                s[:,0]*local_target[:,0]+c[:,0]*local_target[:,1]),-1)
            axis=-target/target.norm(dim=-1,keepdim=True).clamp_min(1e-6)
            cosine=c*axis[:,None,0]+s*axis[:,None,1]
            sine=s*axis[:,None,0]-c*axis[:,None,1]
            def common(vector):
                return torch.stack((cosine[:,:,None]*vector[...,0]-sine[:,:,None]*vector[...,1],
                    sine[:,:,None]*vector[...,0]+cosine[:,:,None]*vector[...,1]),-1)
            geometry=torch.cat((common(public[...,:8].reshape(b,n,4,2)).flatten(-2),public[...,8:9]),-1)
            state=torch.cat((geometry,action[...,:3],public[...,11:13],public[...,14:16],cosine[...,None],sine[...,None]),-1)
            edge=torch.cat((common(edge[...,:2]),common(edge[...,2:4]),edge[...,4:]),-1)
            surge=response.sum(-1)*750./380.
            yaw=(response[...,0]-response[...,1])*750.*1.25/1430.
            effect=torch.stack((surge*cosine[:,None]/5.,surge*sine[:,None]/5.,yaw),-1)
            condition=torch.cat((condition,effect),-1)
        node=torch.cat((state[:,None].expand(-1,o,-1,-1),condition),-1).clamp(-20.,20.)
        edge=edge.clamp(-20.,20.)[:,None].expand(-1,o,-1,-1,-1)
        messages=self.message(torch.cat((node[:,:,:,None].expand(-1,-1,-1,n,-1),
            node[:,:,None].expand(-1,-1,n,-1,-1),edge),-1))
        mask=~torch.eye(n,dtype=torch.bool,device=action.device)[None,None]
        weights=torch.exp(-.75*edge[...,7])*mask
        average=(messages*weights[...,None]).sum(3)/weights.sum(3).clamp_min(1e-6)[...,None]
        maximum=messages.masked_fill(~mask[...,None],-1e6).amax(3)
        encoded=self.node(torch.cat((node,average,maximum),-1))
        value=self.value(torch.cat((encoded.mean(2),encoded.amax(2)),-1)).squeeze(-1)
        if reference_only:return value.sigmoid().squeeze(-1)
        if self.response_features:
            features=packet['_response_features']
            reference=features[:,:1].expand_as(features)
            difference=features-reference
            if self.response_standardized:
                features=((features-self.response_location)/self.response_scale).clamp(-10.,10.)
                reference=((reference-self.response_location)/self.response_scale).clamp(-10.,10.)
                difference=(difference/self.response_difference_scale).clamp(-10.,10.)
            time=torch.as_tensor(span,dtype=action.dtype,device=action.device).reshape(-1,1,1).expand(b,o,1)/40.
            effect=self.response(torch.cat((reference,features,difference,time),-1)).squeeze(-1)
            value=value+effect-effect[:,:1]
        if self.conditional and self.probability_difference:
            probability=value.sigmoid()
            return probability-probability[:,:1]
        return (value-value[:,:1]).tanh() if self.conditional else value.sigmoid()


class CooperativeRoleCredit(nn.Module):
    def __init__(self,states,contexts,spans=(20,40),width=32,confidence=.5,margin=.02,bandwidth=.3,
                 proposal=False,conditional=True,probability_difference=False,response_features=False,
                 response_standardized=False,max_changed=0,max_interventions=0,initial_only=False,wait_span=False,
                 all_pairs=False,coordinate_steps=0,task_frame=False):
        super().__init__();self.models=nn.ModuleList()
        self.proposal=proposal;self.conditional=conditional
        self.response_features=response_features
        self.max_changed=max_changed
        self.max_interventions=max_interventions;self.initial_only=initial_only
        self.wait_span=wait_span
        self.all_pairs=all_pairs;self.coordinate_steps=coordinate_steps
        self.task_frame=task_frame
        for state in states:
            net=CooperativeRoleNetwork(width,proposal,conditional,probability_difference,response_features,response_standardized,all_pairs,task_frame)
            net.load_state_dict(state);self.models.append(net.eval())
        self.spans=tuple(spans);self.confidence=confidence;self.margin=margin;self.bandwidth=bandwidth
        for i,context in enumerate(contexts):
            self.register_buffer('contexts_'+str(i),torch.as_tensor(context,dtype=torch.float32))

    def eligible(self,packet):
        use=torch.ones(len(packet['node']),device=packet['node'].device,dtype=torch.bool)
        if self.initial_only:use=use&(packet['node'][:,0,15].abs()<1e-7)
        if self.max_interventions:
            count=packet.get('_conditional_count',packet['node'].new_zeros((len(use),1)))
            use=use&(count[:,0]<self.max_interventions)
        return use

    def forward(self,packet,action):
        context=public_return_context(packet,version=2) if self.bandwidth>0. else None
        programs,offsets=conditional_action_programs(action.shape[1],action.device,self.proposal,self.all_pairs)
        descriptions=torch.cat((programs[...,None],offsets),-1)
        scores=[]
        for i,span in enumerate(self.spans):
            values=torch.stack([net(packet,action,span) for net in self.models])
            if not self.conditional:values=values-values[:,:,:1]
            score=values.mean(0)-self.confidence*values.std(0,unbiased=False)
            if self.coordinate_steps:
                choice=torch.zeros(len(action),dtype=torch.long,device=action.device)
                for _ in range(self.coordinate_steps):
                    reference=values.gather(2,choice[None,:,None].expand(len(self.models),-1,1))
                    difference=values-reference
                    marginal=difference.mean(0)-self.confidence*difference.std(0,unbiased=False)
                    adjacent=(descriptions[None]!=descriptions[choice,None]).any(-1).sum(-1)<=1
                    marginal=marginal.masked_fill(~adjacent,-1e6)
                    gain,next_choice=marginal.max(-1)
                    choice=torch.where(gain>self.margin,next_choice,choice)
                allowed=torch.zeros_like(score,dtype=torch.bool).scatter_(1,choice[:,None],True)
                allowed[:,0]=True
                score=score.masked_fill(~allowed,-1e6)
            if self.max_changed:
                score=score.masked_fill((programs>=0.).sum(-1)[None]>self.max_changed,-1e6)
            score=score.masked_fill(~self.eligible(packet)[:,None],0.)
            if context is not None:
                distance,_,_=match_fleet_context(context,getattr(self,'contexts_'+str(i)),count=1)
                support=torch.exp(-distance[:,0]/(2.*self.bandwidth**2))
                score=score*support[:,None]
            scores.append(score)
        gain,index=torch.stack(scores,1).flatten(1).max(-1)
        selected=programs[index%len(programs)]
        selected=torch.where((gain>self.margin)[:,None],selected,-torch.ones_like(selected))
        span=action.new_tensor(self.spans)[index//len(programs)]
        result=torch.cat((action[...,:3],selected[...,None],
            span[:,None,None].expand(-1,action.shape[1],1),selected[...,None]),-1)
        if self.proposal:
            correction=offsets[index%len(programs)]*(gain>self.margin)[:,None,None]
            result=torch.cat((result,correction),-1)
        return result


def fit_cooperative_role_credit(datasets,width=32,heads=5,updates=1200,seed=42,device='cpu',
                                proposal=False,conditional=True,auxiliary_weight=0.,probability_difference=False,
                                response_features=False,response_standardized=False,max_changed=0,future_bootstrap=False,
                                all_pairs=False,coordinate_steps=0,task_frame=False):
    """Learn conditional graph contrasts plus a paired free-policy anchor."""
    import numpy as np
    identity=('source_sha256','proposal_initialization_sha256','environment_protocol','mechanism_ablation',
              'peer','preference_strength')
    if any(any(dataset['model'].get(key)!=datasets[0]['model'].get(key) for key in identity)
           for dataset in datasets):
        raise ValueError('Conditional actions must share the same base, ablation and decoder.')
    records={}
    for dataset in datasets:
        if dataset['model'].get('proposal_interventions') and not proposal:
            raise ValueError('Proposal corrections must be encoded as distinct conditional actions.')
        span=dataset['model']['physical_intervention_span']
        for roll in dataset['rollouts']:
            branches=[branch for branch in roll['branches'] if not branch.get('observed_future',False)]
            if not branches:continue
            first=branches[0];packet=first['packet']
            key=(roll['row']['seed'],span,hashlib.sha256(packet['node'].tobytes()).hexdigest())
            if key not in records:records[key]=dict(packet=packet,action=first['reference_action'][:,:3],samples={})
            samples=records[key]['samples']
            for branch in branches:
                reference=branch['factual_outcome'].get('value_target',branch['factual_outcome']['success']);future=branch['continuation_seed']
                for side,name in (('left','before'),('right','after')):
                    arm=branch[side];offset=arm[:,6:8] if arm.shape[-1]>6 else np.zeros((len(arm),2))
                    gates=tuple(float(x) for x in np.concatenate((arm[:,3:4],offset),-1).flatten())
                    value=branch[name].get('value_target',branch[name]['success'])
                    if reference is not None and value is not None:
                        samples.setdefault(gates,{}).setdefault(future,set()).add(float(value-reference if conditional else value))
    keys=list(records);rows=list(records.values());n=len(rows[0]['action'])
    programs,offsets=conditional_action_programs(n,'cpu',proposal,all_pairs)
    descriptions=torch.cat((programs[...,None],offsets),-1).numpy()
    lookup={tuple(float(g) for g in row.flatten()):i for i,row in enumerate(descriptions)}
    programs=programs.numpy()
    target=np.zeros((len(rows),len(programs)),np.float32);mask=np.zeros_like(target);future_tables=[]
    for i,row in enumerate(rows):
        futures=sorted({future for observations in row['samples'].values() for future in observations})
        future_index={future:j for j,future in enumerate(futures)}
        table=np.full((len(programs),len(futures)),np.nan,np.float32)
        for gates,observations in row['samples'].items():
            if gates in lookup:
                j=lookup[gates];values=np.asarray([np.mean(list(values)) for values in observations.values()])
                target[i,j]=values.mean();mask[i,j]=1.
                for future,values in observations.items():table[j,future_index[future]]=np.mean(list(values))
        future_tables.append(table)
    packet={key:torch.as_tensor(np.stack([row['packet'][key] for row in rows]),dtype=torch.float32,device=device)
        for key in (('node','edge','motion','obs') if response_features else ('node','edge'))}
    action=torch.as_tensor(np.stack([row['action'] for row in rows]),dtype=torch.float32,device=device)
    if response_features:
        import develop_source_fusion as d
        model=datasets[0]['model']
        predictor=d.JointLearningActor(peer=model.get('peer',True),graph=True,levels=model.get('levels',3),
            risk_prior=model.get('risk_prior',0.),risk_horizon=model.get('risk_horizon',3.2),
            risk_clearance=model.get('risk_clearance',7.),risk_feedback=model.get('risk_feedback',False),
            prediction_step=model.get('prediction_step',.1),risk_barrier=model.get('risk_barrier',0.),
            risk_tail=model.get('risk_tail',0.),frozen_proposal=True)
        predictor.load_state_dict(model['actor']);predictor.preference_strength=model.get('preference_strength',0.)
        d.ablate_joint_mechanisms(predictor,*model.get('mechanism_ablation',(False,False,False)))
        predictor=predictor.to(device).eval()
        with torch.no_grad():
            responses=[predictor.conditional_response_features(
                {key:value[begin:begin+8] for key,value in packet.items()},action[begin:begin+8],proposal,
                return_decisions=task_frame,all_pairs=all_pairs) for begin in range(0,len(rows),8)]
            features=torch.cat([result[0] for result in responses]) if task_frame else torch.cat(responses)
            if task_frame:packet['_response_gates']=torch.cat([result[1] for result in responses])
        packet['_response_features']=features
        del predictor
    target=torch.as_tensor(target,device=device);mask=torch.as_tensor(mask,device=device)
    spans=torch.tensor([key[1] for key in keys],dtype=torch.float32,device=device)
    layouts=np.asarray([key[0] for key in keys]);unique=np.unique(layouts)
    if len(unique)<8:raise ValueError('Conditional graph credit requires at least eight layouts.')
    if auxiliary_weight:
        factual=[]
        for dataset in datasets:
            for rollout in dataset['rollouts']:
                steps=rollout['steps']
                for index in np.linspace(0,len(steps)-1,min(8,len(steps)),dtype=int):
                    factual.append((rollout['row']['seed'],steps[index],rollout['row']['success']))
        factual_layouts=np.asarray([row[0] for row in factual])
        factual_packet={key:torch.as_tensor(np.stack([row[1]['packet'][key] for row in factual]),
            dtype=torch.float32,device=device) for key in (('node','edge','motion') if task_frame else ('node','edge'))}
        factual_action=torch.as_tensor(np.stack([row[1]['action'][:,:3] for row in factual]),dtype=torch.float32,device=device)
        factual_target=torch.tensor([row[2] for row in factual],dtype=torch.float32,device=device)
    adjacency=np.count_nonzero(np.any(descriptions[:,None]!=descriptions[None],axis=-1),axis=-1)==1
    left,right=np.nonzero(np.triu(adjacency,1))
    left=torch.as_tensor(left,device=device);right=torch.as_tensor(right,device=device)
    def loss_for(prediction,index,training=False):
        errors=prediction-(training_target if training else target)[index]
        valid=(training_mask if training else mask)[index]
        anchor=(errors.square()*valid).sum(-1)/valid.sum(-1).clamp_min(1.)
        if not conditional:return anchor
        edge_mask=valid[:,left]*valid[:,right]
        contrast=((errors[:,left]-errors[:,right]).square()*edge_mask).sum(-1)/edge_mask.sum(-1).clamp_min(1.)
        return contrast+.25*anchor
    states=[];fits=[]
    with torch.enable_grad():
        for head in range(heads):
            torch.manual_seed(seed+head*7919);rng=np.random.default_rng(seed+head*7919)
            sampled=rng.choice(unique,size=len(unique),replace=True)
            train=np.concatenate([np.flatnonzero(layouts==layout) for layout in sampled])
            validation=torch.as_tensor(np.flatnonzero(~np.isin(layouts,sampled)),device=device)
            if not len(validation):raise ValueError('No held-out layout for conditional graph credit.')
            training_target=target;training_mask=mask
            if future_bootstrap:
                values=[];valid=[]
                for table in future_tables:
                    draw=table[:,rng.integers(table.shape[1],size=table.shape[1])]
                    observed=np.isfinite(draw);count=observed.sum(1)
                    values.append(np.nan_to_num(draw).sum(1)/np.maximum(count,1))
                    valid.append(count>0)
                training_target=torch.as_tensor(np.asarray(values),dtype=target.dtype,device=device)
                training_mask=torch.as_tensor(np.asarray(valid),dtype=mask.dtype,device=device)
            net=CooperativeRoleNetwork(width,proposal,conditional,probability_difference,response_features,response_standardized,all_pairs,task_frame).to(device)
            if response_features and response_standardized:
                training_features=packet['_response_features'][torch.as_tensor(np.unique(train),device=device)]
                net.response_location.copy_(training_features[:,0].mean(0))
                net.response_scale.copy_(training_features[:,0].std(0).clamp_min(.05))
                net.response_difference_scale.copy_((training_features-training_features[:,:1]).flatten(0,1).std(0).clamp_min(1e-3))
            if auxiliary_weight:
                factual_train=np.concatenate([np.flatnonzero(factual_layouts==layout) for layout in sampled])
                factual_validation=np.flatnonzero(~np.isin(factual_layouts,sampled))
                factual_validation=torch.as_tensor(factual_validation[::max(1,len(factual_validation)//256)],device=device)
            optimizer=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=.03)
            best=float('inf');best_update=0;best_state=None
            for update in range(updates):
                index=torch.as_tensor(rng.choice(train,size=min(32,len(train)),replace=True),device=device)
                inputs={key:value[index] for key,value in packet.items()}
                loss=loss_for(net(inputs,action[index],spans[index]),index,training=True).mean()
                if auxiliary_weight:
                    selection=torch.as_tensor(rng.choice(factual_train,size=32,replace=True),device=device)
                    prediction=net({key:value[selection] for key,value in factual_packet.items()},
                        factual_action[selection],20,reference_only=True)
                    loss=loss+auxiliary_weight*F.binary_cross_entropy(prediction,factual_target[selection])
                optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
                if (update+1)%25==0:
                    with torch.no_grad():
                        validation_loss=torch.cat([loss_for(net({key:value[part] for key,value in packet.items()},
                            action[part],spans[part]),part) for part in validation.split(32)]).mean().item()
                        if auxiliary_weight:
                            prediction=torch.cat([net({key:value[part] for key,value in factual_packet.items()},
                                factual_action[part],20,reference_only=True) for part in factual_validation.split(32)])
                            validation_loss+=auxiliary_weight*F.binary_cross_entropy(
                                prediction,factual_target[factual_validation]).item()
                    if validation_loss<best:
                        best=validation_loss;best_update=update+1
                        best_state={key:value.detach().cpu().clone() for key,value in net.state_dict().items()}
                    elif update+1-best_update>=(500 if auxiliary_weight else 250):break
            states.append(best_state);fits.append(dict(validation_loss=best,selected_update=best_update,
                training_layouts=len(np.unique(sampled)),validation_layouts=len(np.unique(layouts[validation.cpu().numpy()]))))
    durations=sorted(set(key[1] for key in keys))
    contexts=[]
    for span in durations:
        contexts.append(torch.cat([public_return_context({key:torch.as_tensor(value[None])
            for key,value in row['packet'].items()},version=2)
            for key,row in zip(keys,rows) if key[1]==span]))
    return dict(states=states,contexts=contexts,spans=durations,width=width,confidence=.5,margin=.02,
        bandwidth=.3,proposal=proposal,conditional=conditional,probability_difference=probability_difference,
        response_features=response_features,response_standardized=response_standardized,max_changed=max_changed,
        all_pairs=all_pairs,coordinate_steps=coordinate_steps,task_frame=task_frame),dict(roots=len(rows),layouts=len(unique),
            observed_programs=int(mask.sum()),fits=fits,conditional=conditional,proposal=proposal,
            factual_states=len(factual) if auxiliary_weight else 0,auxiliary_weight=auxiliary_weight,
            probability_difference=probability_difference,
            response_features=response_features,
            response_standardized=response_standardized,max_changed=max_changed,
            future_bootstrap=future_bootstrap,
            all_pairs=all_pairs,coordinate_steps=coordinate_steps,
            task_frame=task_frame,
            supervision=('Single-vessel conditional graph contrasts and paired free-policy anchors.' if conditional
                else 'Capacity-matched ordinary terminal-return regression on the same physical continuations.'))


@torch.no_grad()
def component_relation_features(packet, base, programs, parts, bias=-.05):
    """Bind a pair's current actuator candidates to relative motion geometry."""
    from policy.interaction_sac import CandidateCoordinationGate
    motion, obs = packet['motion'], packet['obs']
    count, options, vessels = programs.shape[:3]
    reference = (base[..., 2] + bias).clamp(0., 1.)
    gates = (reference[:, None, :, None] + programs).clamp(0., 1.)
    rule = 750. * obs[..., 12:14] + 250.
    learned = 750. * base[..., :2] + 250.
    thrust = rule[:, None] + gates * (learned - rule)[:, None]
    _, paths = CandidateCoordinationGate.candidate_responses(
        packet, base, applied_thrust=thrust.transpose(1, 2))
    relative = paths[:, :, -1].transpose(1, 2)
    displacement = (relative[:, :1] - relative) / 5.
    rows = torch.arange(count)[:, None].expand(-1, options)
    first, second = programs[:, parts[:, 0]], programs[:, parts[:, 1]]
    left, right = first.abs().amax(-1).argmax(-1), second.abs().amax(-1).argmax(-1)
    di, dj = displacement[rows, parts[:, 0], left], displacement[rows, parts[:, 1], right]
    pi, pj = motion[rows, left, :2], motion[rows, right, :2]
    vi, vj = motion[rows, left, 3:5], motion[rows, right, 3:5]
    angle = motion[..., 2] + obs[..., 3]
    direction = torch.stack((angle.cos(), angle.sin()), -1)
    ni, nj = direction[rows, left], direction[rows, right]
    distance_i, distance_j = obs[rows, left, 2], obs[rows, right, 2]
    separation = pi - pj
    length = separation.norm(dim=-1)
    unit = separation / length[..., None].clamp_min(1e-6)
    cross = lambda x, y: x[..., 0] * y[..., 1] - x[..., 1] * y[..., 0]
    norm_i, norm_j = di.norm(dim=-1), dj.norm(dim=-1)
    features = torch.stack(((ni * nj).sum(-1), cross(ni, nj).abs(), torch.exp(-length / 10.),
        ((vi - vj) * unit).sum(-1) / 5., torch.minimum(distance_i, distance_j) / 20.,
        (distance_i - distance_j).abs() / 20., (di * dj).sum(-1), ((di - dj) * unit).sum(-1),
        (di * ni + dj * nj).sum(-1), (cross(di, ni) - cross(dj, nj)).abs(),
        norm_i + norm_j, (norm_i - norm_j).abs()), -1).clamp(-3., 3.)
    changed = (first.abs().flatten(2).amax(-1) > 1e-7) & (second.abs().flatten(2).amax(-1) > 1e-7)
    return features * (changed & (parts > 0).all(-1)[None])[..., None]


def conditional_edges(parts):
    """Each edge changes one vessel while retaining its partner's control."""
    edges = []
    for option, (left, right) in enumerate(parts.tolist()):
        if option == 0:
            continue
        if left and right:
            edges.extend(((option, left), (option, right)))
        else:
            edges.append((option, 0))
    return torch.tensor(edges, dtype=torch.long, device=parts.device)


@torch.no_grad()
def component_response_features(packet, base, programs, parts, response, bias=-.05, response_scale=None):
    """Encode conditional joint motion from the same public, filtered rollout.

    The second difference removes each isolated control's motion and common
    target motion. It measures how the partner changes a control's effect.
    No realized return or environment future is available to this encoder.
    """
    features = component_relation_features(packet, base, programs, parts, bias)
    count, options = programs.shape[:2]
    rows = torch.arange(count, device=programs.device)[:, None].expand(-1, options)
    first, second = programs[:, parts[:, 0]], programs[:, parts[:, 1]]
    left = first.abs().amax(-1).argmax(-1)
    right = second.abs().amax(-1).argmax(-1)
    joint = response - response[:, parts[:, 0]] - response[:, parts[:, 1]] + response[:, :1]
    di, dj = joint[rows, torch.arange(options), left] / 5., joint[rows, torch.arange(options), right] / 5.
    pi, pj = response[rows, 0, left], response[rows, 0, right]
    ni = pi / pi.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    nj = pj / pj.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    separation = pi - pj
    unit = separation / separation.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cross = lambda x, y: x[..., 0]*y[..., 1] - x[..., 1]*y[..., 0]
    effects = torch.stack((-(di*ni+dj*nj).sum(-1),
        (di-dj).mul(unit).sum(-1), di.norm(dim=-1)+dj.norm(dim=-1),
        (cross(di, ni)-cross(dj, nj)).abs()), -1).flatten(-2).clamp(-3., 3.)
    changed = (first.abs().flatten(2).amax(-1)>1e-7) & (second.abs().flatten(2).amax(-1)>1e-7)
    effects *= (changed & (parts>0).all(-1)[None])[..., None]
    if response_scale is not None:
        effects = (effects / effects.new_tensor(response_scale)).clamp(-5., 5.)
    return torch.cat((features, effects), -1)


def paired_targets(costs, edges):
    """Cross-fit precision using different futures from each target mean."""
    if costs.shape[1] < 4:
        raise ValueError('At least four paired futures are required.')
    contrast = costs[:, :, edges[:, 0]] - costs[:, :, edges[:, 1]]
    folds = []
    for half in (0, 1):
        target = contrast[:, half::2].mean(1)
        other = contrast[:, 1-half::2]
        precision = 1. / (4. + other.var(1, unbiased=True) / other.shape[1])
        folds.append((target, precision))
    return folds


def paired_covariance_targets(costs, edges):
    """Cross-fit the shared noise directions of conditional return edges."""
    contrast = costs[:, :, edges[:, 0]] - costs[:, :, edges[:, 1]]
    folds = []
    for half in (0, 1):
        target = contrast[:, half::2].mean(1)
        other = contrast[:, 1-half::2]
        centered = other - other.mean(1, keepdim=True)
        size, repeats, edges_count = centered.shape
        covariance = torch.einsum('bre,brf->ef', centered, centered) / (size * (repeats-1) * repeats)
        covariance = .8*covariance + .2*torch.diag(covariance.diag()) + 4.*torch.eye(edges_count, dtype=costs.dtype, device=costs.device)
        precision = torch.cholesky_inverse(torch.linalg.cholesky(covariance))
        folds.append((target, precision))
    return folds


def conditional_auxiliary(energy, edges, folds, mask, *, quadratic=False, covariance=False):
    prediction = energy[:, edges[:, 0]] - energy[:, edges[:, 1]]
    losses = []
    for target, precision in folds:
        if covariance:
            residual = (prediction-target) * mask.sqrt()
            losses.append(.5*torch.einsum('be,ef,bf->b', residual, precision, residual) / mask.sum(-1).clamp_min(1))
            continue
        error = .5 * (prediction-target).square() if quadratic else F.smooth_l1_loss(prediction, target, beta=2., reduction='none')
        losses.append((error * precision * mask).sum(-1) / mask.sum(-1).clamp_min(1))
    return torch.stack(losses).mean(0)


@torch.no_grad()
def decision_credit_mask(subject, initial, data, prior, settings, edges, mask):
    """Focus credit on the incumbent's candidate neighborhood, with coverage."""
    unary = subject.reactive_monotone_estimates(initial, prior.double(), data['parts'], data['packet'], False).mean((0, 2))
    score = unary + settings['prior'] * data['options'].square().mean((-1, -2))
    singles = 1 + 4 * data['base'].shape[1]
    support = torch.zeros_like(score, dtype=torch.bool)
    support[:, :singles] = True
    support.scatter_(1, score[:, singles:].argsort(-1)[:, :settings['component_candidate_budget']]+singles, True)
    probability = (-score.masked_fill(~support, float('inf')) / 2.).softmax(-1)
    weight = .2 + .8 * score.shape[-1] * probability[:, edges[:, 0]]
    return mask * weight


def state_features(data, prior):
    distance = data['packet']['obs'][..., 2].double()
    remaining = data['packet']['time_remaining'].double()
    baseline = prior.double().mean(1)[:, 0]
    return torch.stack((torch.ones_like(baseline), baseline / 20., distance.amin(-1) / 20.,
        distance.mean(-1) / 30., remaining, baseline.square() / 400.), -1)


def decisions(subject, coefficients, data, prior, settings, interaction, risk):
    prediction = subject.reactive_monotone_estimates(
        coefficients.double(), prior.double(), data['parts'], data['packet'], interaction).mean(2)
    independent = subject.reactive_monotone_estimates(
        coefficients.double(), prior.double(), data['parts'], data['packet'], False).mean((0, 2))
    penalty = settings['prior'] * data['options'].square().mean((-1, -2))
    initial = independent + penalty
    reference = initial.argmin(-1)
    if not interaction:
        return reference, prediction
    score = subject.reactive_component_paired_score(prediction, data['options'], reference, risk, settings['prior'])
    singles = 1 + 4 * data['base'].shape[1]
    joints = initial[:, singles:].argsort(-1)[:, :settings['component_candidate_budget']] + singles
    support = torch.zeros_like(score, dtype=torch.bool)
    support[:, :singles] = True
    support.scatter_(1, joints, True)
    return score.masked_fill(~support, float('inf')).argmin(-1), prediction


def fit_conditional_component(args, subject):
    """Keep absolute value fitting and add local conditional credit supervision."""
    torch.set_num_threads(2)
    bank = torch.load(args.reactive_bootstrap_data, map_location='cpu', weights_only=False)
    data, settings = bank['data'], dict(bank['settings'])
    if not bank.get('complete') or not settings.get('component_all_pairs'):
        raise ValueError('Collect complete deployment-supported joint controls first.')
    if len(data['parts']) != 1 + 4 * settings['defenders'] + 16 * settings['defenders'] * (settings['defenders']-1) // 2:
        raise ValueError('Training and deployment must have the same candidate support.')
    settings.update(component_orthogonal=True, component_contextual=True, component_candidate_budget=16,
        conditional_value_version='absolute-plus-single-vessel-credit-v1')
    focused = getattr(args, 'decision_loss', False)
    covariance = getattr(args, 'component_covariance_credit', False)
    if focused:
        settings['conditional_value_version'] = 'absolute-plus-decision-local-mean-credit-v2'
    if covariance:
        settings['conditional_value_version'] = 'absolute-plus-correlated-single-vessel-credit-v3'
    if 'model_response' in data:
        settings['component_response_features'] = True
        features = component_response_features(data['packet'], data['base'], data['options'],
            data['parts'], data['model_response'], settings['bias'])
        training_rows = torch.isin(data['origin'], bank['training_origins'])
        scale = features[training_rows, :, 12:].square().mean((0,1)).sqrt().clamp_min(.02)
        settings['component_response_scale'] = scale.tolist()
        features = torch.cat((features[..., :12], (features[..., 12:]/scale).clamp(-5.,5.)), -1)
    else:
        features = component_relation_features(data['packet'], data['base'], data['options'], data['parts'], settings['bias'])
    packet = dict(data['packet'], candidate_interactions=features)
    data = dict(data, packet=packet)
    train = torch.isin(data['origin'], bank['training_origins']).nonzero().flatten()
    valid = torch.isin(data['origin'], bank['held_origins']).nonzero().flatten()
    if len(train) < 32 or len(valid) < 16:
        raise ValueError('Use at least 32 training and 16 held states.')
    training, validation = subject.reactive_subset(data, train), subject.reactive_subset(data, valid)
    prior, valid_prior = bank['prior'][train], bank['prior'][valid]
    seed = args.seed if args.bootstrap_seed is None else args.bootstrap_seed
    parts = data['parts']; singles = 1 + 4 * settings['defenders']
    unary_data = dict(training, costs=training['costs'][:, :, :singles],
        options=training['options'][:, :singles], parts=parts[:singles],
        packet={k:(v[:, :singles] if k == 'candidate_interactions' else v) for k, v in training['packet'].items()})
    # The shared initialization uses ordinary returns. Conditional supervision
    # is absent from the ordinary-return ablation, including its unary fit.
    initial, _ = subject.fit_prior_monotone(unary_data, prior[:, :, :singles], False, False,
        seed, 3, True, False, False, True)
    initial = F.pad(initial, (0, features.shape[-1])).double()
    costs = training['costs'].double(); target = costs.mean(1)
    variance = 4. + costs.var(1, unbiased=True) / costs.shape[1]
    unique = subject.reactive_unique_control_mask(training['options'].flatten(2)).double()
    state = state_features(training, prior)
    edges = conditional_edges(parts)
    changed = (training['options'][:, edges[:, 0]] - training['options'][:, edges[:, 1]]).abs().flatten(2).amax(-1) > 1e-7
    edge_mask = changed.double() * unique[:, edges[:, 0]]
    if focused:
        edge_mask = decision_credit_mask(subject, initial, training, prior, settings, edges, edge_mask)
    folds = paired_covariance_targets(costs, edges) if covariance else paired_targets(costs, edges)
    for arm in args.evaluation_arms.split(','):
        if arm not in ('difference', 'absolute', 'additive'):
            raise ValueError('Expected difference, absolute, or additive.')
        conditional, interaction = arm != 'absolute', arm != 'additive'
        fitted, histories, baselines = [], [], []
        for head in range(len(initial)):
            generator = torch.Generator().manual_seed(seed + head * 104729)
            rows, _ = subject.reactive_positive_credit_weights(subject.reactive_state_groups(training), costs.shape[1], generator)
            weights = nn.Parameter(initial[head:head+1].clone())
            baseline = nn.Parameter(torch.linalg.lstsq(state, target[:, 0]).solution)
            optimizer = torch.optim.LBFGS([weights, baseline], max_iter=args.updates, line_search_fn='strong_wolfe')
            def losses():
                energy = subject.reactive_monotone_estimates(weights, prior.double(), parts, training['packet'], interaction)[0].mean(1)
                total = energy + (state @ baseline)[:, None]
                absolute = (((total-target).square() / variance) * unique).sum(-1) / unique.sum(-1)
                auxiliary = conditional_auxiliary(energy, edges, folds, edge_mask, quadratic=focused, covariance=covariance)
                regularizer = .02 * (weights[..., :4]-initial[head:head+1, :, :4]).square().mean() + .1 * weights[..., 4:].square().mean()
                loss = (rows * (absolute + (args.aux_weight * auxiliary if conditional else 0.))).mean() + regularizer
                return loss, absolute.mean(), auxiliary.mean()
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss, _, _ = losses(); loss.backward(); return loss
            optimizer.step(closure)
            loss, absolute, auxiliary = losses()
            fitted.append(weights.detach()[0].float()); baselines.append(baseline.detach().float())
            histories.append(dict(loss=float(loss.detach()), absolute=float(absolute.detach()), conditional=float(auxiliary.detach())))
        coefficients = torch.stack(fitted); trials = []
        for risk in (0., .5, 1., 2.):
            choice, _ = decisions(subject, coefficients, validation, valid_prior, settings, interaction, risk)
            actual = validation['costs'].mean(1)[torch.arange(len(valid)), choice]
            trials.append(dict(risk=risk, cost=float(actual.mean())))
        selection = min(trials, key=lambda item: item['cost'])
        actor = subject.ReactiveComponentGate(settings, coefficients, selection['risk'], interaction)
        saved = dict(kind='reactive-component-v1', actor=actor.state_dict(), settings=settings, calibration=coefficients,
            risk=selection['risk'], interaction=interaction, conditional=conditional, seed=args.seed, arm=arm,
            training_seed=seed, selection=selection, calibration_trials=trials, history=histories,
            absolute_baseline=torch.stack(baselines), auxiliary_weight=args.aux_weight if conditional else 0.,
            training_origins=bank['training_origins'], held_origins=bank['held_origins'],
            actual_training_states=len(train), actual_held_states=len(valid), collecting_policy=bank['collecting_policy'],
            data_sha256=hashlib.sha256(args.reactive_bootstrap_data.read_bytes()).hexdigest(),
            training_implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()+Path(subject.__file__).read_bytes()).hexdigest())
        subject.atomic_torch(args.output / f'reactive-{arm}-{args.seed}.pth', saved)
        print(json.dumps(dict(arm=arm, selection=selection, trials=trials, training=histories)), flush=True)
