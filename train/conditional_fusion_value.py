"""Candidate-conditioned values with anchored, single-vessel credit learning."""
import hashlib
import json
from functools import lru_cache
from math import isfinite
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def fusion_response_rollout(actor, packet, gates, *, targets=None, currents=None):
    """Differentiate joint fusion responses through the unchanged source commands.

    ``gates`` has shape (batch, time, vessels, 1 or 2). Both channels stay
    inside the convex hull of their two source commands. Optional measured
    target paths and paired disturbances are training observations; a public
    forecast uses constant target velocity and zero future current instead.
    The returned intermediate states support response-effect supervision,
    which is distinct from subtracting terminal-success means.
    """
    batch, horizon, vessels, channels = gates.shape
    if channels not in (1, 2):
        raise ValueError('Fusion has one shared coefficient or two actuator coefficients.')
    motion = packet['motion'].to(gates)
    pose, water = motion[..., :3], (5. * packet['water']).to(gates)
    ground = motion[..., 3:6]
    node = packet['node'].to(gates)
    heading = motion[:, 0, 2]
    c, s = heading.cos(), heading.sin()
    def world(vector):
        return torch.stack((c*vector[:, 0]-s*vector[:, 1],
                            s*vector[:, 0]+c*vector[:, 1]), -1)
    target = motion[:, 0, :2] + world(node[:, 0, 2:4]*30.)
    target_velocity = world(node[:, 0, 4:6]*5.)
    if targets is None:
        clock = torch.arange(horizon+1, device=gates.device, dtype=gates.dtype)*.2
        targets = torch.cat((target[:, None]+clock[None, :, None]*target_velocity[:, None],
                            target_velocity[:, None].expand(-1, horizon+1, -1)), -1)
    if currents is None:
        currents = gates.new_zeros((batch, horizon, vessels, 3))
    eye = torch.eye(vessels, device=gates.device, dtype=gates.dtype)
    mask = ~eye.bool()
    observation = packet['obs'].to(device=gates.device, dtype=torch.float32)
    rule = node[..., 11:13]
    positions, velocities, controls, observations = [], [], [], []
    for tick in range(horizon):
        source = actor.proposal_distribution(observation)[0].tanh().to(gates)
        command = gates[:, tick]*source + (1.-gates[:, tick])*rule
        force = 750.*command + 250.
        controls.append(command)
        for _ in range(4):
            ground = water+currents[:, tick]
            pose = pose+.05*ground
            h = pose[..., 2]; c, s = h.cos(), h.sin()
            ur = c*water[..., 0]+s*water[..., 1]
            vr = -s*water[..., 0]+c*water[..., 1]
            u = c*ground[..., 0]+s*ground[..., 1]
            v = -s*ground[..., 0]+c*ground[..., 1]
            yaw = ground[..., 2]
            du = (400.*yaw*v-(100.+150.*ur.abs())*ur+force.sum(-1))/380.
            dv = (-400.*yaw*u+20.*yaw*ur-(100.+150.*vr.abs())*vr)/400.
            dr = (-20.*ur*vr-(980.+950.*yaw.abs())*yaw
                  +1.25*(force[..., 0]-force[..., 1]))/1430.
            ur, vr, yaw = ur+.05*du, vr+.05*dv, yaw+.05*dr
            water = torch.stack((c*ur-s*vr, s*ur+c*vr, yaw), -1)
        position = pose[..., :2]
        positions.append(position); velocities.append(water)
        relative = position[:, None]-position[:, :, None]
        distance = (relative+eye[None, :, :, None]).norm(dim=-1)
        separation = (-relative/(distance-3.5).clamp_min(.01)[..., None].square()
                      *((distance < 15.) & mask)[..., None]).sum(-2)
        alignment = ground[..., :2].mean(1, keepdim=True).expand_as(position)
        cohesion = position.mean(1, keepdim=True)-position
        attraction = targets[:, tick+1, None, :2]-position
        field = .5*attraction+10.*separation+.1*alignment+.1*cohesion
        field = field/field.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        forward = c*field[..., 0]+s*field[..., 1]
        turn = torch.atan2(-s*field[..., 0]+c*field[..., 1], forward)
        rule = ((1000.*torch.stack((forward+.75*turn, forward-.75*turn), -1))
                .clamp(-500., 1000.)-250.)/750.
        def polar(vector):
            # atan2(0,0) has an undefined derivative at initially zero speed.
            x, y = vector.unbind(-1)
            x = torch.where((x.abs()+y.abs()) < 1e-12, x.new_full((), 1e-12), x)
            angle = torch.atan2(y, x)-h
            return torch.stack((vector.norm(dim=-1),
                                torch.remainder(angle+torch.pi, 2.*torch.pi)-torch.pi), -1)
        bearing = torch.atan2(relative[..., 1]+eye[None]*1e-12,
                              relative[..., 0]+eye[None]*1e-12)-h[..., None]
        neighbor = torch.stack((distance, torch.remainder(bearing+torch.pi, 2.*torch.pi)-torch.pi), -1)
        observation = torch.cat((polar(-position), polar(attraction),
            polar(targets[:, tick+1, None, 2:4].expand_as(position)), polar(separation),
            polar(alignment), polar(cohesion), rule,
            neighbor[:, mask].reshape(batch, vessels, -1)), -1).float()
        observations.append(observation)
    return dict(position=torch.stack(positions, 1), water=torch.stack(velocities, 1),
                command=torch.stack(controls, 1), observation=torch.stack(observations, 1))


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


def fleet_context_shape(version):
    """Vessel and global coordinates of the public intervention context."""
    return (15,5 if version>=5 else 4) if version>=3 else (8,3 if version>=2 else 0)


def fleet_set_cost(context,centers,context_version=1):
    """Compare public fleet states without padding or a fixed vessel count.

    A symmetric set distance retrieves candidate encounters. Exact injective
    matching below binds an observed control program to its physical vessels.
    The target velocity and clock remain global conditioning coordinates.
    """
    width,extra=fleet_context_shape(context_version)
    # The original public contexts are also used by the legacy role learner.
    if context_version==1 and context.shape[-1]%width==3:extra=3
    if (context.shape[-1]-extra)%width or (centers.shape[-1]-extra)%width:
        raise ValueError('Incompatible public fleet context versions.')
    n=(context.shape[-1]-extra)//width;m=(centers.shape[-1]-extra)//width
    own=context[:,:width*n].reshape(-1,n,width)
    other=centers[:,:width*m].reshape(-1,m,width)
    costs=(own[:,None,:,None]-other[None,:,None]).square().sum(-1)
    distance=.5*m*(costs.amin(-1).mean(-1)+costs.amin(-2).mean(-1))
    if extra:distance=distance+(context[:,None,-extra:]-centers[None,:,-extra:]).square().sum(-1)
    return distance,costs


def match_fleet_context(context,centers,count=4,context_version=1):
    """Transport whole intervention programs through one-to-one role matches."""
    from scipy.optimize import linear_sum_assignment
    coarse,costs=fleet_set_cost(context,centers,context_version)
    shortlist=coarse.topk(min(max(3*count,count),len(centers)),largest=False).indices
    b,_,n,m=costs.shape;_,extra=fleet_context_shape(context_version)
    if context_version==1 and context.shape[-1]%8==3:extra=3
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

    def forward(self,context,set_context=False,context_version=1):
        distances=(torch.cat([fleet_set_cost(context,part,context_version)[0]
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
    def __init__(self,width=48,paired=True,state=None,node_width=8,global_width=3,program_options=None,probabilistic=False,
                 peer_conditioned=True):
        super().__init__();self.width=width;self.paired=paired
        self.probabilistic=probabilistic
        self.node_width=node_width;self.global_width=global_width
        self.peer_conditioned=peer_conditioned
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
        factual_node=node
        if not self.peer_conditioned:
            if self.node_width!=8:raise ValueError('The no-peer decision control uses public eight-feature vessel states.')
            zero=F.one_hot(torch.zeros_like(programs),self.radix).to(rows.dtype)
            factual_node=self.node(torch.cat((rows,zero),-1))
        difference=rows[:,None]-rows[:,:,None]
        messages=self.message(torch.cat((node[:,:,None].expand(-1,-1,n,-1),
            factual_node[:,None].expand(-1,n,-1,-1),difference),-1))
        mask=~torch.eye(n,dtype=torch.bool,device=context.device)[None,:,:,None]
        average=(messages*mask).sum(2)/max(1,n-1)
        maximum=messages.masked_fill(~mask,-1e6).amax(2)
        encoded=self.update(torch.cat((node,average,maximum),-1))
        global_state=context[:,-extra:] if extra==self.global_width else context.new_zeros((len(context),self.global_width))
        if not self.peer_conditioned:
            # Each summand sees only its own choice and the public peer
            # states. No other vessel's choice can enter that summand.
            pooled=torch.cat((encoded,factual_node.mean(1)[:,None].expand(-1,n,-1),
                global_state[:,None].expand(-1,n,-1),context.new_full((len(context),n,1),n/6.)),-1)
            return self.readout(pooled).squeeze(-1).mean(-1)
        pooled=torch.cat((encoded.mean(1),encoded.amax(1),global_state,
            context.new_full((len(context),1),n/6.)),-1)
        raw=self.readout(pooled).squeeze(-1)
        return raw

    def forward(self,context,codes):
        raw=self.logits(context,codes)
        return raw.sigmoid() if self.probabilistic else raw


class PairedResponsePotential(SharedProgramPotential):
    """Conditional response estimator; ordinary return potentials stay unchanged."""
    def __init__(self,width=48,paired=False,state=None,node_width=8,global_width=3,
                 program_options=None,probabilistic=True,outcome_scale=0.,response_rank=0):
        super().__init__(width,paired,node_width=node_width,global_width=global_width,
                         program_options=program_options,probabilistic=probabilistic)
        self.outcome_scale=outcome_scale;self.response_rank=response_rank
        if response_rank:
            self.readout[-1]=nn.Linear(width,1+response_rank)
            with torch.no_grad():
                self.readout[-1].weight[1:]*=.1
                self.readout[-1].bias[1:]=1.
        if outcome_scale:
            import numpy as np
            points,weights=np.polynomial.hermite.hermgauss(15)
            self.register_buffer('outcome_points',torch.as_tensor(points*2.**.5,dtype=torch.float32),persistent=False)
            self.register_buffer('outcome_weights',torch.as_tensor(weights/np.pi**.5,dtype=torch.float32),persistent=False)
        if state is not None:self.load_state_dict(state)

    def response_parameters(self,context,codes):
        raw=super().logits(context,codes)
        return raw if self.response_rank else raw[:,None]

    def logits(self,context,codes):
        return self.response_parameters(context,codes)[:,0]

    def forward(self,context,codes):
        parameters=self.response_parameters(context,codes);raw=parameters[:,0]
        if self.outcome_scale:
            scale=(self.outcome_scale*parameters[:,1:].norm(dim=-1) if self.response_rank
                   else torch.full_like(raw,self.outcome_scale))
            return ((raw[:,None]+scale[:,None]*self.outcome_points).sigmoid()*self.outcome_weights).sum(-1)
        return raw.sigmoid() if self.probabilistic else raw


def load_program_potential(record):
    if record.get('kind')=='paired_factual_task':
        return PairedFactualTaskPotential(**{key:value for key,value in record.items() if key!='kind'}).eval()
    if record.get('kind')=='paired_future_task':
        return PairedFutureTaskPotential(**{key:value for key,value in record.items() if key!='kind'}).eval()
    if record.get('kind')=='calibrated_task_potential':
        return CalibratedTaskPotential(**{key:value for key,value in record.items() if key!='kind'}).eval()
    if record.get('kind')=='paired_latent_task':
        return PairedLatentTaskPotential(**{k:v for k,v in record.items() if k!='kind'}).eval()
    if record.get('kind')=='local_intervention_task':
        return LocalInterventionTaskPotential(**{key:value for key,value in record.items() if key!='kind'}).eval()
    if record.get('kind')=='paired_transport_task':
        return PairedTransportTaskPotential(**{key:value for key,value in record.items() if key!='kind'}).eval()
    if record.get('kind')=='paired_successor_task':
        return PairedSuccessorTaskPotential(**{key:value for key,value in record.items() if key!='kind'}).eval()
    estimator=(PairedResponsePotential if 'response_rank' in record or 'outcome_scale' in record
               else SharedProgramPotential)
    return estimator(**record).eval()


class LocalInterventionTaskPotential(nn.Module):
    """Sparse complete-program task values learned from local interventions."""
    def __init__(self,contexts,codes,values,observed,bandwidth=.15,program_options=None,time_bandwidth=None):
        super().__init__()
        self.radix=4 if program_options is None else len(program_options)
        self.bandwidth=bandwidth
        self.time_bandwidth=time_bandwidth
        self.register_buffer('contexts',torch.as_tensor(contexts,dtype=torch.float32))
        self.register_buffer('codes',torch.as_tensor(codes,dtype=torch.long))
        self.register_buffer('values',torch.as_tensor(values,dtype=torch.float32))
        self.register_buffer('observed',torch.as_tensor(observed,dtype=torch.bool))
        self.register_buffer('program_options',None if program_options is None else
                             torch.as_tensor(program_options,dtype=torch.float32))

    def forward(self,context,codes):
        distance=(context[:,None]-self.contexts[None]).square().sum(-1)
        if self.time_bandwidth is not None:
            distance=distance.masked_fill((context[:,None,-1]-self.contexts[None,:,-1]).abs()
                >self.time_bandwidth,float('inf'))
        nearest,index=distance.topk(min(4,len(self.contexts)),largest=False)
        column=torch.searchsorted(self.codes,codes).clamp_max(len(self.codes)-1)
        known=self.codes[column]==codes
        weight=torch.exp(-nearest/(2*self.bandwidth**2))
        weight=weight*self.observed[index,column[:,None]]*known[:,None]
        mass=weight.sum(-1)
        value=(weight*self.values[index,column[:,None]]).sum(-1)/mass.clamp_min(1e-12)
        # Unsupported programs never enter the maximization as invented
        # zero-success observations; they remain inadmissible alternatives.
        return value.masked_fill(mass<1e-6,-1.)


class CalibratedTaskPotential(nn.Module):
    """Keep paired task-value calibration while learning state-dependent changes."""
    def __init__(self,base,contexts,codes,residuals,observed,bandwidth=.15,time_bandwidth=None):
        super().__init__();self.base=load_program_potential(base);self.radix=self.base.radix
        self.bandwidth=bandwidth
        self.time_bandwidth=time_bandwidth
        for name,value,dtype in (('contexts',contexts,torch.float32),('codes',codes,torch.long),
                ('residuals',residuals,torch.float32),('observed',observed,torch.bool)):
            self.register_buffer(name,torch.as_tensor(value,dtype=dtype))

    @property
    def program_options(self):return self.base.program_options

    def forward(self,context,codes):
        value=self.base(context,codes)
        if context.shape[-1]!=self.contexts.shape[-1]:return value
        distance=(context[:,None]-self.contexts[None]).square().sum(-1)
        if self.time_bandwidth is not None:
            distance=distance.masked_fill((context[:,None,-1]-self.contexts[None,:,-1]).abs()
                >self.time_bandwidth,float('inf'))
        distance,index=distance.min(-1)
        column=torch.searchsorted(self.codes,codes).clamp_max(len(self.codes)-1)
        known=(self.codes[column]==codes)&self.observed[index,column]
        weight=torch.exp(-distance/(2*self.bandwidth**2))*known
        return (value+weight*self.residuals[index,column]).clamp(0.,1.)


def calibrate_program_potential(record,estimates,bandwidth=.15,time_bandwidth=None):
    """Fit local value residuals, never a selected-action lookup or new label."""
    import numpy as np
    base=load_program_potential(record)
    codes=sorted({code for example in estimates for code in example['programs']})
    columns={code:i for i,code in enumerate(codes)}
    residuals=np.zeros((len(estimates),len(codes)),np.float32);observed=np.zeros_like(residuals,dtype=bool)
    with torch.no_grad():
        for i,example in enumerate(estimates):
            q=torch.as_tensor(example['programs'],dtype=torch.long)
            x=torch.as_tensor(example['context'],dtype=torch.float32).expand(len(q),-1)
            prediction=base(x,q).numpy();indices=[columns[code] for code in example['programs']]
            residuals[i,indices]=example['targets']-prediction;observed[i,indices]=True
    calibrated=dict(kind='calibrated_task_potential',base=record,
        contexts=np.asarray([example['context'] for example in estimates]),codes=codes,
        residuals=residuals,observed=observed,bandwidth=bandwidth,time_bandwidth=time_bandwidth)
    return load_program_potential(calibrated),calibrated


def fit_intervention_task_potential(examples,*,conditional=True,difference_weight=4.,
                                   anchor_weight=.25,bandwidth=.15,program_options=None):
    """Fit the deployed task potential, including paired terminal differences.

    Nodes count distinct program/future observations once. Local edges keep
    their paired futures, which can differ from the futures of other edges.
    Both objectives share the complete-program representation and node data.
    """
    import numpy as np
    codes=sorted({sample[0] for example in examples for sample in example['samples'].values()})
    columns={q:i for i,q in enumerate(codes)}
    values=np.zeros((len(examples),len(codes)),np.float32)
    observed=np.zeros_like(values,dtype=bool)
    edge_count=0;informative=0;residual=[];node_error=[]
    for root,example in enumerate(examples):
        groups={};edges={};seen=set()
        for code,_,outcome in example['samples'].values():groups.setdefault(code,[]).append(outcome)
        programs=sorted(groups);lookup={q:i for i,q in enumerate(programs)}
        target=np.asarray([np.mean(groups[q]) for q in programs])
        factual=float(np.mean(groups[0]))
        matrix=(1.+anchor_weight)*np.eye(len(programs));rhs=target+anchor_weight*factual
        for a,b in example['pairs']:
            if (a,b) in seen:continue
            seen.add((a,b))
            qa,_,ya=example['samples'][a];qb,_,yb=example['samples'][b]
            edges.setdefault((qa,qb),[]).append(yb-ya)
        for (a,b),outcomes in edges.items():
            edge_count+=len(outcomes);informative+=int(np.count_nonzero(outcomes))
            if not conditional:continue
            delta=float(np.mean(outcomes));weight=difference_weight*(1.+4.*abs(delta))
            i,j=lookup[a],lookup[b]
            matrix[i,i]+=weight;matrix[j,j]+=weight;matrix[i,j]-=weight;matrix[j,i]-=weight
            rhs[i]-=weight*delta;rhs[j]+=weight*delta
        fitted=fit_bounded_graph_values(matrix,rhs)
        destination=[columns[q] for q in programs]
        values[root,destination]=fitted;observed[root,destination]=True
        node_error.extend((fitted-target)**2)
        residual.extend((fitted[lookup[b]]-fitted[lookup[a]]-np.mean(y))**2
                        for (a,b),y in edges.items())
    record=dict(kind='local_intervention_task',contexts=np.asarray([e['context'] for e in examples]),
        codes=codes,values=values,observed=observed,bandwidth=bandwidth,program_options=program_options)
    fit=dict(roots=len(examples),nodes=int(observed.sum()),terminal_difference_labels=edge_count if conditional else 0,
        informative_terminal_pairs=informative,conditional=conditional,difference_weight=difference_weight if conditional else 0.,
        node_mse=float(np.mean(node_error)),paired_difference_mse=float(np.mean(residual)),
        task_target='true_terminal_task_success',deployed_parameters=int(observed.sum()))
    return load_program_potential(record),record,fit


def fit_future_conditioned_task_potential(examples,*,bandwidth=.15,program_options=None,
                                        contrast_weight=.25,future_precision=.5,anchor_weight=.25,
                                        conditional_ordering=False,residual_calibration=False):
    """Fit paired task consequences before marginalizing the shared future.

    A nuisance offset is shared by all programs measured under one future.
    Local intervention differences constrain probabilities under that SAME
    future. Deployment contains only the marginalized complete-program task
    values, with no future identifier or selected-control target.
    """
    import numpy as np
    from scipy.optimize import minimize
    from scipy.special import expit
    codes=sorted({sample[0] for example in examples for sample in example['samples'].values()})
    columns={q:i for i,q in enumerate(codes)}
    values=np.zeros((len(examples),len(codes)),np.float32);observed=np.zeros_like(values,dtype=bool)
    fits=[]
    for root,example in enumerate(examples):
        programs=sorted({sample[0] for sample in example['samples'].values()})
        lookup={q:i for i,q in enumerate(programs)}
        futures=list(dict.fromkeys(key[1] for key in example['samples']))
        future_index={future:i for i,future in enumerate(futures)}
        keys=list(example['samples']);sample_index={key:i for i,key in enumerate(keys)}
        q=np.asarray([lookup[key[0]] for key in keys]);f=np.asarray([future_index[key[1]] for key in keys])
        y=np.asarray([example['samples'][key][2] for key in keys],dtype=float)
        pairs=list(dict.fromkeys(example['pairs']))
        left=np.asarray([sample_index[a] for a,_ in pairs]);right=np.asarray([sample_index[b] for _,b in pairs])
        if not np.array_equal(f[left],f[right]):raise ValueError('A task contrast must share its continuation future.')
        discordant=y[left]!=y[right]
        ordering_left=q[left[discordant]];ordering_right=q[right[discordant]]
        ordering_sign=y[right[discordant]]-y[left[discordant]]
        count=np.bincount(q,minlength=len(programs));success=np.bincount(q,weights=y,minlength=len(programs))
        factual=float(y[q==lookup[0]].mean());mean=(success+anchor_weight*factual)/(count+anchor_weight)
        initial=np.r_[np.log(np.clip(mean,.01,.99)/np.clip(1.-mean,.01,.99)),np.zeros(len(futures))]
        weight=contrast_weight*len(y)/max(1,int(discordant.sum()) if conditional_ordering else len(pairs));n=len(programs)
        def objective(parameter):
            a=parameter[:n];b=parameter[n:]-parameter[n:].mean();logit=a[q]+b[f];probability=expit(logit)
            value=np.sum(np.logaddexp(0.,logit)-y*logit)
            gradient=probability-y
            if not conditional_ordering:
                residual=probability[right]-probability[left]-(y[right]-y[left])
                value+=weight*np.square(residual).sum()
                slope=probability*(1.-probability)
                np.add.at(gradient,left,-2.*weight*residual*slope[left])
                np.add.at(gradient,right,2.*weight*residual*slope[right])
            ga=np.bincount(q,weights=gradient,minlength=n)
            gb=np.bincount(f,weights=gradient,minlength=len(futures))
            if conditional_ordering:
                # Conditioning on discordant paired outcomes cancels the
                # shared future offset. Concordant outcomes still inform
                # absolute task values; they are not labels of equal value.
                difference=a[ordering_right]-a[ordering_left]
                signed=ordering_sign*difference
                value+=weight*np.logaddexp(0.,-signed).sum()
                derivative=-weight*ordering_sign*expit(-signed)
                np.add.at(ga,ordering_left,-derivative)
                np.add.at(ga,ordering_right,derivative)
            value+=anchor_weight*np.sum(np.logaddexp(0.,a)-factual*a)+.0005*np.square(a).sum()
            ga+=anchor_weight*(expit(a)-factual)+.001*a
            value+=.5*future_precision*np.square(b).sum();gb+=future_precision*b
            return float(value),np.r_[ga,gb-gb.mean()]
        result=minimize(objective,initial,jac=True,method='L-BFGS-B',
            bounds=[(-12.,12.)]*len(initial),options=dict(maxiter=1000,ftol=1e-11,gtol=1e-7))
        if not result.success:raise RuntimeError(f'Paired task-value fit failed: {result.message}')
        a=result.x[:n];b=result.x[n:]-result.x[n:].mean()
        marginalized=expit(a[:,None]+b[None]).mean(-1)
        if residual_calibration=='complete':
            complete=count==len(futures)
            marginalized[complete]=success[complete]/count[complete]
        elif residual_calibration:
            # Standardize to the common future population, then correct the
            # fitted response with its observed outcome residual. When all
            # futures are observed this recovers their empirical task value
            # exactly; a ranking objective cannot distort that known mean.
            residual=np.bincount(q,weights=y-expit(a[q]+b[f]),minlength=n)/count
            marginalized=np.clip(marginalized+residual,0.,1.)
        destination=[columns[code] for code in programs]
        values[root,destination]=marginalized;observed[root,destination]=True
        fits.append(dict(unique_outcomes=len(y),paired_outcomes=len(pairs),future_conditions=len(futures),
            discordant_pairs=int(discordant.sum()),objective=float(result.fun),future_offset_range=float(np.ptp(b))))
    record=dict(kind='local_intervention_task',contexts=np.asarray([e['context'] for e in examples]),
        codes=codes,values=values,observed=observed,bandwidth=bandwidth,program_options=program_options)
    return load_program_potential(record),record,dict(conditional=True,roots=len(examples),
        terminal_difference_labels=sum(item['paired_outcomes'] for item in fits),
        task_target='future_conditioned_terminal_success_then_marginalization',
        contrast_weight=contrast_weight,future_precision=future_precision,conditional_ordering=conditional_ordering,
        residual_calibration=residual_calibration,
        deployed_parameters=int(observed.sum()),fits=fits)


def paired_block_task_estimates(examples,covariance_shrink=.9):
    """Calibrate whole-plan values with paired, shared-outcome observations.

    A program/future can occur on several local edges. Fractional edge weights
    allocate at most one observation's mass to it; any unused mass supplies
    its marginal outcome term. The two endpoints are fitted jointly, so their
    absolute results and relative consequence share one covariance model.
    """
    import numpy as np
    estimates=[];changes=[];nonredundant=0;edge_count=0;paired_samples=0;unique_count=0;shared_count=0
    for example in examples:
        groups={};edges={};seen=set();degree={key:0 for key in example['samples']}
        for key,(code,_,outcome) in example['samples'].items():groups.setdefault(code,[]).append(outcome)
        for a,b in example['pairs']:
            if (a,b) in seen:continue
            if a[1]!=b[1]:raise ValueError('Paired outcomes must share their continuation future.')
            seen.add((a,b));degree[a]+=1;degree[b]+=1
            qa,_,ya=example['samples'][a];qb,_,yb=example['samples'][b]
            edges.setdefault((qa,qb),[]).append((a,b,ya,yb))
        programs=sorted(groups);lookup={q:i for i,q in enumerate(programs)}
        raw=np.asarray([np.mean(groups[q]) for q in programs])
        normal=1e-6*np.eye(len(programs));rhs=1e-6*raw
        counts=np.asarray([len(groups[q]) for q in programs])
        probability=(raw*counts+1.)/(counts+2.)
        variance=probability*(1.-probability)
        used={key:0. for key in example['samples']};score_blocks={}
        for (qa,qb),outcomes in edges.items():
            outcomes_array=np.asarray([row[2:] for row in outcomes]);count=len(outcomes);mean=outcomes_array.mean(0)
            indices=[lookup[qa],lookup[qb]];i,j=indices
            covariance=covariance_shrink*np.mean((outcomes_array[:,0]-probability[i])*
                (outcomes_array[:,1]-probability[j]))
            limit=.95*np.sqrt(variance[i]*variance[j]);covariance=np.clip(covariance,-limit,limit)
            precision=np.linalg.inv(np.asarray([[variance[i],covariance],[covariance,variance[j]]]))
            for a,b,ya,yb in outcomes:
                weight=1./max(degree[a],degree[b]);used[a]+=weight;used[b]+=weight
                block=weight*precision;target=np.asarray([ya,yb])
                normal[np.ix_(indices,indices)]+=block;rhs[indices]+=block@target
                score_blocks.setdefault(a[1],[]).append((indices,block,target))
            nonredundant+=abs(mean[1]-mean[0]-(raw[indices[1]]-raw[indices[0]]))>1e-6
            edge_count+=1;paired_samples+=count
        for key,(code,_,outcome) in example['samples'].items():
            index=lookup[code];weight=max(0.,1.-used[key])/variance[index]
            normal[index,index]+=weight;rhs[index]+=weight*outcome
            if weight:score_blocks.setdefault(key[1],[]).append(([index],np.asarray([[weight]]),np.asarray([outcome])))
        fitted=fit_bounded_graph_values(normal,rhs)
        # Cluster the score by shared future. Repeated graph incidences must
        # not be mistaken for independent training continuations.
        scores=[]
        for blocks in score_blocks.values():
            score=np.zeros(len(programs))
            for indices,precision,target in blocks:score[indices]+=precision@(target-fitted[indices])
            scores.append(score)
        score=np.asarray(scores);inverse=np.linalg.inv(normal)
        covariance=inverse@(score.T@score)@inverse*len(scores)/max(1,len(scores)-1)
        factual=lookup[0]
        gain_variance=np.maximum(0.,np.diag(covariance)+covariance[factual,factual]-2.*covariance[:,factual])
        changes.extend((fitted-raw).tolist())
        unique_count+=len(example['samples']);shared_count+=sum(count>1 for count in degree.values())
        estimates.append(dict(context=example['context'],programs=programs,targets=fitted,gain_variance=gain_variance,
            edges=[(lookup[a],lookup[b],float(np.diff(np.asarray([row[2:] for row in y]).mean(0))[0]))
                for (a,b),y in edges.items()]))
    return estimates,dict(paired_future_blocks=edge_count,paired_terminal_observations=paired_samples,
        unique_terminal_observations=unique_count,shared_graph_observations=shared_count,
        nonredundant_block_fraction=float(nonredundant/max(1,edge_count)),
        mean_absolute_value_correction=float(np.mean(np.abs(changes))),covariance_shrink=covariance_shrink,
        estimation_objective='joint paired outcomes with fractional shared-observation mass and future-clustered uncertainty')


def matched_future_task_estimates(examples,covariance_shrink=.9):
    """Estimate task values from overlapping, jointly observed future blocks.

    A future contributes one vector of outcomes. Small groups of intervention
    edges can therefore share a factual rollout without duplicating its
    likelihood or discarding correlations among the resulting contrasts.
    Different groups retain independent futures and overlapping programs;
    their task means are fitted to one common set of marginal values.
    """
    import numpy as np
    estimates=[];changes=[];blocks_total=0;outcomes_total=0;futures_total=0;nonredundant=0;pairs_total=0
    for example in examples:
        programs=sorted({q for q,_,_ in example['samples'].values()});lookup={q:i for i,q in enumerate(programs)}
        by_program={q:[] for q in programs};by_future={}
        for (q,future),(_,_,outcome) in example['samples'].items():
            by_program[q].append(outcome);by_future.setdefault(future,{})[lookup[q]]=outcome
        raw=np.asarray([np.mean(by_program[q]) for q in programs])
        probability=np.asarray([(sum(by_program[q])+1.)/(len(by_program[q])+2.) for q in programs])
        variance=probability*(1.-probability);groups={}
        for values in by_future.values():
            indices=tuple(sorted(values));groups.setdefault(indices,[]).append([values[i] for i in indices])
        normal=1e-6*np.eye(len(programs));rhs=1e-6*raw;score_blocks=[]
        for indices,observations in groups.items():
            indices=np.asarray(indices);y=np.asarray(observations);count,width=y.shape
            centered=y-y.mean(0);empirical=centered.T@centered/max(1,count-1)
            scale=np.sqrt(np.maximum(np.diag(empirical),variance[indices]))
            correlation=empirical/(scale[:,None]*scale[None,:])
            correlation[np.diag_indices(width)]+=1.-np.diag(correlation)
            shrink=min(covariance_shrink,count/(count+width))
            correlation=shrink*correlation+(1.-shrink)*np.eye(width)
            sigma=np.sqrt(variance[indices]);covariance=correlation*sigma[:,None]*sigma[None,:]
            precision=np.linalg.inv(covariance)
            normal[np.ix_(indices,indices)]+=count*precision;rhs[indices]+=precision@y.sum(0)
            score_blocks.append((indices,precision,y))
            nonredundant+=np.max(np.abs(y.mean(0)-raw[indices]))>1e-6
            blocks_total+=1
        fitted=fit_bounded_graph_values(normal,rhs)
        meat=np.zeros_like(normal)
        for indices,precision,y in score_blocks:
            scores=(y-fitted[indices])@precision
            meat[np.ix_(indices,indices)]+=scores.T@scores
        inverse=np.linalg.inv(normal);count=len(by_future)
        covariance=inverse@meat@inverse*count/max(1,count-1)
        factual=lookup[0]
        gain_variance=np.maximum(0.,np.diag(covariance)+covariance[factual,factual]-2.*covariance[:,factual])
        edges={}
        for left,right in example['pairs']:
            edges.setdefault((left[0],right[0]),[]).append(example['samples'][right][2]-example['samples'][left][2])
        estimates.append(dict(context=example['context'],programs=programs,targets=fitted,gain_variance=gain_variance,
            edges=[(lookup[a],lookup[b],float(np.mean(values))) for (a,b),values in edges.items()]))
        changes.extend((fitted-raw).tolist());outcomes_total+=len(example['samples'])
        futures_total+=count;pairs_total+=len(example['pairs'])
    return estimates,dict(paired_future_blocks=blocks_total,unique_terminal_observations=outcomes_total,
        independent_futures=futures_total,paired_terminal_observations=pairs_total,
        nonredundant_block_fraction=float(nonredundant/max(1,blocks_total)),
        mean_absolute_value_correction=float(np.mean(np.abs(changes))),covariance_shrink=covariance_shrink,
        estimation_objective='one joint outcome vector per future with overlapping block task-value estimation')


def successor_standardized_task_estimates(examples,rank=3,ridge=8.,task_conditioned=False):
    """Use paired factual motion as a cross-fitted task-outcome control variate.

    A factual failure bit may be constant even though its observed successor
    distinguishes easy and difficult futures. Regress actual terminal success
    on that shared response, then standardize every complete program to the
    same empirical future distribution. Held-out residuals retain its observed
    task mean if the response regression is unhelpful. No geometric cost or
    successful-action target replaces the terminal task outcome.
    """
    import numpy as np
    estimates=[];changes=[];matched=0;observations=0
    for example in examples:
        groups={}
        for (q,future),(_,successor,outcome) in example['samples'].items():
            groups.setdefault(q,{})[future]=(successor,float(outcome))
        programs=sorted(groups);lookup={q:i for i,q in enumerate(programs)}
        raw=np.asarray([np.mean([value[1] for value in groups[q].values()]) for q in programs])
        pseudo={q:{future:value[1] for future,value in values.items()} for q,values in groups.items()}
        factual=groups[0];futures=list(factual);future_index={future:i for i,future in enumerate(futures)}
        if task_conditioned and any(future not in factual for values in groups.values() for future in values):
            raise ValueError('Task-conditioned standardization needs the actual factual outcome for every sampled future.')
        if len(futures)>=4:
            state=np.asarray([factual[future][0] for future in futures],dtype=float)
            centered=state-state.mean(0);centered/=np.maximum(centered.std(0),.01)
            _,singular,directions=np.linalg.svd(centered,full_matrices=False)
            dimension=min(rank,len(futures)-2,int((singular>1e-6).sum()))
            z=centered@directions[:dimension].T/(singular[:dimension]/np.sqrt(len(futures)-1))
            if task_conditioned:
                # The paired factual terminal result is an offline measure
                # of continuation difficulty beyond the short motion window.
                # Center it over ALL factual futures before estimating each
                # complete program's task response; never expose it online.
                outcome=np.asarray([factual[future][1] for future in futures])
                z=np.column_stack((z,(outcome-outcome.mean())/max(outcome.std(),.1)))
                dimension=z.shape[1]
            if dimension:
                permutation=np.random.default_rng(19081987).permutation(len(futures))
                fold=np.zeros(len(futures),dtype=int);fold[permutation[len(futures)//2:]]=1
                for heldout in (0,1):
                    reference=z[fold==heldout].mean(0)
                    normal=ridge*np.eye(dimension);rhs=np.zeros(dimension);local={}
                    for q,values in groups.items():
                        keys=[future for future in values if future in future_index
                            and fold[future_index[future]]!=heldout]
                        if len(keys)<2:continue
                        features=z[[future_index[future] for future in keys]]
                        outcome=np.asarray([values[future][1] for future in keys])
                        features=features-features.mean(0);outcome=outcome-outcome.mean()
                        gram=features.T@features;cross=features.T@outcome
                        local[q]=(gram,cross);normal+=gram;rhs+=cross
                    shared=np.linalg.solve(normal,rhs)
                    for q,values in groups.items():
                        gram,cross=local.get(q,(np.zeros((dimension,dimension)),np.zeros(dimension)))
                        coefficient=np.linalg.solve(gram+ridge*np.eye(dimension),cross+ridge*shared)
                        for future in values:
                            index=future_index.get(future)
                            if index is None or fold[index]!=heldout:continue
                            pseudo[q][future]+=float(coefficient@(reference-z[index]));matched+=1
        target=np.asarray([np.mean(list(pseudo[q].values())) for q in programs]).clip(0.,1.)
        variance=[];factual_values=pseudo[0];n0=len(factual_values)
        v0=np.var(list(factual_values.values()),ddof=1)/n0 if n0>1 else .25
        for q in programs:
            values=pseudo[q];count=len(values);common=list(values.keys()&factual_values.keys())
            own=np.var(list(values.values()),ddof=1)/count if count>1 else .25
            covariance=(np.cov([values[key] for key in common],[factual_values[key] for key in common],ddof=1)[0,1]
                if len(common)>1 else 0.)
            variance.append(max(0.,own+v0-2.*len(common)*covariance/(count*n0)))
        edges={}
        for left,right in example['pairs']:
            edges.setdefault((left[0],right[0]),[]).append(example['samples'][right][2]-example['samples'][left][2])
        estimates.append(dict(context=example['context'],programs=programs,targets=target,
            gain_variance=np.asarray(variance),edges=[(lookup[a],lookup[b],float(np.mean(values)))
                for (a,b),values in edges.items()]))
        changes.extend((target-raw).tolist());observations+=len(example['samples'])
    return estimates,dict(unique_terminal_observations=observations,matched_factual_responses=matched,
        mean_absolute_value_correction=float(np.mean(np.abs(changes))),response_rank=rank,
        cross_fit_folds=2,response_ridge=ridge,duplicate_contrast_labels=0,
        factual_terminal_condition=bool(task_conditioned),
        estimation_objective='cross-fitted terminal task regression standardized over paired factual successors')


def matched_duration_task_estimates(examples,physical_contexts,spans,covariance_shrink=.9,shared_factual=False):
    """Estimate complete program/window values jointly under matched futures.

    Duration is part of the intervention, not another independent root.
    The same future therefore contributes one outcome vector across ALL
    sampled windows. Each physical continuation appears exactly once;
    neither a second mean-difference target nor an assumed motion cost
    supplies additional task evidence.
    """
    import numpy as np
    groups={}
    for index,context in enumerate(physical_contexts):
        groups.setdefault(np.asarray(context,dtype=np.float32).tobytes(),[]).append(index)
    combined=[];mappings=[]
    for indices in groups.values():
        def identity(q,span):return (q,0 if shared_factual and q==0 else span)
        # With no inherited value policy, a completely free program follows
        # the identical predecessor at every tick, for every window length.
        # Pool that observed factual control, not the outcomes of interventions.
        arms=sorted({identity(q,spans[index]) for index in indices
            for q,_,_ in examples[index]['samples'].values()})
        lookup={arm:i for i,arm in enumerate(arms)};samples={};pairs=[]
        for index in indices:
            example=examples[index];span=spans[index]
            for (q,future),(_,successor,outcome) in example['samples'].items():
                arm=lookup[identity(q,span)];key=(arm,future)
                if key in samples and samples[key][2]!=outcome:
                    raise ValueError('One complete program/window/future has inconsistent task outcomes.')
                samples[key]=(arm,successor,outcome)
            pairs.extend(((lookup[identity(left[0],span)],left[1]),(lookup[identity(right[0],span)],right[1]))
                for left,right in example['pairs'])
        combined.append(dict(context=np.asarray(physical_contexts[indices[0]]),samples=samples,pairs=pairs))
        mappings.append((indices,lookup))
    joint,statistics=matched_future_task_estimates(combined,covariance_shrink)
    estimates=[None]*len(examples)
    for fitted,(indices,lookup) in zip(joint,mappings):
        target=dict(zip(fitted['programs'],fitted['targets']))
        variance=dict(zip(fitted['programs'],fitted['gain_variance']))
        for index in indices:
            example=examples[index];span=spans[index]
            programs=sorted({q for q,_,_ in example['samples'].values()})
            positions={q:i for i,q in enumerate(programs)};edges={}
            for left,right in example['pairs']:
                edges.setdefault((left[0],right[0]),[]).append(
                    example['samples'][right][2]-example['samples'][left][2])
            estimates[index]=dict(context=example['context'],programs=programs,
                targets=np.asarray([target[lookup[identity(q,span)]] for q in programs]),
                gain_variance=np.asarray([variance[lookup[identity(q,span)]] for q in programs]),
                edges=[(positions[a],positions[b],float(np.mean(values))) for (a,b),values in edges.items()])
    statistics.update(joint_duration_roots=len(groups),candidate_durations=sorted(set(spans)),
        shared_factual_continuation=shared_factual,
        estimation_objective='joint complete program/window task values from one outcome vector per future')
    return estimates,statistics


def paired_task_refinement_programs(examples,max_programs=3,margin=.05,standard_errors=1.,action_radix=4):
    """Allocate paired rollouts to unresolved task-value rankings.

    A high empirical return alone does not request more samples. The priority
    uses the estimated intervention gain and its paired uncertainty, retaining
    competitors whose upper estimate can still beat the current lower one.
    The collector expands these programs into their single-vessel paths.
    """
    import numpy as np
    estimates,_=paired_block_task_estimates(examples);requests=[]
    for example in estimates:
        programs=np.asarray(example['programs']);factual=int(np.flatnonzero(programs==0)[0])
        n=(len(example['context'])-3)//8
        changed=sum((programs//action_radix**vessel)%action_radix>0 for vessel in range(n))
        gain=example['targets']-example['targets'][factual]-.005*changed
        deviation=np.sqrt(example['gain_variance'])
        lower=gain-standard_errors*deviation;upper=gain+standard_errors*deviation
        competitor=(programs!=0)&(upper>max(margin,float(lower.max())))
        priority=np.where(competitor,deviation/(np.maximum(0.,gain.max()-gain)+margin),0.)
        selected=[int(i) for i in np.argsort(-priority)[:max_programs] if priority[i]>0.]
        requests.append(dict(context=example['context'],program_codes=[0]+[int(programs[i]) for i in selected],
            estimated_gains=[float(gain[i]) for i in selected],
            paired_standard_errors=[float(deviation[i]) for i in selected]))
    return requests


@torch.no_grad()
def task_value_program_expansion(record,examples,max_programs=8,parents=4):
    """Propose unobserved complete programs by learned conditional task value.

    A one-vessel edit is applied to a promising COMPLETE program, allowing
    higher-order interventions to grow beyond the initial single/pair set.
    Proposals are sampling requests only: they cannot be deployed until
    their paired physical task outcomes have been collected and fitted.
    """
    network=load_program_potential(record).cpu().eval();radix=network.radix;requests=[]
    for example in examples:
        context=torch.as_tensor(example['context'],dtype=torch.float32)
        n=(len(context)-network.global_width)//network.node_width
        observed=sorted({q for q,_,_ in example['samples'].values()})
        codes=torch.as_tensor(observed,dtype=torch.long)
        digits=(codes[:,None]//(radix**torch.arange(n)))%radix
        score=network(context[None].expand(len(codes),-1),codes)-.005*(digits>0).sum(-1)
        seed_codes=codes[score.argsort(descending=True)[:parents]].tolist()
        candidates=set();known=set(observed)
        for code in seed_codes:
            for vessel in range(n):
                place=radix**vessel;current=(code//place)%radix
                candidates.update(code+(choice-current)*place for choice in range(radix))
        candidates=sorted(candidates-known)
        if not candidates:
            requests.append(dict(program_codes=[],predicted_task_values=[],parent_codes=seed_codes));continue
        codes=torch.as_tensor(candidates,dtype=torch.long)
        value=network(context[None].expand(len(codes),-1),codes)
        digits=(codes[:,None]//(radix**torch.arange(n)))%radix
        score=value-.005*(digits>0).sum(-1)
        selected=score.argsort(descending=True)[:max_programs]
        requests.append(dict(program_codes=codes[selected].tolist(),
            predicted_task_values=value[selected].tolist(),parent_codes=seed_codes))
    return requests


class PairedFutureTaskPotential(PairedResponsePotential):
    """Marginalize program-dependent responses to shared paired futures."""
    def __init__(self,width=64,response_rank=2,contexts=None,future_counts=None,
                 global_width=3,program_options=None,state=None,time_bandwidth=None):
        super().__init__(width,False,node_width=8,global_width=global_width,
            program_options=program_options,probabilistic=True,response_rank=response_rank)
        self.register_buffer('contexts',torch.as_tensor(contexts,dtype=torch.float32))
        self.time_bandwidth=time_bandwidth
        counts=torch.as_tensor(future_counts,dtype=torch.long)
        self.register_buffer('future_mask',torch.arange(int(counts.max()))[None]<counts[:,None])
        self.future_latents=nn.Parameter(.05*torch.randn(len(counts),int(counts.max()),response_rank))
        if state is not None:self.load_state_dict(state)

    def centered_futures(self):
        mask=self.future_mask[...,None]
        mean=(self.future_latents*mask).sum(1,keepdim=True)/mask.sum(1,keepdim=True)
        return self.future_latents-mean

    def conditional_logits(self,context,codes,roots,futures):
        parameters=self.response_parameters(context,codes)
        latent=self.centered_futures()[roots,futures]
        return parameters[:,0]+(parameters[:,1:]*latent).sum(-1)

    def forward(self,context,codes):
        if context.shape[-1]==self.contexts.shape[-1]:
            distance=(context[:,None]-self.contexts[None]).square().sum(-1)
        else:
            distance=(SuccessorTaskDecision.summary(context)[:,None]
                -SuccessorTaskDecision.summary(self.contexts)[None]).square().sum(-1)
        if self.time_bandwidth is not None:
            distance=distance.masked_fill((context[:,None,-1]-self.contexts[None,:,-1]).abs()
                >self.time_bandwidth,float('inf'))
        roots=distance.argmin(-1)
        parameters=self.response_parameters(context,codes)
        logits=parameters[:,:1]+(parameters[:,None,1:]*self.centered_futures()[roots]).sum(-1)
        mask=self.future_mask[roots]
        return (logits.sigmoid()*mask).sum(-1)/mask.sum(-1)


def fit_paired_future_task_potential(examples,*,program_options=None,global_width=3,
        width=64,response_rank=2,updates=3000,seed=42,device='cpu',difference_weight=1.):
    """Learn conditional task ordering without equating concordance to a tie.

    Each program/future outcome enters the absolute objective once. Local
    discordant pairs train their relative conditional logits. Future factors
    are shared across paired arms and integrated out before policy selection.
    """
    import numpy as np
    contexts=[];codes=[];roots=[];futures=[];outcomes=[];pairs=[];future_counts=[]
    for root,example in enumerate(examples):
        future_keys=list(dict.fromkeys(key[1] for key in example['samples']))
        lookup={key:i for i,key in enumerate(future_keys)};future_counts.append(len(lookup));indices={}
        for key,(code,_,outcome) in example['samples'].items():
            indices[key]=len(codes);contexts.append(example['context']);codes.append(code)
            roots.append(root);futures.append(lookup[key[1]]);outcomes.append(outcome)
        for a,b in dict.fromkeys(example['pairs']):
            if a[1]!=b[1]:raise ValueError('Conditional task ordering requires a shared paired future.')
            if example['samples'][a][2]!=example['samples'][b][2]:pairs.append((indices[a],indices[b]))
    torch.manual_seed(seed);rng=np.random.default_rng(seed+19081987)
    net=PairedFutureTaskPotential(width,response_rank,np.asarray([e['context'] for e in examples]),
        future_counts,global_width,program_options).to(device)
    x=torch.as_tensor(np.asarray(contexts),dtype=torch.float32,device=device)
    q=torch.as_tensor(codes,dtype=torch.long,device=device)
    root=torch.as_tensor(roots,dtype=torch.long,device=device)
    future=torch.as_tensor(futures,dtype=torch.long,device=device)
    y=torch.as_tensor(outcomes,dtype=torch.float32,device=device)
    pair=torch.as_tensor(pairs,dtype=torch.long,device=device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=.005);last={}
    with torch.enable_grad():
        for step in range(updates):
            selected=torch.as_tensor(rng.integers(len(q),size=256),device=device)
            absolute=F.binary_cross_entropy_with_logits(
                net.conditional_logits(x[selected],q[selected],root[selected],future[selected]),y[selected])
            ordering=absolute.new_zeros(())
            if len(pairs):
                chosen=torch.as_tensor(rng.integers(len(pair),size=128),device=device)
                a,b=pair[chosen].unbind(-1);indices=torch.cat((a,b))
                logits=net.conditional_logits(x[indices],q[indices],root[indices],future[indices])
                ordering=F.softplus(-(y[b]-y[a])*(logits[128:]-logits[:128])).mean()
            regularizer=.002*net.centered_futures().square().mean()
            loss=absolute+difference_weight*ordering+regularizer
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            if (step+1)%500==0 or step+1==updates:
                last=dict(phase='paired_future_task_fit',update=step+1,
                    absolute_task_loss=float(absolute.detach()),conditional_order_loss=float(ordering.detach()))
                print(last,flush=True)
    record=dict(kind='paired_future_task',width=width,response_rank=response_rank,
        contexts=np.asarray([e['context'] for e in examples]),future_counts=future_counts,
        global_width=global_width,program_options=program_options,
        state={key:value.detach().cpu() for key,value in net.state_dict().items()})
    return net.cpu().eval(),record,dict(roots=len(examples),unique_task_outcomes=len(q),
        terminal_difference_labels=sum(len(e['pairs']) for e in examples),discordant_pairs=len(pairs),
        task_target='conditional terminal success marginalized over shared futures',
        deployed_parameters=sum(p.numel() for p in net.parameters()),updates=updates,
        difference_weight=difference_weight,last=last)


def future_task_likelihood_factors(logits,target,mask):
    """Exact count and conditional assignment log-likelihoods for one future."""
    count=(target*mask).sum(-1).long();length=mask.sum(-1)
    complement=count>length//2
    logits=torch.where(complement[:,None],-logits,logits)
    target=torch.where(complement[:,None],mask.to(target.dtype)-target,target)*mask
    count=torch.minimum(count,length-count);degree=int(count.max())
    polynomial=logits.new_full((len(logits),degree+1),-1e4);polynomial[:,0]=0.
    for column in range(logits.shape[1]):
        shifted=torch.cat((polynomial.new_full((len(logits),1),-1e4),
            polynomial[:,:-1]+logits[:,column,None]),-1)
        polynomial=torch.where(mask[:,column,None],torch.logaddexp(polynomial,shifted),polynomial)
    normalizer=polynomial.gather(1,count[:,None]).squeeze(1)
    ordering=normalizer-(target*logits).sum(-1)
    cardinality=(F.softplus(logits)*mask).sum(-1)-normalizer
    return cardinality,ordering


def fit_joint_future_task_potential(examples,*,program_options=None,global_width=3,
        width=61,response_rank=2,updates=5000,seed=42,device='cpu',difference_weight=2.):
    """Factor each future's task vector into cardinality and winning programs.

    Conditioning on the observed success count compares WHICH complete
    programs succeed under a common future. Cardinality retains the absolute
    task-success evidence. At unit ordering weight the two factors recover
    the Bernoulli vector likelihood exactly; no endpoint or contrast is
    presented as a second independent observation. Shared latent futures are
    marginalized before deployment, never supplied as online information.
    """
    import numpy as np
    contexts=[];codes=[];roots=[];futures=[];outcomes=[];blocks=[];future_counts=[]
    for root,example in enumerate(examples):
        by_future={}
        for key,sample in example['samples'].items():by_future.setdefault(key[1],[]).append(sample)
        future_counts.append(len(by_future))
        for future,samples in enumerate(by_future.values()):
            block=[]
            for code,_,outcome in samples:
                block.append(len(codes));contexts.append(example['context']);codes.append(code)
                roots.append(root);futures.append(future);outcomes.append(outcome)
            blocks.append(block)
    torch.manual_seed(seed);rng=np.random.default_rng(seed+19081987)
    net=PairedFutureTaskPotential(width,response_rank,np.asarray([e['context'] for e in examples]),
        future_counts,global_width,program_options).to(device)
    x=torch.as_tensor(np.asarray(contexts),dtype=torch.float32,device=device)
    q=torch.as_tensor(codes,dtype=torch.long,device=device)
    root=torch.as_tensor(roots,dtype=torch.long,device=device)
    future=torch.as_tensor(futures,dtype=torch.long,device=device)
    y=torch.as_tensor(outcomes,dtype=torch.float32,device=device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=.005);last={}
    with torch.enable_grad():
        for step in range(updates):
            chosen=[blocks[i] for i in rng.integers(len(blocks),size=48)]
            size=max(map(len,chosen));indices=np.zeros((len(chosen),size),dtype=np.int64)
            valid=np.zeros_like(indices,dtype=bool)
            for row,block in enumerate(chosen):indices[row,:len(block)]=block;valid[row,:len(block)]=True
            index=torch.as_tensor(indices,device=device);mask=torch.as_tensor(valid,device=device)
            flat=index[mask]
            logits=x.new_zeros(index.shape).masked_scatter(mask,
                net.conditional_logits(x[flat],q[flat],root[flat],future[flat]))
            target=y[index]*mask;length=mask.sum(-1)
            cardinality,ordering=future_task_likelihood_factors(logits,target,mask)
            joint=(cardinality+difference_weight*ordering)/length
            regularizer=.002*net.centered_futures().square().mean()
            loss=joint.mean()+regularizer
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            if (step+1)%500==0 or step+1==updates:
                last=dict(phase='joint_future_task_fit',update=step+1,
                    cardinality_nll=float((cardinality/length).mean().detach()),
                    conditional_order_nll=float((ordering/length).mean().detach()))
                print(last,flush=True)
    record=dict(kind='paired_future_task',width=width,response_rank=response_rank,
        contexts=np.asarray([e['context'] for e in examples]),future_counts=future_counts,
        global_width=global_width,program_options=program_options,
        state={key:value.detach().cpu() for key,value in net.state_dict().items()})
    return net.cpu().eval(),record,dict(roots=len(examples),unique_terminal_observations=len(codes),
        paired_future_blocks=len(blocks),updates=updates,seed=seed,duplicate_contrast_labels=0,
        ordering_weight=difference_weight,deployed_parameters=sum(p.numel() for p in net.parameters()),
        likelihood_factorization='future success cardinality and conditional successful-program assignment',last=last)


class SuccessorStateEncoder(nn.Module):
    def __init__(self,width,action_conditioned,conditioned_successor=False,action_radix=4,global_width=3):
        super().__init__();self.action_conditioned=action_conditioned
        self.conditioned_successor=conditioned_successor
        self.action_radix=action_radix;self.global_width=global_width
        self.node=nn.Sequential(nn.Linear(8+(action_radix if action_conditioned else 0)+(8+global_width if conditioned_successor else 0),width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.message=nn.Sequential(nn.Linear(2*width+8,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.update=nn.Sequential(nn.Linear(3*width,width),nn.SiLU())

    def forward(self,context,codes=None,following=None):
        n=(context.shape[-1]-self.global_width)//8;rows=context[:,:8*n].reshape(-1,n,8)
        inputs=rows
        if self.action_conditioned:
            program=(codes[:,None]//(self.action_radix**torch.arange(n,device=context.device)))%self.action_radix
            inputs=torch.cat((rows,F.one_hot(program,self.action_radix).to(rows.dtype)),-1)
        if self.conditioned_successor:
            change=following-context
            inputs=torch.cat((inputs,change[:,:8*n].reshape(-1,n,8),change[:,-self.global_width:,None].transpose(1,2).expand(-1,n,-1)),-1)
        node=self.node(inputs)
        difference=rows[:,None]-rows[:,:,None]
        messages=self.message(torch.cat((node[:,:,None].expand(-1,-1,n,-1),
            node[:,None].expand(-1,n,-1,-1),difference),-1))
        mask=~torch.eye(n,dtype=torch.bool,device=context.device)[None,:,:,None]
        encoded=self.update(torch.cat((node,(messages*mask).sum(2)/max(n-1,1),
            messages.masked_fill(~mask,-1e6).amax(2)),-1))
        pooled=torch.cat((encoded.mean(1),encoded.amax(1),context[:,-self.global_width:],
            context.new_full((len(context),1),n/6.)),-1)
        return encoded,pooled


class PairedSuccessorTaskPotential(nn.Module):
    """Complete-plan task value through a learned conditional successor.

    Paired interventions supervise physical state effects. True terminal task
    outcomes supervise the value of successor states and calibrate deployed
    program probabilities. No stored successful action target or hand-designed
    collision or tracking objective supplies this task value.
    Both encoders share parameters over vessels, including unseen fleet sizes.
    """
    def __init__(self,width=59,response_width=24,state=None,program_options=None,global_width=3):
        super().__init__();self.width=width;self.response_width=response_width
        self.radix=4 if program_options is None else len(program_options);self.paired=False
        self.global_width=global_width
        self.register_buffer('program_options',None if program_options is None else
            torch.as_tensor(program_options,dtype=torch.float32))
        self.response=SuccessorStateEncoder(response_width,True,action_radix=self.radix,global_width=global_width)
        self.response_node=nn.Linear(response_width,8)
        self.response_global=nn.Linear(2*response_width+global_width+1,global_width)
        self.task=SuccessorStateEncoder(width,False,global_width=global_width)
        self.task_head=nn.Sequential(nn.Linear(2*width+global_width+1,width),nn.SiLU(),nn.Linear(width,1))
        self.register_buffer('noise_scale',torch.zeros(8+global_width))
        # Symmetric common-disturbance quadrature is invariant to vessel order.
        signs=torch.tensor([[1 if (i&j).bit_count()%2==0 else -1 for j in range(1,9+global_width)]
                            for i in range(16)],dtype=torch.float32)
        self.register_buffer('noise_points',signs,persistent=False)
        if state is not None:self.load_state_dict(state)

    def successor(self,context,codes):
        node,pooled=self.response(context,codes)
        return context+torch.cat((self.response_node(node).flatten(1),self.response_global(pooled)),-1)

    def task_logits(self,successor):
        return self.task_head(self.task(successor)[1]).squeeze(-1)

    def forward(self,context,codes):
        mean=self.successor(context,codes);n=(context.shape[-1]-self.global_width)//8
        perturbation=self.noise_points*self.noise_scale
        perturbation=torch.cat((perturbation[:,:8,None].transpose(1,2).expand(-1,n,-1).flatten(1),
                                perturbation[:,-self.global_width:]),-1)
        states=mean[:,None]+perturbation[None]
        return self.task_logits(states.flatten(0,1)).sigmoid().reshape(len(context),-1).mean(-1)


def intervention_task_context(packet,return_order=False,absolute_frame=False):
    context,order=public_return_context(packet,True,2)
    if absolute_frame:
        # The author's reset sets attacker heading to minus its world
        # bearing. Rotating away that observable bearing therefore merges
        # states with different future dynamics, especially at zero speed.
        motion=packet['motion'];heading=motion[:,0,2]
        relative=30.*packet['node'][:,0,2:4]
        target=motion[:,0,:2]+torch.stack((heading.cos()*relative[:,0]-heading.sin()*relative[:,1],
            heading.sin()*relative[:,0]+heading.cos()*relative[:,1]),-1)
        bearing=target/target.norm(dim=-1,keepdim=True).clamp_min(1.)
        context=torch.cat((context[:,:-3],bearing,context[:,-3:]),-1)
    return (context,order) if return_order else context


def append_vessel_identity(context, order):
    """Retain observable physical vessel order after canonical spatial sorting."""
    n = order.shape[-1]
    if context.shape[:-1] != order.shape[:-1] or context.shape[-1] != 8*n+5:
        raise ValueError('Vessel identity requires the original public context and its measured order.')
    expected = torch.arange(n, device=order.device).expand_as(order)
    if not torch.equal(order.sort(-1).values, expected):
        raise ValueError('Measured vessel order must be a permutation of the physical vessel indices.')
    identity = order.to(device=context.device, dtype=context.dtype)/max(n-1, 1)
    return torch.cat((context[..., :-5], identity, context[..., -5:]), -1)


def successor_task_examples(data,require_candidate_support=False,absolute_frame=False):
    """Recover identity-aligned physical successors and terminal task labels."""
    import numpy as np
    groups={}
    for rollout in data['rollouts']:
        for branch in rollout['branches']:
            packet={key:torch.as_tensor(value[None]) for key,value in branch['packet'].items()}
            context,order=intervention_task_context(packet,True,absolute_frame);order=order[0]
            key=np.round(context.numpy(),6).tobytes()
            if key not in groups:
                radix=len(data['program_options']) if data.get('program_options') is not None else 4
                candidates=successor_candidate_programs(packet,order[None],
                    data.get('include_all_vessels',False),radix)[0]
                required=(candidates*(radix**torch.arange(len(order)))).sum(-1).tolist()
                groups[key]=dict(context=context[0].numpy(),samples={},pairs=[],
                    required_programs=sorted(set(required)))
            example=groups[key]
            pair=[]
            observations=[('left','before'),('right','after')]
            if branch.get('factual_outcome') is not None and branch.get('reference_action') is not None:
                observations.append(('reference_action','factual_outcome'))
            for side,outcome in observations:
                if branch[side].shape[-1]==13:
                    gates=branch[side][order.numpy(),9:11]
                    choices=np.where((gates<0.).any(-1),0,1+6*gates[:,0]+2*gates[:,1]).astype(int)
                    radix=10
                else:
                    gates=branch[side][order.numpy(),3]
                    choices=np.where(gates<0.,0,1+2*gates).astype(int);radix=4
                code=int((choices*radix**np.arange(len(choices))).sum())
                record=branch[outcome];first=record.get('first_step')
                if first is None or first.get('next') is None:
                    raise ValueError('Successor task learning requires the observed post-intervention state.')
                future=branch['continuation_seed'];sample_key=(code,future)
                following={name:torch.as_tensor(value[None]) for name,value in first['next'].items()}
                after,after_order=intervention_task_context(following,True,absolute_frame);n=len(order)
                rows=after[0,:8*n].reshape(n,8)[after_order[0].argsort()][order]
                successor=torch.cat((rows.flatten(),after[0,8*n:])).numpy()
                sample=(code,successor,float(record['success']))
                previous=example['samples'].get(sample_key)
                if previous is not None and (previous[2]!=sample[2] or
                        not np.allclose(previous[1],sample[1],rtol=1e-5,atol=1e-6)):
                    raise ValueError('One program/future has conflicting physical outcomes; '
                        'recollect with shared continuation identities before fitting paired task values.')
                if previous is None:example['samples'][sample_key]=sample
                if side!='reference_action':pair.append(sample_key)
            example['pairs'].append(tuple(pair))
    if require_candidate_support:
        for index,example in enumerate(groups.values()):
            observed={sample[0] for sample in example['samples'].values()}
            missing=set(example['required_programs'])-observed
            if missing:
                raise ValueError(f'Successor root {index} lacks {len(missing)} deployed programs; '
                    'recollect this root with successor_intervention_library before training.')
    return list(groups.values())


def fit_successor_task_potential(data,width=59,response_width=24,updates=5000,seed=42,device='cpu',
                                 decision_weight=1.,terminal_difference_weight=4.,absolute_context=False,
                                 paired_calibration=False,covariance_shrink=.5,
                                 examples=None,global_width=None,conditional_ordering=False):
    """Fit transition effects and task values from disjoint label objects."""
    import numpy as np
    if examples is None:examples=successor_task_examples(data,absolute_frame=absolute_context)
    contexts=[];codes=[];successors=[];outcomes=[];pairs=[];anchors=[];groups=[]
    if global_width is None:global_width=5 if absolute_context else 3
    calibration,calibration_fit=(paired_block_task_estimates(examples,covariance_shrink)
        if paired_calibration else (None,None))
    marginal_targets=[]
    residuals=[]
    for root,example in enumerate(examples):
        lookup={};programs={}
        calibrated=(dict(zip(calibration[root]['programs'],calibration[root]['targets']))
            if calibration is not None else None)
        for key,(code,successor,outcome) in example['samples'].items():
            lookup[key]=len(codes);contexts.append(example['context']);codes.append(code)
            successors.append(successor);outcomes.append(outcome);programs.setdefault(code,[]).append(len(codes)-1)
            marginal_targets.append(outcome if calibrated is None else float(calibrated[code]))
        pairs.extend((lookup[a],lookup[b]) for a,b in example['pairs'])
        groups.extend(programs.values())
        factual=np.asarray([successors[i] for i in programs.get(0,[])])
        if not len(factual):raise ValueError('Each conditional root needs factual successor observations.')
        anchors.append((example['context'],factual.mean(0)))
        residuals.extend(factual-factual.mean(0))
    context=torch.tensor(np.asarray(contexts),dtype=torch.float32,device=device)
    code=torch.tensor(codes,device=device);target=torch.tensor(np.asarray(successors),dtype=torch.float32,device=device)
    outcome=torch.tensor(outcomes,dtype=torch.float32,device=device);pair=torch.tensor(pairs,device=device)
    marginal=torch.tensor(marginal_targets,dtype=torch.float32,device=device)
    anchor_context=torch.tensor(np.asarray([x[0] for x in anchors]),dtype=torch.float32,device=device)
    anchor_target=torch.tensor(np.asarray([x[1] for x in anchors]),dtype=torch.float32,device=device)
    n=(context.shape[-1]-global_width)//8
    difference=target[pair[:,1]]-target[pair[:,0]]
    node_scale=difference[:,:8*n].reshape(-1,n,8).square().mean((0,1)).sqrt().clamp_min(.01)
    global_scale=difference[:,-global_width:].square().mean(0).sqrt().clamp_min(.01)
    scale=torch.cat((node_scale.repeat(n),global_scale))
    torch.manual_seed(seed);rng=np.random.default_rng(seed+19081987)
    net=PairedSuccessorTaskPotential(width,response_width,program_options=data.get('program_options'),global_width=global_width).to(device)
    residual=torch.tensor(np.asarray(residuals),dtype=torch.float32,device=device)
    with torch.no_grad():
        net.noise_scale[:8]=residual[:,:8*n].reshape(-1,n,8).square().mean((0,1)).sqrt()
        net.noise_scale[-global_width:]=residual[:,-global_width:].square().mean(0).sqrt()
    optimizer=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=.005)
    last={}
    with torch.enable_grad():
        for step in range(updates):
            selected=torch.as_tensor(rng.integers(len(pair),size=128),device=device)
            a,b=pair[selected].unbind(-1)
            predicted=net.successor(context[torch.cat((a,b))],code[torch.cat((a,b))])
            effect_loss=((predicted[128:]-predicted[:128]-(target[b]-target[a]))/scale).square().mean()
            anchored=net.successor(anchor_context,torch.zeros(len(anchors),dtype=torch.long,device=device))
            anchor_loss=((anchored-anchor_target)/scale).square().mean()
            # Uniform programs, then uniform distinct futures: a graph hub is
            # not replicated as extra task evidence merely due to its degree.
            task_indices=torch.tensor([rng.choice(groups[i]) for i in rng.integers(len(groups),size=256)],device=device)
            logits=net.task_logits(target[task_indices])
            task_loss=F.binary_cross_entropy_with_logits(logits,outcome[task_indices])
            physical_loss=effect_loss+anchor_loss
            # The auxiliary state coordinates have different units and can
            # initially dominate the actual task objective by three orders
            # of magnitude. Normalize their gradient scale while retaining
            # the physical prediction problem itself.
            physical_weight=1./physical_loss.detach().clamp_min(1.)
            loss=physical_weight*physical_loss+task_loss
            # Supervise the complete-program probabilities used to choose
            # controls, rather than only an auxiliary physical-state effect.
            paired_indices=torch.cat((a[:32],b[:32]))
            paired_value=net(context[paired_indices],code[paired_indices])
            if conditional_ordering:
                direction=outcome[b[:32]]-outcome[a[:32]]
                discordant=direction!=0.
                logit=torch.logit(paired_value.clamp(1e-6,1.-1e-6))
                terms=F.softplus(-direction*(logit[32:]-logit[:32]))
                terminal_contrast=(terms*discordant).sum()/discordant.sum().clamp_min(1)
            else:
                terminal_contrast=(paired_value[32:]-paired_value[:32]
                    -(marginal[b[:32]]-marginal[a[:32]])).square().mean()
            loss=loss+terminal_difference_weight*terminal_contrast
            if paired_calibration:
                # Individual futures supervise conditional successor values.
                # Their population-calibrated differences supervise the
                # marginal values used for decisions. These are different
                # targets: a single realized rescue is not a unit change in
                # expected success probability.
                observed=net.task_logits(target[paired_indices]).sigmoid()
                observed_contrast=(observed[32:]-observed[:32]
                    -(outcome[b[:32]]-outcome[a[:32]])).square().mean()
                loss=loss+terminal_difference_weight*observed_contrast
            if decision_weight:
                # Calibrate the complete-program probability actually used
                # for decisions, alongside its paired physical-effect model.
                decision_indices=task_indices[:64]
                probability=net(context[decision_indices],code[decision_indices])
                decision_loss=F.binary_cross_entropy(probability,marginal[decision_indices])
                loss=loss+decision_weight*decision_loss
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            if (step+1)%500==0 or step+1==updates:
                last=dict(phase='paired_successor_task_fit',update=step+1,effect_loss=float(effect_loss.detach()),
                    factual_motion_loss=float(anchor_loss.detach()),task_value_loss=float(task_loss.detach()),
                    deployed_terminal_contrast_loss=float(terminal_contrast.detach()),
                    physical_objective_weight=float(physical_weight))
                if decision_weight:last['deployed_task_value_loss']=float(decision_loss.detach())
                if paired_calibration:last['observed_future_contrast_loss']=float(observed_contrast.detach())
                print(last,flush=True)
    record=dict(kind='paired_successor_task',width=width,response_width=response_width,global_width=global_width,
        program_options=data.get('program_options'),
        state={key:value.detach().cpu() for key,value in net.state_dict().items()})
    net=net.cpu().eval()
    if paired_calibration:net,record=calibrate_program_potential(record,calibration,
        time_bandwidth=None if data.get('initial_only',True) else 1e-4)
    return net,record,dict(roots=len(examples),unique_task_outcomes=len(codes),paired_state_effects=len(pairs),
        terminal_difference_labels=len(pairs),task_target='true_terminal_task_success',state_target='paired_intervention_successor',
        deployed_parameters=sum(p.numel() for p in net.parameters()),updates=updates,
        decision_weight=decision_weight,terminal_difference_weight=terminal_difference_weight,last=last,
        paired_calibration=calibration_fit,conditional_ordering=conditional_ordering)


class PairedFactualTaskPotential(nn.Module):
    """Marginal task value from paired repair and damage probabilities.

    The factual successor and its terminal result identify a common sampled
    continuation during learning. A complete program predicts whether it
    repairs a factual failure or damages a factual success. Execution averages
    both probabilities over the entire empirical factual distribution; no
    realized future, selected successful action, or geometric proxy is used.
    """
    def __init__(self,width=62,global_width=3,support_shape=(1,1,51),state=None,program_options=None,
                 response_supervision=False,factual_features=6,intervention_contrast=False,future_rank=0,
                 future_normalization=False):
        super().__init__();self.width=width;self.global_width=global_width
        if factual_features not in (0,6,8):raise ValueError('Use terminal pairing, measured motion, or complete factual state.')
        self.factual_features=factual_features;self.node_width=8+factual_features
        self.intervention_contrast=intervention_contrast
        if future_rank and factual_features:
            raise ValueError('Factorized factual responses use separate program and future encoders.')
        self.future_rank=future_rank
        self.future_normalization=future_normalization
        self.radix=4 if program_options is None else len(program_options)
        self.register_buffer('program_options',None if program_options is None else
            torch.as_tensor(program_options,dtype=torch.float32))
        self.task=SharedProgramPotential(width,False,node_width=self.node_width,global_width=2*global_width,
            program_options=program_options)
        if self.node_width>=15:
            # These extra columns are factual kinematics, not the source
            # control columns used by the generic program network.
            self.task.node[0]=nn.Linear(self.node_width+self.radix,width)
        self.task.readout[-1]=nn.Linear(width,2*(1+future_rank))
        if future_rank:
            self.future_encoder=nn.Sequential(nn.Linear(16+global_width,16),nn.SiLU(),
                nn.Linear(16,future_rank,bias=False))
        self.response_supervision=response_supervision
        if response_supervision:
            self.response_node=nn.Linear(width,8,bias=False)
            self.response_global=nn.Linear(2*width,global_width,bias=False)
        self.register_buffer('support_contexts',torch.zeros((support_shape[0],support_shape[-1])))
        self.register_buffer('support_delta',torch.zeros(tuple(support_shape)))
        self.register_buffer('support_outcomes',torch.zeros(tuple(support_shape[:2])))
        self.register_buffer('support_mask',torch.zeros(tuple(support_shape[:2]),dtype=torch.bool))
        if future_normalization:
            self.register_buffer('future_center',torch.zeros((support_shape[0],16+global_width)))
            self.register_buffer('future_scale',torch.ones((support_shape[0],16+global_width)))
        if state is not None:self.load_state_dict(state)

    def conditional_encoding(self,context,codes,factual):
        extra=self.global_width;n=(context.shape[-1]-extra)//8
        current=context[:,:8*n].reshape(-1,n,8)
        response=(factual[:,:8*n]-context[:,:8*n]).reshape(-1,n,8)
        # Angular velocity carries turning inertia; radial progress carries
        # the task consequence of a common future. Both are observed factual
        # state, rather than an extra copy of the terminal difference label.
        inputs=torch.cat((torch.cat((current,response[...,:self.factual_features]),-1).flatten(1),
            context[:,-extra:],factual[:,-extra:]-context[:,-extra:] if self.factual_features
                else torch.zeros_like(context[:,-extra:])), -1)
        rows=inputs[:,:self.node_width*n].reshape(-1,n,self.node_width)
        programs=(codes[:,None]//(self.radix**torch.arange(n,device=context.device)))%self.radix
        node=self.task.node(torch.cat((rows,F.one_hot(programs,self.radix).to(rows.dtype)),-1))
        difference=rows[:,None]-rows[:,:,None]
        messages=self.task.message(torch.cat((node[:,:,None].expand(-1,-1,n,-1),
            node[:,None].expand(-1,n,-1,-1),difference),-1))
        mask=~torch.eye(n,dtype=torch.bool,device=context.device)[None,:,:,None]
        average=(messages*mask).sum(2)/max(1,n-1)
        maximum=messages.masked_fill(~mask,-1e6).amax(2)
        encoded=self.task.update(torch.cat((node,average,maximum),-1))
        pooled=torch.cat((encoded.mean(1),encoded.amax(1),inputs[:,-2*extra:],
            context.new_full((len(context),1),n/6.)),-1)
        return encoded,pooled

    def task_parameters(self,context,codes,factual):
        encoded,pooled=self.conditional_encoding(context,codes,factual)
        if self.intervention_contrast:
            baseline,_=self.conditional_encoding(context,torch.zeros_like(codes),factual)
            pooled=self.intervention_pool(encoded-baseline,pooled)
        return self.task.readout(pooled)

    def future_state_features(self,context,factual):
        extra=self.global_width;n=(context.shape[-1]-extra)//8
        delta=factual-context
        node=delta[:,:8*n].reshape(-1,n,8)
        return torch.cat((node.mean(1),node.square().mean(1).sqrt(),delta[:,-extra:]),-1)

    def future_features(self,context,factual):
        features=self.future_state_features(context,factual)
        if self.future_normalization:
            root=self.factual_roots(context)
            # Remove the root's expected motion before learning sensitivity
            # to its common future. Otherwise large between-state motion
            # changes can hide the much smaller paired stochastic response.
            features=(features-self.future_center[root])/self.future_scale[root]
        return self.future_encoder(features)

    def conditional_logits(self,context,codes,factual):
        parameters=self.task_parameters(context,codes,factual)
        if not self.future_rank:return parameters
        response=self.future_features(context,factual)
        return parameters[:,:2]+(parameters[:,2:].reshape(-1,2,self.future_rank)*response[:,None]).sum(-1)

    def intervention_pool(self,change,pooled):
        # Pool program-induced feature changes before reducing across vessels.
        # Unchanged dominant vessel features must not hide a local fusion
        # intervention in a max over the complete factual fleet state.
        return torch.cat((change.mean(1),change.amax(1),pooled[:,2*self.width:]),-1)

    def conditional_predictions(self,context,codes,factual):
        encoded,pooled=self.conditional_encoding(context,codes,factual)
        base_node,base_global=self.conditional_encoding(context,torch.zeros_like(codes),factual)
        # The same learned features must explain the observed intervention
        # response and its terminal task consequences. The factual program
        # has identically zero physical change, including after training.
        change=torch.cat((F.linear(encoded-base_node,self.response_node.weight).flatten(1),
            F.linear((pooled-base_global)[:,:2*self.width],self.response_global.weight)),-1)
        if self.intervention_contrast:pooled=self.intervention_pool(encoded-base_node,pooled)
        parameters=self.task.readout(pooled)
        if self.future_rank:
            parameters=parameters[:,:2]+(parameters[:,2:].reshape(-1,2,self.future_rank)
                *self.future_features(context,factual)[:,None]).sum(-1)
        return parameters,change

    def factual_roots(self,context):
        extra=self.global_width
        def summary(value):
            rows=value[:,:value.shape[-1]-extra].reshape(len(value),-1,8)
            return torch.cat((rows.mean(1),rows.square().mean(1).sqrt(),value[:,-extra:]),-1)
        return (summary(context)[:,None]-summary(self.support_contexts)[None]).square().sum(-1).argmin(-1)

    def factual_particles(self,context):
        extra=self.global_width;n=(context.shape[-1]-extra)//8
        m=(self.support_contexts.shape[-1]-extra)//8
        nearest=self.factual_roots(context)
        own=context[:,:8*n].reshape(-1,n,8)
        reference=self.support_contexts[nearest,:8*m].reshape(-1,m,8)
        assignment=(own[:,:,None]-reference[:,None]).square().sum(-1).argmin(-1)
        delta=self.support_delta[nearest];count=delta.shape[1]
        node=delta[:,:,:8*m].reshape(-1,count,m,8).gather(2,
            assignment[:,None,:,None].expand(-1,count,-1,8))
        following=context[:,None]+torch.cat((node.flatten(2),delta[:,:,-extra:]),-1)
        return following,self.support_outcomes[nearest],self.support_mask[nearest]

    def forward(self,context,codes):
        if self.future_rank:
            result=[]
            for x,q in zip(context.split(16),codes.split(16)):
                factual,outcome,mask=self.factual_particles(x);valid=mask.any(0)
                factual=factual[:,valid];outcome=outcome[:,valid];mask=mask[:,valid];count=factual.shape[1]
                parameters=self.task_parameters(x,q,x)
                response=self.future_features(x[:,None].expand(-1,count,-1).flatten(0,1),
                    factual.flatten(0,1)).reshape(len(x),count,self.future_rank)
                logits=parameters[:,None,:2]+(parameters[:,None,2:].reshape(-1,1,2,self.future_rank)
                    *response[:,:,None]).sum(-1)
                probability=logits.sigmoid()
                value=(1.-outcome)*probability[...,0]+outcome*(1.-probability[...,1])
                value=torch.where(q[:,None]==0,outcome,value)
                result.append((value*mask).sum(-1)/mask.sum(-1).clamp_min(1))
            return torch.cat(result)
        if not self.factual_features:
            # Integrate the two observed factual outcome strata exactly.
            # This restricted conditional model shares repair/damage rates
            # across future draws instead of fitting a high-dimensional
            # successor response separately from few samples per program.
            _,outcome,mask=self.factual_particles(context)
            reference=(outcome*mask).sum(-1)/mask.sum(-1).clamp_min(1)
            probability=self.conditional_logits(context,codes,context).sigmoid()
            value=(1.-reference)*probability[:,0]+reference*(1.-probability[:,1])
            return torch.where(codes==0,reference,value)
        result=[]
        for x,q in zip(context.split(4),codes.split(4)):
            factual,outcome,mask=self.factual_particles(x)
            valid=mask.any(0)
            factual=factual[:,valid];outcome=outcome[:,valid];mask=mask[:,valid]
            count=factual.shape[1]
            probability=self.conditional_logits(x[:,None].expand(-1,count,-1).flatten(0,1),
                q[:,None].expand(-1,count).flatten(),factual.flatten(0,1)).sigmoid().reshape(len(x),count,2)
            value=(1.-outcome)*probability[...,0]+outcome*(1.-probability[...,1])
            value=torch.where(q[:,None]==0,outcome,value)
            result.append((value*mask).sum(-1)/mask.sum(-1).clamp_min(1))
        return torch.cat(result)


def fit_paired_factual_task_potential(examples,*,program_options=None,global_width=3,
                                    width=62,updates=5000,seed=42,device='cpu',response_supervision=False,
                                    response_weight=.25,factual_features=6,intervention_contrast=False,future_rank=0,
                                    future_normalization=False):
    """Use each observed task result once under its matched factual future."""
    import numpy as np
    contexts=[];codes=[];factual_states=[];factual_outcomes=[];outcomes=[];groups=[];supports=[];successors=[]
    for example in examples:
        factual={future:(state,outcome) for (q,future),(_,state,outcome) in example['samples'].items() if q==0}
        if not factual:raise ValueError('Paired task learning requires factual continuations.')
        supports.append(list(factual.values()));programs={}
        for (q,future),(_,following,outcome) in example['samples'].items():
            if future not in factual:raise ValueError('Collect a factual branch for every intervention future.')
            if q==0:continue
            reference,reference_outcome=factual[future]
            programs.setdefault(q,[]).append(len(codes));contexts.append(example['context']);codes.append(q)
            factual_states.append(reference);factual_outcomes.append(reference_outcome);outcomes.append(outcome)
            successors.append(following)
        groups.extend(programs.values())
    if not groups:raise ValueError('Paired task learning requires observed non-factual programs.')
    torch.manual_seed(seed);rng=np.random.default_rng(seed+19081987)
    shape=(len(examples),max(map(len,supports)),len(examples[0]['context']))
    net=PairedFactualTaskPotential(width,global_width,shape,program_options=program_options,
        response_supervision=response_supervision,factual_features=factual_features,
        intervention_contrast=intervention_contrast,future_rank=future_rank,
        future_normalization=future_normalization).to(device)
    with torch.no_grad():
        for i,(example,support) in enumerate(zip(examples,supports)):
            x=torch.as_tensor(example['context'],device=device);count=len(support)
            net.support_contexts[i].copy_(x)
            net.support_delta[i,:count].copy_(torch.as_tensor(np.asarray([item[0] for item in support]),device=device)-x)
            net.support_outcomes[i,:count].copy_(torch.as_tensor([item[1] for item in support],device=device))
            net.support_mask[i,:count]=True
            if future_normalization:
                features=net.future_state_features(x[None].expand(count,-1),
                    x[None]+net.support_delta[i,:count])
                net.future_center[i].copy_(features.mean(0))
                net.future_scale[i].copy_(features.std(0,unbiased=False).clamp_min(.001))
    x=torch.as_tensor(np.asarray(contexts),dtype=torch.float32,device=device)
    q=torch.as_tensor(codes,dtype=torch.long,device=device)
    factual=torch.as_tensor(np.asarray(factual_states),dtype=torch.float32,device=device)
    y0=torch.as_tensor(factual_outcomes,dtype=torch.float32,device=device)
    y=torch.as_tensor(outcomes,dtype=torch.float32,device=device)
    delta=torch.as_tensor(np.asarray(successors),dtype=torch.float32,device=device)-factual
    n=(x.shape[-1]-global_width)//8
    scale=torch.cat((delta[:,:8*n].reshape(-1,n,8).square().mean((0,1)).sqrt().clamp_min(.01).repeat(n),
        delta[:,-global_width:].square().mean(0).sqrt().clamp_min(.01)))
    optimizer=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=.005);last={}
    with torch.enable_grad():
        for step in range(updates):
            selected=torch.as_tensor([rng.choice(groups[i]) for i in rng.integers(len(groups),size=256)],device=device)
            response_loss=x.new_zeros(())
            if response_supervision:
                logits,change=net.conditional_predictions(x[selected],q[selected],factual[selected])
                response_loss=((change-delta[selected])/scale).square().mean()
            else:logits=net.conditional_logits(x[selected],q[selected],factual[selected])
            observed=logits.gather(1,y0[selected].long()[:,None]).squeeze(1)
            target=torch.where(y0[selected]>0.,1.-y[selected],y[selected])
            task_loss=F.binary_cross_entropy_with_logits(observed,target)
            loss=task_loss+response_weight*response_loss/response_loss.detach().clamp_min(1.)
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            if (step+1)%500==0 or step+1==updates:
                last=dict(phase='paired_factual_task_fit',update=step+1,conditional_task_nll=float(task_loss.detach()),
                    paired_response_error=float(response_loss.detach()))
                print(last,flush=True)
    record=dict(kind='paired_factual_task',width=width,global_width=global_width,support_shape=shape,
        program_options=program_options,response_supervision=response_supervision,factual_features=factual_features,
        intervention_contrast=intervention_contrast,
        future_rank=future_rank,
        future_normalization=future_normalization,
        state={key:value.detach().cpu() for key,value in net.state_dict().items()})
    return net.cpu().eval(),record,dict(roots=len(examples),unique_terminal_observations=len(q)+sum(map(len,supports)),
        factual_future_conditions=sum(map(len,supports)),terminal_difference_labels=0,
        task_target='terminal repair and damage probabilities integrated over paired factual futures',
        deployed_parameters=sum(p.numel() for p in net.parameters()),updates=updates,seed=seed,last=last,
        paired_response_supervision=response_supervision,response_weight=response_weight if response_supervision else 0.,
        factual_state_features=factual_features,intervention_contrast=intervention_contrast,
        factual_response_rank=future_rank,factual_future_normalization=future_normalization)


class PairedTransportTaskPotential(nn.Module):
    """Integrate learned intervention transport over factual future states.

    The empirical support contains only factual successors, never selected
    controls or successful-plan labels. Paired intervention states identify
    how a plan changes each realized response. Terminal task outcomes train
    both the successor-state value and the deployed marginal task value.
    """
    def __init__(self,width=59,response_width=24,support_shape=(1,32,51),state=None,program_options=None):
        super().__init__();self.width=width;self.response_width=response_width
        self.radix=4 if program_options is None else len(program_options);self.paired=False
        self.register_buffer('program_options',None if program_options is None else torch.as_tensor(program_options,dtype=torch.float32))
        self.response=SuccessorStateEncoder(response_width,True,True,self.radix)
        self.response_node=nn.Linear(response_width,8)
        self.response_global=nn.Linear(2*response_width+4,3)
        self.task=SuccessorStateEncoder(width,False)
        self.task_head=nn.Sequential(nn.Linear(2*width+4,width),nn.SiLU(),nn.Linear(width,1))
        self.register_buffer('support_contexts',torch.zeros((support_shape[0],support_shape[-1])))
        self.register_buffer('support_delta',torch.zeros(tuple(support_shape)))
        if state is not None:self.load_state_dict(state)

    def effect(self,context,following,codes):
        node,pooled=self.response(context,codes,following)
        return torch.cat((self.response_node(node).flatten(1),self.response_global(pooled)),-1)

    def transport(self,context,following,left,right):
        return following+(self.effect(context,following,right)-self.effect(context,following,left))

    def task_logits(self,following):
        return self.task_head(self.task(following)[1]).squeeze(-1)

    def factual_particles(self,context,particle_indices=None):
        n=(context.shape[-1]-3)//8;m=(self.support_contexts.shape[-1]-3)//8
        summary=SuccessorTaskDecision.summary
        nearest=(summary(context)[:,None]-summary(self.support_contexts)[None]).square().sum(-1).argmin(-1)
        base=self.support_contexts[nearest]
        own=context[:,:8*n].reshape(-1,n,8);reference=base[:,:8*m].reshape(-1,m,8)
        assignment=(own[:,:,None]-reference[:,None]).square().sum(-1).argmin(-1)
        delta=self.support_delta[nearest]
        if particle_indices is not None:delta=delta[:,particle_indices]
        count=delta.shape[1]
        node=delta[:,:,:8*m].reshape(-1,count,m,8).gather(2,
            assignment[:,None,:,None].expand(-1,count,-1,8))
        return context[:,None]+torch.cat((node.flatten(2),delta[:,:,-3:]),-1)

    def forward(self,context,codes,particle_indices=None):
        factual=self.factual_particles(context,particle_indices);count=factual.shape[1]
        x=context[:,None].expand(-1,count,-1).flatten(0,1)
        q=codes[:,None].expand(-1,count).flatten()
        following=self.transport(x,factual.flatten(0,1),torch.zeros_like(q),q)
        return self.task_logits(following).sigmoid().reshape(len(context),count).mean(-1)


class PairedLatentTaskPotential(PairedTransportTaskPotential):
    """Separate a paired future from the fusion program that exposes it.

    Encoders of both intervention arms identify one shared future. Decoding
    that future under another program predicts its task-relevant successor.
    Deployment integrates only factual futures, without knowing the realized
    future or retrieving a previously successful action.
    """
    def __init__(self,width=54,response_width=24,support_shape=(1,32,51),state=None,program_options=None):
        nn.Module.__init__(self);self.width=width;self.response_width=response_width
        self.radix=4 if program_options is None else len(program_options);self.paired=False
        self.register_buffer('program_options',None if program_options is None else torch.as_tensor(program_options,dtype=torch.float32))
        self.encoder=SuccessorStateEncoder(response_width,True,True,self.radix)
        self.encoder_node=nn.Linear(response_width,8)
        self.encoder_global=nn.Linear(2*response_width+4,3)
        self.decoder=SuccessorStateEncoder(response_width,True,True,self.radix)
        self.decoder_node=nn.Linear(response_width,8)
        self.decoder_global=nn.Linear(2*response_width+4,3)
        self.task=SuccessorStateEncoder(width,False)
        self.task_head=nn.Sequential(nn.Linear(2*width+4,width),nn.SiLU(),nn.Linear(width,1))
        self.register_buffer('support_contexts',torch.zeros((support_shape[0],support_shape[-1])))
        self.register_buffer('support_delta',torch.zeros(tuple(support_shape)))
        if state is not None:self.load_state_dict(state)

    def encode_future(self,context,following,codes):
        node,pooled=self.encoder(context,codes,following)
        return context+torch.cat((self.encoder_node(node).flatten(1),self.encoder_global(pooled)),-1)

    def decode_future(self,context,future,codes):
        node,pooled=self.decoder(context,codes,future)
        return context+torch.cat((self.decoder_node(node).flatten(1),self.decoder_global(pooled)),-1)

    def transport(self,context,following,left,right):
        return self.decode_future(context,self.encode_future(context,following,left),right)

    def forward(self,context,codes,particle_indices=None):
        factual=self.factual_particles(context,particle_indices);count=factual.shape[1]
        x=context[:,None].expand(-1,count,-1).flatten(0,1)
        q=codes[:,None].expand(-1,count).flatten()
        future=self.encode_future(x,factual.flatten(0,1),torch.zeros_like(q))
        following=self.decode_future(x,future,q)
        return self.task_logits(following).sigmoid().reshape(len(context),count).mean(-1)


def fit_successor_transport_potential(data,width=59,response_width=24,updates=5000,seed=42,
                                      device='cpu',particle_count=32,latent=False,terminal_difference_weight=4.,
                                      paired_standardization=False,covariance_shrink=.9):
    import numpy as np
    examples=successor_task_examples(data)
    contexts=[];codes=[];successors=[];outcomes=[];pairs=[];groups=[];support=[];calibration=[]
    standardized,standardization_fit=(paired_block_task_estimates(examples,covariance_shrink)
        if paired_standardization else (None,{}))
    for root,example in enumerate(examples):
        task_mean=(dict(zip(standardized[root]['programs'],standardized[root]['targets']))
            if paired_standardization else None)
        lookup={};programs={}
        for key,(code,following,outcome) in example['samples'].items():
            index=len(codes);lookup[key]=index;programs.setdefault(code,[]).append(index)
            contexts.append(example['context']);codes.append(code);successors.append(following);outcomes.append(outcome)
            calibration.append(task_mean[code] if task_mean is not None else outcome)
        pairs.extend((lookup[a],lookup[b]) for a,b in example['pairs'])
        groups.extend(programs.values())
        factual=np.asarray([successors[i] for i in programs.get(0,[])])
        if not len(factual):raise ValueError('Paired transport needs observed factual successor states.')
        # Fixed coverage of all factual draws, independent of their outcomes.
        selected=np.linspace(0,len(factual)-1,particle_count).round().astype(int)
        support.append(factual[selected]-example['context'])
    context=torch.tensor(np.asarray(contexts),dtype=torch.float32,device=device)
    code=torch.tensor(codes,device=device);target=torch.tensor(np.asarray(successors),dtype=torch.float32,device=device)
    outcome=torch.tensor(outcomes,dtype=torch.float32,device=device);pair=torch.tensor(pairs,device=device)
    calibration=torch.tensor(calibration,dtype=torch.float32,device=device).clamp(0.,1.)
    n=(context.shape[-1]-3)//8;difference=target[pair[:,1]]-target[pair[:,0]]
    scale=torch.cat((difference[:,:8*n].reshape(-1,n,8).square().mean((0,1)).sqrt().clamp_min(.01).repeat(n),
        difference[:,-3:].square().mean(0).sqrt().clamp_min(.01)))
    torch.manual_seed(seed);rng=np.random.default_rng(seed+19081987)
    shape=(len(examples),particle_count,context.shape[-1])
    if latent and width==59:width=54
    potential=PairedLatentTaskPotential if latent else PairedTransportTaskPotential
    net=potential(width,response_width,shape,program_options=data.get('program_options')).to(device)
    with torch.no_grad():
        net.support_contexts.copy_(torch.tensor(np.asarray([x['context'] for x in examples]),device=device))
        net.support_delta.copy_(torch.tensor(np.asarray(support),device=device))
    optimizer=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=.005);last={}
    with torch.enable_grad():
        for step in range(updates):
            selected=torch.as_tensor(rng.integers(len(pair),size=128),device=device)
            a,b=pair[selected].unbind(-1)
            left=torch.cat((a,b));right=torch.cat((b,a))
            predicted=net.transport(context[left],target[left],code[left],code[right])
            effect_loss=((predicted-target[right])/scale).square().mean()
            contrast_loss=effect_loss.new_zeros(())
            counterfactual_task_loss=effect_loss.new_zeros(())
            if latent:
                future=net.encode_future(context[left],target[left],code[left])
                paired_future=net.encode_future(context[right],target[right],code[right])
                reconstructed=net.decode_future(context[left],future,code[left])
                effect_loss=.5*(effect_loss+((reconstructed-target[left])/scale).square().mean())
                effect_loss=effect_loss+.1*((future-paired_future)/scale).square().mean()
                # This is the terminal contrast for the SAME realized future,
                # not the redundant difference of two marginal return means.
                logits_left=net.task_logits(reconstructed);logits_right=net.task_logits(predicted)
                value_left=logits_left.sigmoid();value_right=logits_right.sigmoid()
                contrast_loss=(value_right-value_left-(outcome[right]-outcome[left])).square().mean()
                counterfactual_task_loss=.5*(F.binary_cross_entropy_with_logits(logits_left,outcome[left])
                    +F.binary_cross_entropy_with_logits(logits_right,outcome[right]))
            elif paired_standardization:
                # Calibrate the transported physical consequence to its own
                # realized terminal task result, before integrating futures.
                logits_left=net.task_logits(target[left]);logits_right=net.task_logits(predicted)
                contrast_loss=(logits_right.sigmoid()-logits_left.sigmoid()
                    -(outcome[right]-outcome[left])).square().mean()
                counterfactual_task_loss=F.binary_cross_entropy_with_logits(logits_right,outcome[right])
            task_indices=torch.tensor([rng.choice(groups[i]) for i in rng.integers(len(groups),size=128)],device=device)
            task_loss=F.binary_cross_entropy_with_logits(net.task_logits(target[task_indices]),outcome[task_indices])
            sampled=torch.as_tensor(rng.choice(particle_count,min(8,particle_count),replace=False),device=device)
            decision_indices=task_indices[:64]
            probability=net(context[decision_indices],code[decision_indices],sampled)
            # Deployment integrates the pooled factual future distribution.
            # Its target must use that same distribution, not each arm's
            # finite, differently difficult set of sampled futures.
            decision_loss=F.binary_cross_entropy(probability.clamp(1e-6,1.-1e-6),calibration[decision_indices])
            # One task-label budget split across observed and deployed views.
            physical_weight=1./effect_loss.detach().clamp_min(1.)
            loss=(physical_weight*effect_loss+.25*(task_loss+decision_loss+counterfactual_task_loss+contrast_loss)
                if latent or paired_standardization else physical_weight*effect_loss+.5*(task_loss+decision_loss))
            deployed_contrast=effect_loss.new_zeros(())
            if not paired_standardization:
                paired_indices=torch.cat((a[:32],b[:32]))
                paired_value=net(context[paired_indices],code[paired_indices],sampled)
                deployed_contrast=(paired_value[32:]-paired_value[:32]
                    -(outcome[b[:32]]-outcome[a[:32]])).square().mean()
            # A realized future-specific difference is already supervised
            # above. Do not also identify it with a marginal value difference.
            loss=loss+terminal_difference_weight*deployed_contrast
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            if (step+1)%500==0 or step+1==updates:
                last=dict(phase='paired_latent_task_fit' if latent else 'paired_transport_task_fit',update=step+1,
                    paired_transport_loss=float(effect_loss.detach()),paired_terminal_contrast_loss=float(contrast_loss.detach()),
                    deployed_terminal_contrast_loss=float(deployed_contrast.detach()),
                    physical_objective_weight=float(physical_weight),
                    counterfactual_task_loss=float(counterfactual_task_loss.detach()),
                    observed_task_loss=float(task_loss.detach()),deployed_task_loss=float(decision_loss.detach()))
                print(last,flush=True)
    record=dict(kind='paired_latent_task' if latent else 'paired_transport_task',width=width,response_width=response_width,support_shape=shape,
        program_options=data.get('program_options'),
        state={key:value.detach().cpu() for key,value in net.state_dict().items()})
    return net.cpu().eval(),record,dict(roots=len(examples),unique_task_outcomes=len(codes),paired_state_effects=len(pairs),
        terminal_difference_labels=len(pairs),task_target='true_terminal_task_success',
        state_target='shared_future_counterfactual_successor' if latent else 'paired_realized_successor_transport',
        paired_future_task_calibration=latent,
        factual_particles_per_root=particle_count,deployed_parameters=sum(p.numel() for p in net.parameters()),
        updates=updates,terminal_difference_weight=0. if paired_standardization else terminal_difference_weight,
        paired_standardization=paired_standardization,standardization_fit=standardization_fit,last=last)


def successor_candidate_programs(packet,order,include_all_vessels=False,radix=4):
    """Complete fleet programs around the closest encounter and each vessel."""
    b,n=order.shape;device=order.device
    positions=packet['motion'][...,:2].gather(1,order[...,None].expand(-1,-1,2))
    separation=(positions[:,:,None]-positions[:,None,:]).norm(dim=-1)
    separation=separation.masked_fill(torch.eye(n,dtype=torch.bool,device=device)[None],float('inf'))
    nearest=separation.flatten(1).argmin(-1);left=nearest//n;right=nearest%n
    if radix==10:
        index=torch.arange(9*n,device=device)
        singles=torch.zeros((b,9*n,n),dtype=torch.long,device=device)
        singles.scatter_(2,(index//9)[None,:,None].expand(b,-1,-1),(1+index%9)[None,:,None].expand(b,-1,-1))
        corners=torch.tensor([1,3,7,9],device=device)
        digits=torch.arange(16,device=device)
        pairs=torch.zeros((b,16,n),dtype=torch.long,device=device)
        pairs.scatter_(2,left[:,None,None].expand(-1,16,1),corners[digits//4][None,:,None].expand(b,-1,-1))
        pairs.scatter_(2,right[:,None,None].expand(-1,16,1),corners[digits%4][None,:,None].expand(b,-1,-1))
        return torch.cat((singles.new_zeros((b,1,n)),singles,pairs),1)
    digits=torch.arange(16,device=device)
    programs=torch.zeros((b,16,n),dtype=torch.long,device=device)
    programs.scatter_(2,left[:,None,None].expand(-1,16,1),(digits//4)[None,:,None].expand(b,-1,-1))
    programs.scatter_(2,right[:,None,None].expand(-1,16,1),(digits%4)[None,:,None].expand(b,-1,-1))
    if include_all_vessels:
        singles=torch.zeros((b,3*n,n),dtype=torch.long,device=device)
        index=torch.arange(3*n,device=device)
        singles.scatter_(2,(index//3)[None,:,None].expand(b,-1,-1),
            (1+index%3)[None,:,None].expand(b,-1,-1))
        programs=torch.cat((programs,singles),1)
    return programs


def successor_intervention_library(packet,reference,span,include_all_vessels=False,program_options=None,
                                   joint_programs=None):
    """Sample exactly the complete programs offered by the executing policy."""
    import numpy as np
    tensors={key:torch.as_tensor(value[None]) for key,value in packet.items()}
    _,order=public_return_context(tensors,True,2)
    radix=4 if program_options is None else len(program_options)
    digits=successor_candidate_programs(tensors,order,include_all_vessels,radix)[0]
    inverse=order[0].argsort()
    if joint_programs is not None:
        # Grow complete fleet programs along single-vessel intervention
        # paths. A local comparison constrains the learning edge, not the
        # number of vessels that may participate in the complete program.
        choices=[tuple(map(int,row)) for row in digits.tolist()]
        choices=list(dict.fromkeys(choices));lookup={q:i for i,q in enumerate(choices)}
        zero=(0,)*len(order[0]);relations=set()
        def insert_path(target):
            current=zero
            for vessel in range(len(target)):
                if not target[vessel]:continue
                following=list(current);following[vessel]=target[vessel];following=tuple(following)
                if following not in lookup:
                    lookup[following]=len(choices);choices.append(following)
                relations.add((lookup[current],lookup[following]));current=following
        for target in list(choices)+[tuple(map(int,q)) for q in joint_programs]:insert_path(target)
        programs=[];n=len(order[0]);inverse=inverse.numpy()
        for choice in choices:
            if radix==10:
                preferences=np.asarray(program_options,dtype=np.float32)[list(choice)][inverse]
                programs.append(np.column_stack((reference,np.full(n,-1.),np.full(n,span),np.full(n,-2.),
                    np.zeros((n,2)),np.full(n,span),preferences,preferences)).astype(np.float32))
            else:
                choice=np.asarray(choice);gates=np.where(choice>0,(choice-1)*.5,-1.)[inverse]
                second=gates.copy()
                if not choice.any():second.fill(-2.)
                programs.append(np.column_stack((reference,gates,np.full(n,span),second)).astype(np.float32))
        return programs,sorted(relations),tuple(order[0].tolist())
    if radix==10:
        choices=digits.numpy();lookup={tuple(choice):index for index,choice in enumerate(choices)}
        option_array=np.asarray(program_options,dtype=np.float32);programs=[];edges=[];n=len(order[0])
        subjects=tuple(order[0,(digits[-16:]>0).any(0)].tolist())
        for index,choice in enumerate(choices):
            preferences=option_array[choice][inverse.numpy()]
            programs.append(np.column_stack((reference,np.full(n,-1.),np.full(n,span),np.full(n,-2.),
                np.zeros((n,2)),np.full(n,span),preferences,preferences)).astype(np.float32))
            if index:
                parent=choice.copy();parent[np.flatnonzero(parent)[-1]]=0
                edges.append((lookup[tuple(parent)],index))
        return programs,edges,subjects
    subjects=tuple(order[0,(digits[:16]>0).any(0)].tolist())
    programs=[];edges=[];known={}
    for original,choices in enumerate(digits):
        gates=torch.where(choices>0,(choices.float()-1.)*.5,-1.)[inverse].numpy()
        key=tuple(gates.tolist())
        if key in known:continue
        index=len(programs);known[key]=index
        second=gates.copy()
        if index==0:second.fill(-2.)
        programs.append(np.column_stack((reference,gates,np.full(len(gates),span),second)).astype(np.float32))
        if index:
            parent=(4*(original//4) if original<16 and original%4 else 0)
            edges.append((parent,index))
    return programs,edges,subjects


def transport_successor_programs(context,centers,codes,support,radix,bandwidth,absolute_frame=False,variance=None,
                                 role_conditioned=False,time_bandwidth=None):
    """Bind sampled intervention roles to a different-sized observed fleet.

    The task network evaluates the resulting complete current-fleet program.
    Unmatched current vessels retain their factual controller; a program that
    needs a missing intervened vessel has no observational support.
    """
    extra=5 if absolute_frame else 3
    n=(context.shape[-1]-extra)//8;m=(centers.shape[-1]-extra)//8
    query=torch.cat((context[:,:8*n],context[:,-3:]),-1)
    roots=torch.cat((centers[:,:8*m],centers[:,-3:]),-1)
    distance,indices,mapping=match_fleet_context(query,roots,count=4,context_version=2)
    if absolute_frame:
        distance=distance+(context[:,None,8*n:8*n+2]-centers[indices,8*m:8*m+2]).square().sum(-1)
    digits=(codes[:,None]//(radix**torch.arange(m,device=context.device)))%radix
    b,k=indices.shape;p=len(codes)
    programs=torch.zeros((b,k,p,n),dtype=torch.long,device=context.device)
    present=mapping>=0
    for vessel in range(m):
        destination=mapping[:,:,vessel].clamp_min(0)
        contribution=digits[None,None,:,vessel]*present[:,:,None,vessel]
        programs.scatter_add_(3,destination[:,:,None,None].expand(-1,-1,p,1),contribution[...,None].expand(b,k,-1,1))
    valid=~((digits[None,None]>0)&~present[:,:,None]).any(-1)
    if support is not None:valid&=support[indices]
    if role_conditioned:
        # Observational support concerns the roles that a program changes.
        # Extra factual vessels remain inputs to the learned COMPLETE-plan
        # value, but do not invalidate an otherwise observed local action.
        own=context[:,:8*n].reshape(b,n,8)
        source=centers[indices,:8*m].reshape(b,k,m,8)
        matched=own[:,None].expand(-1,k,-1,-1).gather(2,
            mapping.clamp_min(0)[...,None].expand(-1,-1,-1,8))
        role_distance=(matched-source).square().sum(-1)
        changed=digits>0
        task_distance=(context[:,None,-extra:]-centers[indices,-extra:]).square().sum(-1)
        distance=task_distance[:,:,None]+(role_distance[:,:,None]*changed[None,None]).sum(-1)/changed.sum(-1).clamp_min(1)
        mass=torch.exp(-distance/(2.*bandwidth**2))*valid*(distance<=bandwidth**2)
    else:
        mass=torch.exp(-distance/(2.*bandwidth**2))[:,:,None]*valid
    if time_bandwidth is not None:
        mass=mass.masked_fill((context[:,None,-1]-centers[indices,-1]).abs()[:,:,None]>time_bandwidth,0.)
    transferred_variance=None if variance is None else variance[indices].flatten(1,2)
    return programs.flatten(1,2),mass.flatten(1,2),transferred_variance


class PairedOptionTaskDecision(nn.Module):
    """Task-success transitions under complete joint programs and real delays.

    Each terminal observation enters one Bernoulli likelihood. Shared factual
    futures condition repair and damage probabilities; their distribution is
    integrated out before an action is chosen. There is no terminal-difference
    pseudo-observation and no future outcome is supplied by the running task.
    """
    def __init__(self,contexts,factual_contexts,factual_outcomes,factual_mask,
                 options,observed,width=48,rank=4,bandwidth=.15,margin=.02,
                 paired=True,peer_conditioned=True,use_time=True,time_bandwidth=.01,state=None,
                 prior=None,learning_objective='conditional',anchor_contexts=None,
                 anchor_outcomes=None,anchor_option_indices=None,marginal_likelihood=False,
                 leave_anchor_out=False,controls=None,structured_plan=False,predecessor_wait=False):
        super().__init__()
        self.width=width;self.rank=rank;self.bandwidth=bandwidth;self.margin=margin
        self.paired=paired;self.peer_conditioned=peer_conditioned;self.use_time=use_time
        self.time_bandwidth=time_bandwidth
        self.marginal_likelihood=marginal_likelihood
        self.leave_anchor_out=leave_anchor_out
        self.structured_plan=structured_plan
        self.predecessor_wait=predecessor_wait
        for name,value,dtype in [('contexts',contexts,torch.float32),
                ('factual_contexts',factual_contexts,torch.float32),
                ('factual_outcomes',factual_outcomes,torch.float32),
                ('factual_mask',factual_mask,torch.bool),('options',options,torch.long),
                ('observed',observed,torch.bool)]:
            self.register_buffer(name,torch.as_tensor(value,dtype=dtype))
        self.defenders=(self.contexts.shape[-1]-5)//8
        self.register_buffer('controls',torch.zeros((len(self.contexts),self.defenders,6)) if controls is None
            else torch.as_tensor(controls,dtype=torch.float32))
        shape=self.factual_contexts.shape
        if anchor_contexts is None:
            anchor_contexts=torch.zeros((shape[0],shape[1],0,shape[2]))
            anchor_outcomes=torch.zeros((shape[0],shape[1],0))
            anchor_option_indices=torch.empty((shape[0],0),dtype=torch.long)
        self.register_buffer('anchor_contexts',torch.as_tensor(anchor_contexts,dtype=torch.float32))
        self.register_buffer('anchor_outcomes',torch.as_tensor(anchor_outcomes,dtype=torch.float32))
        self.register_buffer('anchor_option_indices',torch.as_tensor(anchor_option_indices,dtype=torch.long))
        program_options=[[-1.,-1.]]+[[left,right] for left in (0.,.5,1.) for right in (0.,.5,1.)]
        plan=SharedProgramPotential(width,paired=False,global_width=11 if structured_plan else 7,
            program_options=program_options,peer_conditioned=peer_conditioned)
        if structured_plan:plan.node[0]=nn.Linear(28,width)
        plan.readout[-1]=nn.Linear(width,2*(rank+1))
        features=self.future_design(self.contexts,torch.arange(len(self.contexts)))
        self.network=nn.ModuleDict(dict(plan=plan,
            future=nn.Sequential(nn.Linear(features.shape[-1],32),nn.SiLU(),nn.Linear(32,rank),nn.Tanh())))
        weight=self.factual_mask[...,None].to(features.dtype)
        center=(features*weight).sum(1)/weight.sum(1).clamp_min(1.)
        scale=(((features-center[:,None]).square()*weight).sum(1)/weight.sum(1).clamp_min(1.)).sqrt()
        self.register_buffer('future_center',center)
        self.register_buffer('future_scale',scale.clamp_min(.01))
        if state is not None:self.network.load_state_dict(state)
        if prior is None:self.prior=None
        elif prior.get('kind')=='paired_option_task':
            self.prior=PairedOptionTaskDecision(**{k:v for k,v in prior.items() if k!='kind'})
        else:self.prior=SuccessorTaskDecision(**prior)

    def future_features(self,context,following):
        difference=following-context
        nodes=difference[...,:8*self.defenders].reshape(*difference.shape[:-1],self.defenders,8)
        return torch.cat((nodes.mean(-2),nodes.square().mean(-2).sqrt(),difference[...,-5:]),-1)

    def future_design(self,context,roots,futures=None):
        if futures is None:
            factual=self.factual_contexts[roots];anchors=self.anchor_contexts[roots]
            outcomes=self.anchor_outcomes[roots]
            features=self.future_features(context[:,None],factual)
            anchor_features=self.future_features(context[:,None,None],anchors)
        else:
            factual=self.factual_contexts[roots,futures];anchors=self.anchor_contexts[roots,futures]
            outcomes=self.anchor_outcomes[roots,futures]
            features=self.future_features(context,factual)
            anchor_features=self.future_features(context[:,None],anchors)
        if not self.anchor_option_indices.shape[1]:return features
        codes=self.options[self.anchor_option_indices[roots],0]
        digits=(codes[...,None]//(10**torch.arange(self.defenders,device=context.device)))%10
        programs=F.one_hot(digits,10).to(features.dtype).flatten(-2)
        if futures is None:programs=programs[:,None].expand(-1,features.shape[1],-1,-1)
        anchors=torch.cat((anchor_features,outcomes[...,None],programs),-1)
        return torch.cat((features,anchors.flatten(-2)),-1)

    def parameters_for_options(self,context,options,controls=None):
        extended=torch.cat((context,options[...,1:].to(context.dtype)/200.),-1)
        plan=self.network['plan']
        if self.structured_plan:
            n=(context.shape[-1]-5)//8;rows=context[:,:8*n].reshape(-1,n,8)
            programs=(options[:,0,None]//(10**torch.arange(n,device=context.device)))%10
            source,rule,factual=controls.split(2,-1)
            preference=plan.program_options[programs]
            preference=torch.where(preference>=0.,preference,factual)
            baseline=factual*source+(1.-factual)*rule
            command=preference*source+(1.-preference)*rule
            own=plan.node(torch.cat((rows,F.one_hot(programs,10).to(rows.dtype),controls,
                command,command-baseline),-1))
            if self.peer_conditioned:peers=own
            else:
                peers=plan.node(torch.cat((rows,F.one_hot(torch.zeros_like(programs),10).to(rows.dtype),
                    controls,baseline,torch.zeros_like(baseline)),-1))
            messages=plan.message(torch.cat((own[:,:,None].expand(-1,-1,n,-1),
                peers[:,None].expand(-1,n,-1,-1),rows[:,None]-rows[:,:,None]),-1))
            mask=~torch.eye(n,dtype=torch.bool,device=context.device)[None,:,:,None]
            encoded=plan.update(torch.cat((own,(messages*mask).sum(2)/max(1,n-1),
                messages.masked_fill(~mask,-1e6).amax(2)),-1))
            # A temporal option is a response window, not a tiny perturbation
            # of the global clock. Log-time bases resolve near-term waiting
            # and long commitments with the same task-success likelihood.
            delay=options[:,1].to(context.dtype).log1p()
            span=options[:,2].to(context.dtype).log1p()
            delay_centers=context.new_tensor((0.,5.,20.)).log1p()
            span_centers=context.new_tensor((10.,40.,200.)).log1p()
            timing=torch.cat((torch.exp(-((delay[:,None]-delay_centers)/.7).square()),
                torch.exp(-((span[:,None]-span_centers)/.7).square())),-1)
            global_state=torch.cat((context[:,-5:],timing),-1)
            if self.peer_conditioned:
                pooled=torch.cat((encoded.mean(1),encoded.amax(1),global_state,
                    context.new_full((len(context),1),n/6.)),-1)
                raw=plan.readout(pooled)
            else:
                pooled=torch.cat((encoded,peers.mean(1)[:,None].expand(-1,n,-1),
                    global_state[:,None].expand(-1,n,-1),context.new_full((len(context),n,1),n/6.)),-1)
                raw=plan.readout(pooled).mean(1)
        elif self.peer_conditioned:raw=plan.logits(extended,options[...,0])
        else:
            n=self.defenders;rows=context[:,:8*n].reshape(-1,n,8)
            programs=(options[:,0,None]//(10**torch.arange(n,device=context.device)))%10
            own=plan.node(torch.cat((rows,F.one_hot(programs,10).to(rows.dtype)),-1))
            factual=plan.node(torch.cat((rows,F.one_hot(torch.zeros_like(programs),10).to(rows.dtype)),-1))
            messages=plan.message(torch.cat((own[:,:,None].expand(-1,-1,n,-1),
                factual[:,None].expand(-1,n,-1,-1),rows[:,None]-rows[:,:,None]),-1))
            mask=~torch.eye(n,dtype=torch.bool,device=context.device)[None,:,:,None]
            encoded=plan.update(torch.cat((own,(messages*mask).sum(2)/max(1,n-1),
                messages.masked_fill(~mask,-1e6).amax(2)),-1))
            pooled=torch.cat((encoded,factual.mean(1)[:,None].expand(-1,n,-1),
                extended[:,-7:,None].transpose(1,2).expand(-1,n,-1),
                context.new_full((len(context),n,1),n/6.)),-1)
            raw=plan.readout(pooled).mean(1)
        return raw.reshape(-1,2,self.rank+1)

    def conditional_logits(self,context,options,roots,futures):
        parameters=self.parameters_for_options(context,options,self.controls[roots])
        if not self.paired:return parameters[:,0,0]
        features=self.future_design(context,roots,futures)
        normalized=(features-self.future_center[roots])/self.future_scale[roots]
        latent=self.network['future'](self.hide_target_anchor(normalized,options,roots))
        logits=parameters[:,:,0]+(parameters[:,:,1:]*latent[:,None]).sum(-1)
        factual=self.factual_outcomes[roots,futures]
        return torch.where(factual>.5,-logits[:,1],logits[:,0])

    def hide_target_anchor(self,features,options,roots):
        if not self.leave_anchor_out or not self.anchor_option_indices.shape[1]:return features
        anchors=self.options[self.anchor_option_indices[roots]]
        visible=~(anchors==options[:,None]).all(-1)
        block=(features.shape[-1]-21)//anchors.shape[1]
        keep=torch.cat((torch.ones((len(roots),21),dtype=torch.bool,device=features.device),
            visible[...,None].expand(-1,-1,block).flatten(1)),-1)
        if features.ndim==3:keep=keep[:,None]
        return features*keep

    def marginal_values(self,context,options,roots,controls=None):
        if controls is None:controls=self.controls[roots]
        parameters=self.parameters_for_options(context,options,controls)
        features=self.future_design(context,roots)
        normalized=(features-self.future_center[roots,None])/self.future_scale[roots,None]
        latent=self.network['future'](self.hide_target_anchor(normalized,options,roots))
        logits=parameters[:,:,0][:,None,:]+torch.einsum('bdr,bfr->bfd',parameters[:,:,1:],latent)
        probability=torch.where(self.factual_outcomes[roots]>.5,(-logits[...,1]).sigmoid(),logits[...,0].sigmoid())
        mask=self.factual_mask[roots]
        return (probability*mask).sum(-1)/mask.sum(-1).clamp_min(1)

    def option_values(self,context,roots,controls=None):
        batch=len(context);count=len(self.options)
        if controls is None:controls=self.controls[roots]
        parameters=self.parameters_for_options(context[:,None].expand(-1,count,-1).reshape(batch*count,-1),
            self.options[None].expand(batch,-1,-1).reshape(-1,3),
            controls[:,None].expand(-1,count,-1,-1).reshape(batch*count,self.defenders,6)).reshape(batch,count,2,self.rank+1)
        mask=self.factual_mask[roots];factual=self.factual_outcomes[roots]
        if self.paired or self.marginal_likelihood:
            features=self.future_design(context,roots)
            latent=self.network['future']((features-self.future_center[roots,None])/self.future_scale[roots,None])
            logits=parameters[...,0][:,:,None,:]+torch.einsum('bcdr,bfr->bcfd',parameters[...,1:],latent)
            probability=torch.where(factual[:,None,:]>.5,(-logits[...,1]).sigmoid(),logits[...,0].sigmoid())
            value=(probability*mask[:,None]).sum(-1)/mask.sum(-1)[:,None].clamp_min(1)
        else:value=parameters[:,:,0,0].sigmoid()
        factual_value=(factual*mask).sum(-1)/mask.sum(-1).clamp_min(1)
        value=torch.where((self.options[:,0]==0)[None],factual_value[:,None],value)
        if self.anchor_option_indices.shape[1]:
            columns=self.anchor_option_indices[roots]
            if self.leave_anchor_out:
                anchor_values=torch.stack([self.marginal_values(context,self.options[columns[:,slot]],roots,controls)
                    for slot in range(columns.shape[1])],-1)
            else:
                anchor_values=(self.anchor_outcomes[roots]*mask[...,None]).sum(1)/mask.sum(-1)[:,None].clamp_min(1)
            value=value.scatter(1,columns,anchor_values)
        return value

    def forward(self,packet,action):
        if self.prior is not None:action=self.prior(packet,action)
        if action.shape[-1]==14:
            action=action.clone()
            action[...,13]=torch.where(action[...,13]>=0,action[...,13]+1,action[...,13])
        context,order=intervention_task_context(packet,True,True)
        if order.shape[-1]!=self.defenders:return action
        distance=(context[:,None]-self.contexts[None]).square().sum(-1)
        distance=distance.masked_fill((context[:,None,-1]-self.contexts[None,:,-1]).abs()>self.time_bandwidth,
            float('inf'))
        separation,roots=distance.min(-1);supported=separation<=self.bandwidth**2
        if not bool(supported.any()):return action
        controls=torch.cat((action[...,:2],packet['node'][...,11:13],action[...,2:3].expand(-1,-1,2)),-1)
        controls=controls.gather(1,order[...,None].expand(-1,-1,6))
        value=self.option_values(context,roots,controls)
        valid=self.observed[roots].clone()
        if not self.use_time:valid&=(self.options[:,1]==0)[None]
        gain=value-value[:,:1]
        gain=gain.masked_fill(~valid,float('-inf'))
        best,choice=gain.max(-1);accept=supported&(best>self.margin)&(self.options[choice,0]!=0)
        if not bool(accept.any()):return action
        selected=self.options[choice];code,delay,span=selected.unbind(-1)
        programs=(code[:,None]//(10**torch.arange(self.defenders,device=code.device)))%10
        preference=self.network['plan'].program_options[programs]
        preference=preference.gather(1,order.argsort(-1)[...,None].expand(-1,-1,2))
        one=torch.ones_like(action[...,:1]);free=-one
        duration=(delay+span)[:,None,None]*one
        first=torch.where((delay==0)[:,None,None],preference,free.expand(-1,-1,2))
        replacement=torch.cat((action[...,:3],free,duration,2.*free,
            torch.zeros_like(action[...,:2]),delay[:,None,None]*one,first,preference),-1)
        if action.shape[-1]==3:
            action=torch.cat((action,free,one,2.*free,torch.zeros_like(action[...,:2]),one,
                free.expand(-1,-1,2),free.expand(-1,-1,2)),-1)
        if self.predecessor_wait or action.shape[-1]==14:
            replacement=torch.cat((replacement,one if self.predecessor_wait else free),-1)
            if action.shape[-1]==13:action=torch.cat((action,free),-1)
        return torch.where(accept[:,None,None],replacement,action)


def fit_paired_option_task_policy(data,output,*,updates=3000,width=48,rank=4,
        seed=42,device='cpu',paired=True,peer_conditioned=True,use_time=True,
        bandwidth=.15,margin=.02,time_bandwidth=.01):
    """Fit complete temporal plans using unique factual-conditioned outcomes."""
    import copy
    import numpy as np
    roots=data['roots'];vocabulary=sorted({tuple(option) for root in roots for option in root['options']})
    if vocabulary[0]!=(0,0,0):raise ValueError('The first option must be the actual predecessor continuation.')
    lookup={option:i for i,option in enumerate(vocabulary)}
    contexts=np.asarray([root['context'] for root in roots]);future_count=max(len(root['futures']) for root in roots)
    following=np.zeros((len(roots),future_count,contexts.shape[-1]),np.float32)
    factual=np.zeros((len(roots),future_count),np.float32);mask=np.zeros_like(factual,bool)
    anchor_count=min(len(root.get('anchors',())) for root in roots)
    anchor_contexts=np.zeros((len(roots),future_count,anchor_count,contexts.shape[-1]),np.float32)
    anchor_outcomes=np.zeros((len(roots),future_count,anchor_count),np.float32)
    anchor_columns=np.zeros((len(roots),anchor_count),np.int64)
    observed=np.zeros((len(roots),len(vocabulary)),bool);samples=[]
    for index,root in enumerate(roots):
        for future in range(len(root['futures'])):
            item=root['outcomes'][(0,future)]
            following[index,future]=item['following'];factual[index,future]=item['success'];mask[index,future]=True
        anchors=root.get('anchors',())[:anchor_count]
        for slot,option_index in enumerate(anchors):
            anchor_columns[index,slot]=lookup[tuple(root['options'][option_index])]
            for future in range(len(root['futures'])):
                item=root['outcomes'][(option_index,future)]
                anchor_contexts[index,future,slot]=item['following']
                anchor_outcomes[index,future,slot]=item['success']
        for option_index,option in enumerate(root['options']):
            column=lookup[tuple(option)]
            observed[index,column]=any((option_index,f) in root['outcomes'] for f in range(len(root['futures'])))
            if option_index==0:continue
            for future in range(len(root['futures'])):
                item=root['outcomes'].get((option_index,future))
                if item is not None:samples.append((index,column,future,float(item['success'])))
    torch.manual_seed(seed);rng=np.random.default_rng(seed+104729)
    record=dict(kind='paired_option_task',contexts=contexts,factual_contexts=following,
        factual_outcomes=factual,factual_mask=mask,options=vocabulary,observed=observed,
        width=width,rank=rank,bandwidth=bandwidth,margin=margin,time_bandwidth=time_bandwidth,paired=paired,
        peer_conditioned=peer_conditioned,use_time=use_time,
        anchor_contexts=anchor_contexts,anchor_outcomes=anchor_outcomes,anchor_option_indices=anchor_columns,
        marginal_likelihood=not paired,leave_anchor_out=True,
        controls=np.asarray([root['controls'] for root in roots]) if all('controls' in root for root in roots) else None,
        structured_plan=all('controls' in root for root in roots),
        predecessor_wait=data.get('recipe',{}).get('wait_continuation')=='predecessor',
        learning_objective='conditional' if paired else 'marginal',
        prior=copy.deepcopy(data['model'].get('successor_task_value')))
    net=PairedOptionTaskDecision(**{k:v for k,v in record.items() if k!='kind'}).to(device)
    indices=torch.as_tensor(samples,dtype=torch.float32,device=device)
    root_ids=indices[:,0].long();option_ids=indices[:,1].long();future_ids=indices[:,2].long();targets=indices[:,3]
    optimizer=torch.optim.AdamW(net.network.parameters(),lr=8e-4,weight_decay=.005);last={}
    with torch.enable_grad():
        for step in range(updates):
            batch=torch.as_tensor(rng.integers(len(samples),size=256),device=device)
            r=root_ids[batch];q=option_ids[batch];f=future_ids[batch]
            if paired:
                logits=net.conditional_logits(net.contexts[r],net.options[q],r,f)
                loss=F.binary_cross_entropy_with_logits(logits,targets[batch])
            else:
                probability=net.marginal_values(net.contexts[r],net.options[q],r).clamp(1e-6,1.-1e-6)
                loss=F.binary_cross_entropy(probability,targets[batch])
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.network.parameters(),5.);optimizer.step()
            if (step+1)%500==0 or step+1==updates:
                last=dict(phase='paired_temporal_task_fit',update=step+1,paired=paired,
                    peer_conditioned=peer_conditioned,task_nll=float(loss.detach()))
                print(last,flush=True)
    record['state']={k:v.detach().cpu() for k,v in net.network.state_dict().items()}
    model=copy.deepcopy(data['model']);model['successor_task_value']=record
    removal=tuple(model.get('mechanism_ablation',(False,False,False)))+(False,)*3
    model['mechanism_ablation']=(removal[0],False,removal[2])
    model.update(gate_period=1,physical_interventions=True,fixed_intervention_clock=True,success_objective=True)
    model['successor_task_fit']=dict(paired=paired,peer_conditioned=peer_conditioned,use_time=use_time,
        roots=len(roots),unique_terminal_observations=sum(len(root['outcomes']) for root in roots),
        conditional_targets=len(samples),anchor_plans_per_root=anchor_count,
        duplicate_terminal_difference_labels=0,updates=updates,seed=seed,
        task_target='joint-program success conditioned on shared factual/anchor task fates and motion, then marginalized',
        temporal_options='actual immediate and delayed interventions followed by the same predecessor',
        last=last,deployed_parameters=sum(p.numel() for p in net.network.parameters()))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);torch.save(model,output)
    return model


@torch.no_grad()
def paired_effect_controls(actor,packet,action):
    """Condition intervention effects on the command the predecessor executes."""
    source=action[...,:2];rule=packet['node'][...,11:13]
    physical=action[...,:3]
    if actor.generic_fusion_adapter is not None:
        physical=actor.generic_fusion_adapter.control(actor,packet,physical)
    if getattr(actor,'actuator_fusion_refinement',None):
        query=dict(packet)
        query['_preferred_actuator_coefficients']=action[...,2:3].expand(-1,-1,2)
        physical=refine_actuator_interactions(actor,query,action[...,:3],physical,
            **actor.actuator_fusion_refinement)
    factual=physical[...,:2]*physical[...,2:3]+rule*(1.-physical[...,2:3])
    return torch.cat((source,rule,factual),-1)


def paired_effect_programs(packet,order,fleet_programs=False):
    programs=successor_candidate_programs(packet,order,True,10)
    if fleet_programs:
        # Fleet-wide alternatives expose coordinated task effects that a
        # library of isolated one-boat changes cannot identify.
        b,n=order.shape
        joint=torch.arange(1,10,device=order.device)[None,:,None].expand(b,-1,n)
        programs=torch.cat((programs,joint),1)
    return programs


def paired_control_responses(context,controls,programs,timing):
    """Nominal paired motion features, calibrated by terminal task outcomes.

    Both branches start from public measured motion and refresh neither
    source policy inside this short feature forecast. The learned task value,
    rather than a prescribed geometric cost, determines their usefulness.
    """
    n=programs.shape[-1];rows=context[:,:8*n].reshape(-1,n,8)
    source,rule,factual=controls.split(2,-1)
    choices=context.new_tensor([[-1.,-1.]]+[[a,b] for a in (0.,.5,1.) for b in (0.,.5,1.)])
    preference=choices[programs]
    command=torch.where((programs>0)[...,None],preference*source+(1.-preference)*rule,factual)
    heading=torch.atan2(rows[...,5],rows[...,4])
    initial=torch.cat((30.*rows[...,:2],heading[...,None],3.*rows[...,2:4],rows[...,6:7]),-1)
    state=initial[:,None].expand(-1,2,-1,-1).clone();snapshots=[]
    delay,span=timing.to(context.dtype).unbind(-1)
    for step in range(20):
        held=(step>=delay)&(step<delay+span)
        selected=torch.where(held[:,None,None],command,factual)
        force=750.*torch.stack((factual,selected),1)+250.
        x,y,yaw,vx,vy,r=state.unbind(-1)
        x=x+.2*vx;y=y+.2*vy;yaw=yaw+.2*r
        cosine,sine=yaw.cos(),yaw.sin();u=cosine*vx+sine*vy;v=-sine*vx+cosine*vy
        du=(400.*r*v-(100.+150.*u.abs())*u+force.sum(-1))/380.
        dv=(-380.*r*u-(100.+150.*v.abs())*v)/400.
        dr=(-20.*u*v-(980.+950.*r.abs())*r+1.25*(force[...,0]-force[...,1]))/1430.
        u=u+.2*du;v=v+.2*dv
        state=torch.stack((x,y,yaw,cosine*u-sine*v,sine*u+cosine*v,r+.2*dr),-1)
        if step in (4,9,19):snapshots.append(state)
    path=torch.stack(snapshots,-2)
    time=context.new_tensor((1.,2.,4.))
    target=context[:,-3:-1]*5.
    position=path[...,:2]-target[:,None,None,None]*time[None,None,None,:,None]
    body=torch.stack((path[...,2].cos(),path[...,2].sin()),-1)
    absolute=torch.cat((position/30.,path[...,3:5]/5.,body,path[...,5:6]/2.),-1)
    delta=torch.cat(((path[:,1,...,:2]-path[:,0,...,:2])/5.,
        (path[:,1,...,3:5]-path[:,0,...,3:5])/3.,body[:,1]-body[:,0],
        (path[:,1,...,5:6]-path[:,0,...,5:6])/2.),-1)
    changed=torch.cat((absolute[:,1],delta),-1).flatten(-2)
    baseline=torch.cat((absolute[:,0],torch.zeros_like(delta)),-1).flatten(-2)
    return changed,baseline,path


class PairedEffectGraph(nn.Module):
    """Fleet-shared likelihood of repair, agreement and damage of a full plan."""
    def __init__(self,width=64,peer_conditioned=True,response_features=False):
        super().__init__();self.width=width;self.peer_conditioned=peer_conditioned
        self.response_features=response_features
        self.node=nn.Sequential(nn.Linear(29+42*response_features,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.message=nn.Sequential(nn.Linear(2*width+10+18*response_features,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        self.update=nn.Sequential(nn.Linear(3*width,width),nn.SiLU())
        self.readout=nn.Sequential(nn.Linear(4*width+12,width),nn.SiLU(),nn.Linear(width,3))
        self.register_buffer('coefficients',torch.tensor([[-1.,-1.]]+
            [[a,b] for a in (0.,.5,1.) for b in (0.,.5,1.)]))

    def forward(self,context,controls,programs,timing,responses=None):
        n=programs.shape[-1];rows=context[:,:8*n].reshape(-1,n,8)
        source,rule,factual=controls.split(2,-1)
        held=programs>0;preference=self.coefficients[programs]
        command=torch.where(held[...,None],preference*source+(1.-preference)*rule,factual)
        if self.response_features and responses is None:
            responses=paired_control_responses(context,controls,programs,timing)
        def encode(q,u,response):
            features=(rows,controls,F.one_hot(q,10).to(rows.dtype),u,
                u-factual,(q>0).to(rows.dtype)[...,None])
            if response is not None:features=features+(response,)
            return self.node(torch.cat(features,-1))
        own=encode(programs,command,responses[0] if responses is not None else None)
        baseline=encode(torch.zeros_like(programs),factual,responses[1] if responses is not None else None)
        peers=own if self.peer_conditioned else baseline
        difference=rows[:,:,None]-rows[:,None]
        distance=difference[...,:2].square().sum(-1).clamp_min(1e-6).sqrt()
        closing=(difference[...,:2]*difference[...,2:4]).sum(-1)/distance
        geometry=torch.cat((difference,distance[...,None],closing[...,None]),-1)
        if responses is not None:
            own_path=responses[2][:,1];peer_path=responses[2][:,1 if self.peer_conditioned else 0]
            displacement=own_path[:,:,None,...,:2]-peer_path[:,None,...,:2]
            velocity=own_path[:,:,None,...,3:5]-peer_path[:,None,...,3:5]
            separation=displacement.square().sum(-1).clamp_min(1e-6).sqrt()
            approach=(displacement*velocity).sum(-1)/separation
            forecast=torch.cat((displacement/30.,velocity/5.,separation[...,None]/10.,
                approach[...,None]/5.),-1).flatten(-2)
            geometry=torch.cat((geometry,forecast),-1)
        message=self.message(torch.cat((own[:,:,None].expand(-1,-1,n,-1),
            peers[:,None].expand(-1,n,-1,-1),geometry),-1))
        mask=~torch.eye(n,dtype=torch.bool,device=context.device)[None,:,:,None]
        encoded=self.update(torch.cat((own,(message*mask).sum(2)/max(1,n-1),
            message.masked_fill(~mask,-1e6).amax(2)),-1))
        centers=context.new_tensor((0.,5.,20.,10.,40.,200.)).log1p()
        times=timing.to(context.dtype).log1p()
        basis=torch.cat((torch.exp(-((times[:,:1]-centers[None,:3])/.8).square()),
            torch.exp(-((times[:,1:]-centers[None,3:])/.8).square())),-1)
        global_state=torch.cat((context[:,-5:],basis,context.new_full((len(context),1),n/6.)),-1)
        if self.peer_conditioned:
            changed=(encoded*held[...,None]).sum(1)/held.sum(1,keepdim=True).clamp_min(1)
            pooled=torch.cat((encoded.mean(1),encoded.amax(1),baseline.mean(1),changed,global_state),-1)
            return self.readout(pooled)
        # Each additive factor sees public teammate states and their factual
        # commands, but no teammate intervention. Summing factors preserves
        # the possibility that several independent changes jointly matter.
        pooled=torch.cat((encoded,baseline,baseline.mean(1)[:,None].expand(-1,n,-1),
            own-baseline,global_state[:,None].expand(-1,n,-1)),-1)
        return (self.readout(pooled)*held[...,None]).sum(1)


class SharedPairedEffectDecision(nn.Module):
    """Portable paired task improvement with a real delayed-action vocabulary.

    There is no training-root admission gate or stored successful action.
    The three probabilities estimate E[Y(plan)-Y(factual)] directly. One
    observed paired terminal transition supplies one categorical likelihood.
    """
    def __init__(self,width=64,ensemble_size=3,states=None,spans=(10,40,200),delays=(0,5),
                 peer_conditioned=True,use_time=True,margin=.03,uncertainty_weight=.5,
                 decision_period=5,intervention_budget=1,learning_objective='conditional',response_features=False,
                 fleet_programs=False,class_prior=None):
        super().__init__();self.width=width;self.peer_conditioned=peer_conditioned
        self.response_features=response_features
        self.fleet_programs=fleet_programs
        self.register_buffer('class_prior',torch.tensor(class_prior if class_prior is not None else [1.,1.,1.]))
        self.models=nn.ModuleList([PairedEffectGraph(width,peer_conditioned,response_features) for _ in range(ensemble_size)])
        if states is not None:
            for model,state in zip(self.models,states):model.load_state_dict(state)
        self.spans=tuple(spans);self.delays=tuple(delays);self.use_time=use_time
        self.margin=margin;self.uncertainty_weight=uncertainty_weight
        self.decision_period=decision_period;self.intervention_budget=intervention_budget
        self.prior=None

    def eligible(self,packet):
        tick=(packet['node'][:,0,15]*400.).round().long()
        active=tick%self.decision_period==0
        if self.intervention_budget and '_conditional_count' in packet:
            active&=packet['_conditional_count'].flatten()<self.intervention_budget
        return active

    @torch.no_grad()
    def prepare_packet(self,actor,packet,action):
        if not bool(self.eligible(packet).any()):return packet
        result=dict(packet);result['_paired_effect_controls']=paired_effect_controls(actor,packet,action)
        return result

    def gains(self,context,controls,programs,timing):
        predictions=[]
        responses=paired_control_responses(context,controls,programs,timing) if self.response_features else None
        for model in self.models:
            probability=(model(context,controls,programs,timing,responses)+self.class_prior.clamp_min(1e-8).log()).softmax(-1)
            predictions.append(probability[:,2]-probability[:,0])
        value=torch.stack(predictions)
        return value.mean(0),value.std(0,unbiased=False)

    @torch.no_grad()
    def forward(self,packet,action):
        eligible=self.eligible(packet)
        if not bool(eligible.any()):return action
        context,order=intervention_task_context(packet,True,True);b,n=order.shape
        controls=packet['_paired_effect_controls'].gather(1,order[...,None].expand(-1,-1,6))
        programs=paired_effect_programs(packet,order,self.fleet_programs)[:,1:]
        timings=torch.tensor([(delay,span) for span in self.spans
            for delay in self.delays if self.use_time or delay==0],device=action.device)
        count=programs.shape[1];number=len(timings)
        query=programs[:,:,None].expand(-1,-1,number,-1).reshape(b,-1,n)
        timing=timings[None,None].expand(b,count,-1,-1).reshape(b,-1,2)
        values=[]
        for start in range(0,query.shape[1],96):
            q=query[:,start:start+96];t=timing[:,start:start+96];k=q.shape[1]
            mean,deviation=self.gains(context[:,None].expand(-1,k,-1).flatten(0,1),
                controls[:,None].expand(-1,k,-1,-1).flatten(0,1),q.flatten(0,1),t.flatten(0,1))
            score=(mean-self.uncertainty_weight*deviation).reshape(b,k)
            values.append(score-.002*(q>0).sum(-1))
        best,choice=torch.cat(values,-1).max(-1)
        accept=eligible&(best>self.margin)
        if not bool(accept.any()):return action
        batch=torch.arange(b,device=action.device);selected=query[batch,choice]
        delay,span=timing[batch,choice].unbind(-1)
        preference=self.models[0].coefficients[selected]
        preference=preference.gather(1,order.argsort(-1)[...,None].expand(-1,-1,2))
        one=torch.ones_like(action[...,:1]);free=-one
        first=torch.where((delay==0)[:,None,None],preference,free.expand(-1,-1,2))
        proposed=torch.cat((action[...,:3],free,(delay+span)[:,None,None]*one,2.*free,
            torch.zeros_like(action[...,:2]),delay[:,None,None]*one,first,preference,one),-1)
        factual=torch.cat((action[...,:3],free,one,2.*free,torch.zeros_like(action[...,:2]),one,
            free.expand(-1,-1,2),free.expand(-1,-1,2),free),-1)
        return torch.where(accept[:,None,None],proposed,factual)


def fit_shared_paired_effect(data,output,*,updates=4000,width=64,ensemble_size=3,seed=42,
        device='cpu',peer_conditioned=True,use_time=True,margin=.03,uncertainty_weight=.5,response_features=False,
        balanced_likelihood=False):
    """Fit paired terminal transitions, with roots as the bootstrap units."""
    import copy
    import numpy as np
    roots=data['roots'];n=(len(roots[0]['context'])-5)//8
    samples=[]
    for root_id,root in enumerate(roots):
        if 'effect_controls' not in root:raise ValueError('Collect actual predecessor commands for paired effects.')
        for (q,future),outcome in root['outcomes'].items():
            if q==0:continue
            factual=root['outcomes'][0,future]['success']
            code,delay,span=root['options'][q]
            samples.append((root_id,code,delay,span,int(outcome['success']-factual)+1))
    samples=np.asarray(samples,dtype=np.int64)
    classes=[np.flatnonzero(samples[:,-1]==label) for label in range(3)]
    available=[label for label,indices in enumerate(classes) if len(indices)]
    class_mass=np.asarray([len(indices)/len(samples) for indices in classes])
    x=torch.as_tensor(np.asarray([root['context'] for root in roots]),device=device,dtype=torch.float32)
    u=torch.as_tensor(np.asarray([root['effect_controls'] for root in roots]),device=device,dtype=torch.float32)
    root_ids=torch.as_tensor(samples[:,0],device=device)
    digits=torch.as_tensor((samples[:,1,None]//(10**np.arange(n)))%10,device=device)
    timing=torch.as_tensor(samples[:,2:4],device=device)
    target=torch.as_tensor(samples[:,4],device=device)
    responses=None
    if response_features:
        with torch.no_grad():
            chunks=[paired_control_responses(x[root_ids[start:start+512]],u[root_ids[start:start+512]],
                digits[start:start+512],timing[start:start+512]) for start in range(0,len(samples),512)]
            responses=tuple(torch.cat([chunk[i] for chunk in chunks]) for i in range(3))
    torch.manual_seed(seed);rng=np.random.default_rng(seed+100003)
    recipe=dict(width=width,ensemble_size=ensemble_size,spans=tuple(data['recipe']['spans']),
        delays=tuple(data['recipe']['delays']),peer_conditioned=peer_conditioned,use_time=use_time,
        margin=margin,uncertainty_weight=uncertainty_weight,decision_period=data['recipe'].get('root_stride',5),
        intervention_budget=1,response_features=response_features,
        fleet_programs=data['recipe'].get('fleet_programs',False),
        class_prior=class_mass.tolist() if balanced_likelihood else None)
    learner=SharedPairedEffectDecision(**recipe).to(device);states=[];diagnostics=[]
    for member,net in enumerate(learner.models):
        root_weight=np.ones(len(roots)) if member==0 else rng.exponential(1.,size=len(roots))
        weights=torch.as_tensor(root_weight,device=device,dtype=torch.float32)
        optimizer=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=.005)
        with torch.enable_grad():
            for update in range(updates):
                labels=rng.choice(available,size=128)
                indices=np.asarray([rng.choice(classes[label]) for label in labels])
                batch=torch.as_tensor(indices,device=device);r=root_ids[batch]
                logits=net(x[r],u[r],digits[batch],timing[batch],
                    None if responses is None else tuple(response[batch] for response in responses))
                mass=np.ones(len(labels)) if balanced_likelihood else class_mass[labels]*len(available)
                importance=torch.as_tensor(mass,device=device,dtype=torch.float32)*weights[r]
                loss=(F.cross_entropy(logits,target[batch],reduction='none')*importance).mean()
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
                if (update+1)%1000==0 or update+1==updates:
                    print(dict(phase='shared_paired_effect_fit',member=member,update=update+1,nll=float(loss.detach())),flush=True)
        states.append({key:value.detach().cpu() for key,value in net.state_dict().items()})
        diagnostics.append(float(loss.detach()))
    recipe['states']=states
    model=copy.deepcopy(data['model']);model['successor_task_value']=dict(kind='shared_paired_effect',
        learning_objective='conditional',**recipe)
    removal=tuple(model.get('mechanism_ablation',(False,False,False)))+(False,)*3
    model['mechanism_ablation']=(removal[0],False,removal[2])
    model.update(gate_period=1,physical_interventions=True,fixed_intervention_clock=True,success_objective=True)
    model['successor_task_fit']=dict(roots=len(roots),unique_terminal_observations=sum(len(r['outcomes']) for r in roots),
        paired_transitions=len(samples),damage_tie_repair_counts=[len(a) for a in classes],
        duplicate_terminal_difference_labels=0,updates=updates,ensemble_size=ensemble_size,seed=seed,
        task_target='paired damage, agreement and repair probabilities; expected terminal success improvement',
        actual_predecessor_command_context=True,root_admission_gate=False,variable_fleet=True,
        nominal_paired_response_features=response_features,balanced_likelihood=balanced_likelihood,last_nll=diagnostics,
        training_data=data.get('training_data'),predecessor_sha256=data.get('predecessor_sha256'))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);torch.save(model,output)
    return model


class OrdinaryJointReturnDecision(nn.Module):
    """Greedy absolute return over one common program/window action space.

    There is no predecessor cascade, root lookup, support weighting, or
    intervention opportunity rule. A factual program is an ordinary action
    in the same maximization. Its window has the same physical meaning as
    in the observations; the underlying source/M1 controller keeps updating.
    """
    def __init__(self,program_model,candidate_codes,candidate_spans,defenders,
                 learning_objective='absolute'):
        super().__init__()
        if learning_objective!='absolute':
            raise ValueError('The ordinary controller requires absolute task returns.')
        self.program_model=load_program_potential(program_model)
        if not isinstance(self.program_model,SharedProgramPotential):
            raise ValueError('Ordinary return execution requires a shared return regressor.')
        if self.program_model.radix!=10 or self.program_model.global_width!=6:
            raise ValueError('Use actuator programs and public context plus duration.')
        self.defenders=int(defenders)
        codes=torch.as_tensor(candidate_codes,dtype=torch.long)
        spans=torch.as_tensor(candidate_spans,dtype=torch.long)
        if codes.ndim!=1 or not len(codes) or codes.shape!=spans.shape:
            raise ValueError('Provide one duration for every complete program.')
        if (codes<0).any() or (codes>=10**self.defenders).any() or (spans<1).any():
            raise ValueError('Invalid complete program or physical execution window.')
        if not (codes==0).any():
            raise ValueError('Include the factual controller as an ordinary candidate.')
        self.register_buffer('candidate_codes',codes)
        self.register_buffer('candidate_spans',spans)

    def candidate_values(self,value_context,codes,action,order):
        return self.program_model(value_context.flatten(0,1),codes.flatten()).reshape(codes.shape)

    def forward(self,packet,action):
        context,order=intervention_task_context(packet,True,True)
        batch,n=order.shape
        if n!=self.defenders or action.shape[-1]!=3:
            raise ValueError('Ordinary return selection needs its source/M1 action and trained fleet size.')
        count=len(self.candidate_codes)
        value_context=torch.cat((context[:,None].expand(-1,count,-1),
            (self.candidate_spans.to(context.dtype)/20.)[None,:,None].expand(batch,-1,-1)),-1)
        codes=self.candidate_codes[None].expand(batch,-1)
        value=self.candidate_values(value_context,codes,action,order)
        choice=value.argmax(-1)
        selected=self.candidate_codes[choice]
        programs=(selected[:,None]//(10**torch.arange(n,device=action.device)))%10
        preferences=self.program_model.program_options[programs]
        preferences=preferences.gather(1,order.argsort(-1)[...,None].expand(-1,-1,2))
        one=torch.ones_like(action[...,:1]);free=-one
        duration=self.candidate_spans[choice,None,None].to(action.dtype)*one
        # This is only the common action decoder. Even program zero is
        # selected by the absolute return, never by a separate waiting rule.
        return torch.cat((action,free,duration,2.*free,torch.zeros_like(action[...,:2]),
            duration,preferences,preferences),-1)


class SuccessorTaskDecision(nn.Module):
    """Choose complete intervention preferences by learned task-success value."""
    def __init__(self,program_model,contexts,span=10,bandwidth=.15,margin=.05,initial_only=True,
                 learning_objective=None,include_all_vessels=False,time_bandwidth=None,
                 candidate_codes=None,candidate_contexts=None,candidate_support=None,
                 absolute_context=False,support_weighted=False,candidate_variance=None,confidence_scale=0.,
                 explicit_wait=False,prior=None,candidate_spans=None,duration_conditioned=False,
                 role_conditioned_support=False,use_factual_support=True):
        super().__init__();self.program_model=load_program_potential(program_model)
        self.register_buffer('contexts',torch.as_tensor(contexts,dtype=torch.float32))
        self.span=span;self.bandwidth=bandwidth;self.margin=margin;self.initial_only=initial_only
        self.include_all_vessels=include_all_vessels
        self.time_bandwidth=time_bandwidth
        self.absolute_context=absolute_context;self.support_weighted=support_weighted
        self.confidence_scale=confidence_scale
        self.explicit_wait=explicit_wait
        self.duration_conditioned=duration_conditioned
        self.role_conditioned_support=role_conditioned_support
        self.use_factual_support=use_factual_support
        if role_conditioned_support and (candidate_codes is None or candidate_contexts is None):
            raise ValueError('Role-conditioned support needs observed complete programs and their contexts.')
        if role_conditioned_support and isinstance(self.program_model,LocalInterventionTaskPotential):
            raise ValueError('Role transfer requires a fleet-shared task-value model, not a fixed-size value table.')
        self.register_buffer('candidate_spans',None if candidate_spans is None else
            torch.as_tensor(candidate_spans,dtype=torch.float32))
        if duration_conditioned and candidate_spans is None:
            raise ValueError('Duration-conditioned task values require their sampled intervention windows.')
        self.prior=None if prior is None else SuccessorTaskDecision(**prior)
        self.register_buffer('candidate_codes',None if candidate_codes is None else torch.as_tensor(candidate_codes,dtype=torch.long))
        self.register_buffer('candidate_contexts',None if candidate_contexts is None else torch.as_tensor(candidate_contexts,dtype=torch.float32))
        self.register_buffer('candidate_support',None if candidate_support is None else torch.as_tensor(candidate_support,dtype=torch.bool))
        self.register_buffer('candidate_variance',None if candidate_variance is None else torch.as_tensor(candidate_variance,dtype=torch.float32))

    @staticmethod
    def summary(context):
        n=(context.shape[-1]-3)//8;rows=context[:,:8*n].reshape(-1,n,8)
        return torch.cat((rows.mean(1),rows.square().mean(1).sqrt(),context[:,8*n:]),-1)

    def forward(self,packet,action):
        if self.prior is not None:action=self.prior(packet,action)
        context,order=intervention_task_context(packet,True,self.absolute_context);b,n=order.shape
        if self.use_factual_support:
            summary=self.summary(context)
            distance=(summary[:,None]-self.contexts[None]).square().sum(-1)
            if self.time_bandwidth is not None:
                distance=distance.masked_fill((summary[:,None,-1]-self.contexts[None,:,-1]).abs()
                    >self.time_bandwidth,float('inf'))
            distance=distance.amin(-1)
            supported=torch.isfinite(distance) if self.role_conditioned_support else distance<=self.bandwidth**2
        else:supported=torch.ones(b,dtype=torch.bool,device=action.device)
        if self.initial_only:supported&=packet['node'][:,0,15]<1e-6
        if not bool(supported.any()):return action
        radix=self.program_model.radix
        if not bool(supported.all()):
            subset={key:value[supported] if isinstance(value,torch.Tensor)
                and value.ndim and len(value)==b else value for key,value in packet.items()}
            selected=self.forward(subset,action[supported])
            if selected.shape[-1]!=action.shape[-1]:
                one=torch.ones_like(action[...,:1]);free=-one
                action=(torch.cat((action,free,one,2.*free,torch.zeros_like(action[...,:2]),one,
                    free.expand(-1,-1,2),free.expand(-1,-1,2)),-1) if radix==10 else
                    torch.cat((action,free,one,free),-1))
            result=action.clone();result[supported]=selected
            return result
        transferred_mass=None;transferred_variance=None
        if self.candidate_codes is None:
            programs=successor_candidate_programs(packet,order,self.include_all_vessels,radix)
        elif self.use_factual_support and self.candidate_contexts is not None and (self.role_conditioned_support
                or context.shape[-1]!=self.candidate_contexts.shape[-1]):
            programs,transferred_mass,transferred_variance=transport_successor_programs(context,self.candidate_contexts,
                self.candidate_codes,self.candidate_support,radix,self.bandwidth,self.absolute_context,self.candidate_variance,
                role_conditioned=self.role_conditioned_support,time_bandwidth=self.time_bandwidth)
        else:
            programs=((self.candidate_codes[:,None]//(radix**torch.arange(n,device=action.device)))%radix)
            programs=programs[None].expand(b,-1,-1)
        count=programs.shape[1]
        codes=(programs*(radix**torch.arange(n,device=action.device))[None,None]).sum(-1)
        value_context=context[:,None].expand(-1,count,-1)
        durations=None
        if self.duration_conditioned:
            durations=self.candidate_spans.repeat(count//len(self.candidate_spans))[None].expand(b,-1)
            value_context=torch.cat((value_context,durations[...,None]/20.),-1)
        value=self.program_model(value_context.flatten(0,1),codes.flatten()).reshape(b,count)
        gain=value-value[:,:1]
        if self.use_factual_support and (self.candidate_support is not None or transferred_mass is not None):
            if transferred_mass is not None:mass=transferred_mass
            else:
                separation=(context[:,None]-self.candidate_contexts[None]).square().sum(-1)
                kernel=torch.exp(-separation/(2*self.bandwidth**2))
                if self.time_bandwidth is not None:
                    kernel=kernel.masked_fill((context[:,None,-1]-self.candidate_contexts[None,:,-1]).abs()
                        >self.time_bandwidth,0.)
                mass=kernel@self.candidate_support.to(kernel.dtype)
            if self.duration_conditioned:
                factual=(codes==0)&(mass>=1e-6)
                factual_index=durations.masked_fill(~factual,float('inf')).argmin(-1)
                gain=value-value.gather(1,factual_index[:,None])
                supported=supported&factual.any(-1)
            else:supported=supported&(mass[:,0]>=1e-6)
            if self.candidate_variance is not None and self.confidence_scale:
                variance=(transferred_variance if transferred_variance is not None else
                    (kernel@self.candidate_variance)/mass.clamp_min(1e-12))
                gain=gain-self.confidence_scale*variance.clamp_min(0.).sqrt()
            if self.support_weighted:gain=gain*mass.clamp_max(1.)
            gain=gain.masked_fill(mass<1e-6,float('-inf'))
        gain-=.005*(programs>0).sum(-1)
        best,choice=gain.max(-1);accept=supported&(best>self.margin)
        selected=programs[torch.arange(b,device=action.device),choice]
        selected_span=self.span if durations is None else durations.gather(1,choice[:,None])[:,:,None]
        if radix==10:
            preferences=self.program_model.program_options[selected]
            preferences=preferences.gather(1,order.argsort(-1)[...,None].expand(-1,-1,2))
            one=torch.ones_like(action[...,:1]);free=-one
            replacement=torch.cat((action[...,:3],free,selected_span*one,2.*free,
                torch.zeros_like(action[...,:2]),selected_span*one,preferences,preferences),-1)
            if action.shape[-1]==3:
                action=torch.cat((action,free,one,2.*free,torch.zeros_like(action[...,:2]),one,
                    free.expand(-1,-1,2),free.expand(-1,-1,2)),-1)
            if self.explicit_wait:
                wait=torch.cat((action[...,:3],free,self.span*one,2.*free,
                    torch.zeros_like(action[...,:2]),self.span*one,free.expand(-1,-1,2),free.expand(-1,-1,2)),-1)
                action=torch.where(supported[:,None,None],wait,action)
            return torch.where(accept[:,None,None],replacement,action)
        gates=torch.where(selected>0,(selected.to(action.dtype)-1.)*.5,-1.)
        gates=gates.gather(1,order.argsort(-1))
        replacement=torch.cat((action[...,:3],gates[...,None],torch.ones_like(gates[...,None])*selected_span,gates[...,None]),-1)
        if action.shape[-1]==3:
            free=torch.full_like(action[...,:1],-1.)
            action=torch.cat((action,free,torch.ones_like(free),free),-1)
        if self.explicit_wait:
            free=torch.full_like(action[...,:1],-1.)
            wait=torch.cat((action[...,:3],free,-self.span*free,2.*free),-1)
            action=torch.where(supported[:,None,None],wait,action)
        return torch.where(accept[:,None,None],replacement,action)


def fit_successor_task_policy(data,output,*,ordinary=False,updates=5000,seed=42,device='cpu',
                             bandwidth=.15,margin=.05,decision_weight=1.,transport=False,
                             task_contrasts=False,difference_weight=None,latent_transport=False,
                             absolute_context=None,support_weighted=False,task_width=None,response_width=24,
                              shared_task_value=False,paired_graph=False,confidence_scale=0.,retain_predecessor=False,
                              paired_calibration=False,covariance_shrink=.5,paired_ordering=None,
                              future_precision=.5,paired_future_rank=0,residual_calibration=None,
                              paired_joint_likelihood=False,factual_task=False,factual_response=False,
                              factual_features=6,factual_contrast=False,factual_rank=0,
                              factual_normalization=False):
    """Fit conditional task ordering, or an explicitly selected estimator.

    The default marginalizes shared futures and calibrates completely
    observed program values. Ordinary learning keeps its absolute-return
    objective. Alternative estimators remain explicit research comparisons.
    """
    import copy
    import numpy as np
    import develop_source_fusion as development
    if ordinary:
        if retain_predecessor or factual_task or factual_response or factual_contrast:
            raise ValueError('Ordinary return learning cannot retain conditional stages or factual-effect learners.')
        datasets=data.get('duration_datasets') or [data]
        return fit_ordinary_task_return_policy(data['model'],datasets,output,width=task_width or 64,
            updates=updates,seed=seed,device=device)
    if factual_response and not factual_task:
        raise ValueError('Paired physical-response supervision requires the paired factual task-value learner.')
    if absolute_context is None:absolute_context=not (transport or latent_transport)
    if paired_ordering is None:
        paired_ordering=not (ordinary or transport or latent_transport or task_contrasts
            or shared_task_value or paired_graph or paired_future_rank or paired_calibration
            or paired_joint_likelihood or factual_task or decision_weight!=1. or task_width is not None or response_width!=24)
    if difference_weight is None:difference_weight=10. if paired_ordering else 4.
    if residual_calibration is None:residual_calibration='complete' if paired_ordering else False
    if not data.get('completed',False):
        raise ValueError('Finish collecting the paired task branches before fitting the value.')
    predecessor=copy.deepcopy(data['model'])
    previous=predecessor.pop('successor_task_value',None) if data.get('policy_iteration',False) else None
    if retain_predecessor and previous is None:
        raise ValueError('A retained value stage requires its own sampled predecessor.')
    if development.joint_value_objectives(predecessor):
        raise ValueError('Whole mechanism comparisons cannot inherit another value learner.')
    if (previous is not None and development.joint_value_objectives({'successor_task_value':previous})
            !={'absolute' if ordinary else 'conditional'}):
        raise ValueError('Each policy-iteration control must collect its own continuation data.')
    if absolute_context and (transport or latent_transport):
        raise ValueError('The transport decoder requires its original relative context.')
    duration_data=data.get('duration_datasets')
    duration_conditioned=duration_data is not None
    if factual_task and ordinary:
        raise ValueError('Paired factual task learning requires a conditional objective.')
    global_width=(5 if absolute_context else 3)+int(duration_conditioned)
    if duration_conditioned:
        if not ordinary and not factual_task and (not paired_graph or paired_ordering or paired_future_rank):
            raise ValueError('Duration-conditioned decisions require a joint task-value estimator.')
        identity=development.joint_continuation_identity(data['model'])
        examples=[];physical_contexts=[];example_spans=[]
        for part in duration_data:
            if not part.get('completed',False):raise ValueError('Finish every sampled intervention duration before fitting.')
            if development.joint_continuation_identity(part['model'])!=identity:
                raise ValueError('Intervention durations must be measured against the same executing predecessor.')
            if part.get('program_options')!=data.get('program_options'):
                raise ValueError('Duration comparisons require the same complete fusion program library.')
            part_examples=successor_task_examples(part,require_candidate_support=(part.get('focus_codes') is None
                and part.get('candidate_programs') is None),absolute_frame=absolute_context)
            declared=part.get('candidate_programs')
            if declared is not None and len(declared)!=len(part_examples):
                raise ValueError('Declare candidates for every intervention context and duration.')
            for root,example in enumerate(part_examples):
                required=(set(declared[root]) if declared is not None else
                    set(map(int,part['focus_codes']))|{0} if part.get('focus_codes') is not None else set())
                if not required<={sample[0] for sample in example['samples'].values()}:
                    raise ValueError('Every program/window candidate needs its own observed task outcomes.')
                physical_contexts.append(example['context'].copy());example_spans.append(int(part['span']))
                example['context']=np.r_[example['context'],float(part['span'])/20.].astype(np.float32)
                if factual_task:
                    example['samples']={key:(q,np.r_[following,float(part['span'])/20.].astype(np.float32),outcome)
                        for key,(q,following,outcome) in example['samples'].items()}
                examples.append(example)
    else:
        examples=successor_task_examples(data,require_candidate_support=(data.get('focus_codes') is None
            and data.get('candidate_programs') is None),absolute_frame=absolute_context)
        physical_contexts=[example['context'] for example in examples]
        example_spans=[int(data['span'])]*len(examples)
    if not duration_conditioned and data.get('candidate_programs') is not None:
        if len(data['candidate_programs'])!=len(examples):raise ValueError('Declare candidates for every intervention context.')
        for required,example in zip(data['candidate_programs'],examples):
            if not set(required)<={sample[0] for sample in example['samples'].values()}:
                raise ValueError('Declared candidates require observed task outcomes.')
    if not duration_conditioned and data.get('focus_codes') is not None:
        required=set(map(int,data['focus_codes']))|{0}
        for example in examples:
            if not required<={sample[0] for sample in example['samples'].values()}:
                raise ValueError('Focused intervention candidates need their own observed task outcomes.')
    if factual_task:
        network,record,fit=fit_paired_factual_task_potential(examples,
            program_options=data.get('program_options'),global_width=global_width,
            width=62 if task_width is None else task_width,updates=updates,seed=seed,device=device,
            response_supervision=factual_response,factual_features=factual_features,
            intervention_contrast=factual_contrast,future_rank=factual_rank,
            future_normalization=factual_normalization)
    elif paired_future_rank and not ordinary:
        trainer=fit_joint_future_task_potential if paired_joint_likelihood else fit_paired_future_task_potential
        network,record,fit=trainer(examples,program_options=data.get('program_options'),
            global_width=5 if absolute_context else 3,response_rank=paired_future_rank,
            width=(61 if paired_joint_likelihood else 64) if task_width is None else task_width,
            updates=updates,seed=seed,device=device,difference_weight=difference_weight)
    elif paired_ordering and not ordinary:
        network,record,fit=fit_future_conditioned_task_potential(examples,bandwidth=bandwidth,
            program_options=data.get('program_options'),contrast_weight=difference_weight,
            future_precision=future_precision,anchor_weight=.01,conditional_ordering=True,
            residual_calibration=residual_calibration)
    elif paired_graph and not ordinary:
        estimator=((lambda examples:successor_standardized_task_estimates(examples,task_conditioned=True))
            if paired_graph=='factual' else successor_standardized_task_estimates if paired_graph=='successor' else
            matched_future_task_estimates if paired_graph=='blocks' else paired_block_task_estimates)
        corrected,statistics=(matched_duration_task_estimates(examples,physical_contexts,example_spans,
            shared_factual=not data.get('policy_iteration',False))
            if duration_conditioned and paired_graph=='blocks' else estimator(examples))
        network,record,fit=fit_shared_program_potential(corrected,width=64,updates=updates,
            paired=False,conditional=False,seed=seed,device=device,
            program_options=data.get('program_options'),global_width=global_width)
        fit.update(statistics,task_target=('paired successor-standardized terminal task success'
            if paired_graph in ('successor','factual') else 'paired covariance-corrected terminal task success'),
            terminal_difference_labels=0,
            paired_intervention_records=sum(len(example['pairs']) for example in examples),
            deployed_parameters=sum(parameter.numel() for parameter in network.parameters()))
    elif task_contrasts and not ordinary and not shared_task_value:
        network,record,fit=fit_intervention_task_potential(examples,conditional=not ordinary,
            difference_weight=difference_weight,bandwidth=bandwidth,program_options=data.get('program_options'))
    elif ordinary or shared_task_value:
        ordinary_examples=[]
        for example in examples:
            samples={};edges={}
            for code,_,outcome in example['samples'].values():samples.setdefault(code,[]).append(outcome)
            for a,b in example['pairs']:
                ya=example['samples'][a][2];yb=example['samples'][b][2]
                edges.setdefault((a[0],b[0]),[]).append(yb-ya)
            programs=sorted(samples);index={q:i for i,q in enumerate(programs)}
            ordinary_examples.append(dict(context=example['context'],programs=programs,
                targets=np.asarray([np.mean(samples[q]) for q in programs]),
                edges=[(index[a],index[b],float(np.mean(values))) for (a,b),values in edges.items()]))
        trainer=fit_shared_program_potential if ordinary else fit_conditional_shared_program_potential
        conditional_settings={} if ordinary else dict(bounded_success=True)
        network,record,fit=trainer(ordinary_examples,width=64,updates=updates,
            paired=False,conditional=not ordinary,difference_weight=difference_weight,
            seed=seed,device=device,program_options=data.get('program_options'),
            global_width=global_width,**conditional_settings)
        fit.update(task_target='true_terminal_task_success',
            terminal_difference_labels=sum(len(example['pairs']) for example in examples) if not ordinary else 0,
            deployed_parameters=sum(parameter.numel() for parameter in network.parameters()))
    elif transport or latent_transport:
        network,record,fit=fit_successor_transport_potential(data,updates=updates,seed=seed,device=device,
            latent=latent_transport,width=59 if task_width is None else task_width,response_width=response_width,
            terminal_difference_weight=difference_weight,paired_standardization=paired_calibration,
            covariance_shrink=covariance_shrink)
    else:
        network,record,fit=fit_successor_task_potential(data,updates=updates,seed=seed,
            device=device,decision_weight=decision_weight,terminal_difference_weight=difference_weight,
            absolute_context=absolute_context,paired_calibration=paired_calibration,covariance_shrink=covariance_shrink)
    if record.get('kind') in ('local_intervention_task','paired_future_task') and not data.get('initial_only',True):
        record['time_bandwidth']=1e-4
    contexts=torch.tensor(np.asarray(physical_contexts),dtype=torch.float32)
    candidate_actions=sorted({(q,span) for example,span in zip(examples,example_spans)
        for q,_,_ in example['samples'].values()})
    candidate_codes=[q for q,span in candidate_actions]
    candidate_support=np.asarray([[window==span and q in {sample[0] for sample in example['samples'].values()}
        for q,window in candidate_actions] for example,span in zip(examples,example_spans)])
    candidate_variance=None
    if paired_graph and not ordinary:
        candidate_variance=np.asarray([[dict(zip(example['programs'],example['gain_variance'])).get(q,0.)
            if window==span else 0. for q,window in candidate_actions]
            for example,span in zip(corrected,example_spans)])
    model=predecessor
    removal=tuple(model.get('mechanism_ablation') or (False,False,False))+(False,)*3
    # A new value learner is active even when its F00/F10 predecessor had
    # disabled value learning. Preserve the independently removed M1 paths.
    model['mechanism_ablation']=(removal[0],False,removal[2])
    model['successor_task_value']=dict(program_model=record,
        contexts=SuccessorTaskDecision.summary(contexts).numpy(),span=data['span'],
        bandwidth=bandwidth,margin=margin,initial_only=data.get('initial_only',True),
        time_bandwidth=None if data.get('initial_only',True) else 1e-4,
        candidate_codes=candidate_codes,candidate_contexts=contexts.numpy(),candidate_support=candidate_support,
        absolute_context=absolute_context,support_weighted=support_weighted,
        candidate_variance=candidate_variance,confidence_scale=confidence_scale if not ordinary else 0.,
        explicit_wait=data.get('policy_iteration',False),
        prior=previous if retain_predecessor else None,
        include_all_vessels=data.get('include_all_vessels',False),
        learning_objective='absolute' if ordinary else 'conditional')
    if duration_conditioned:
        model['successor_task_value'].update(duration_conditioned=True,
            candidate_spans=[span for q,span in candidate_actions],span=min(example_spans))
    model['successor_task_fit']=dict(ordinary=ordinary,**fit,
        source='identical paired interventions and true terminal continuations',
        inherited_value_stages=(data['model'].get('successor_task_fit',{}).get('inherited_value_stages',0)+1
            if retain_predecessor else 0),complete_mechanism_replacement=not retain_predecessor,
        policy_iteration=(data['model'].get('successor_task_fit',{}).get('policy_iteration',0)+1
            if data.get('policy_iteration',False) else 0),
        continuation_objective=None if previous is None else previous['learning_objective'])
    if duration_conditioned:
        model['successor_task_fit'].update(duration_conditioned=True,
            candidate_durations=sorted(set(example_spans)))
    model['whole_mechanism_replacement']=ordinary
    model['comparison_scope']=('whole conditional-value replacement by absolute-return learning'
        if ordinary else 'whole conditional intervention task-value learning'
        if paired_graph or shared_task_value or task_contrasts else 'whole conditional successor-task value learning')
    model.pop('selection',None)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);torch.save(model,output)
    return model


def align_task_stage_support(stage,examples):
    """Ground every deployed candidate in this stage's actual observations.

    A retained controller can have a wider candidate library than a later
    corrected dataset. Keeping its old support would let an estimator choose
    programs for which the replacement learner has no measured consequence.
    Match the physical context exactly; sharing network parameters does not
    grant new empirical support to a program at another context.
    """
    import numpy as np
    contexts=np.asarray([example['context'] for example in examples])
    reference=np.asarray(stage['candidate_contexts'])
    codes=np.asarray(stage['candidate_codes'])
    support=np.zeros((len(reference),len(codes)),dtype=bool)
    for row,context in enumerate(reference):
        matches=np.flatnonzero(np.max(np.abs(contexts-context),axis=-1)<1e-5)
        if not len(matches):
            raise ValueError('A deployed support context has no matching training observations.')
        observed={sample[0] for i in matches for sample in examples[i]['samples'].values()}
        if 0 not in observed:
            raise ValueError('Every deployed value context needs its own factual outcome.')
        support[row]=np.isin(codes,list(observed))
    previous=np.asarray(stage.get('candidate_support',support),dtype=bool)
    stage['candidate_support']=support
    return dict(observed_candidates=int(support.sum()),
        removed_unobserved_candidates=int((previous&~support).sum()))


def ordinary_task_return_examples(datasets):
    """Read unique absolute observations without paired contrasts or supports."""
    import numpy as np
    groups={};actions=set();options=None;defenders=None;counts=[]
    for data in datasets:
        if not data.get('completed',False):
            raise ValueError('Ordinary return learning needs completed task observations.')
        current=np.asarray(data.get('program_options'))
        if current.shape!=(10,2):
            raise ValueError('Use the common ten-choice actuator program encoding.')
        if options is None:options=current
        elif not np.array_equal(options,current):
            raise ValueError('Every dataset must use the same physical program encoding.')
        span=int(data['span']);dataset_samples=set()
        for rollout in data['rollouts']:
            for branch in rollout['branches']:
                packet={k:torch.as_tensor(v[None]) for k,v in branch['packet'].items()}
                raw,order=intervention_task_context(packet,True,True)
                n=order.shape[-1]
                if defenders is None:defenders=n
                elif n!=defenders:raise ValueError('Fit this fixed-fleet comparison at one fleet size.')
                context=np.r_[raw[0].numpy(),span/20.].astype(np.float32)
                key=np.round(context,6).tobytes()
                group=groups.setdefault(key,dict(context=context,samples={}))
                observations=[('left','before'),('right','after')]
                if branch.get('factual_outcome') is not None and branch.get('reference_action') is not None:
                    observations.append(('reference_action','factual_outcome'))
                for side,label in observations:
                    action=np.asarray(branch[side])
                    if action.shape!=(n,13) or not np.all(action[:,4]==span):
                        raise ValueError('Return labels must match the complete program and its window.')
                    preferences=action[order[0].numpy(),9:11]
                    if not np.array_equal(preferences,action[order[0].numpy(),11:13]):
                        raise ValueError('A phase-changing program needs its own action encoding.')
                    matches=np.all(preferences[:,None]==options[None],axis=-1)
                    if not np.all(matches.sum(-1)==1):
                        raise ValueError('An observed action is outside the common candidate encoding.')
                    code=int((matches.argmax(-1)*10**np.arange(n)).sum())
                    outcome=float(branch[label]['success']);sample=(code,branch['continuation_seed'])
                    if sample in group['samples'] and group['samples'][sample]!=outcome:
                        raise ValueError('Conflicting absolute outcomes for the same context/program/future.')
                    group['samples'][sample]=outcome
                    dataset_samples.add((key,sample));actions.add((code,span))
        counts.append(dict(window=span,unique_outcomes=len(dataset_samples)))
    examples=[]
    for group in groups.values():
        outcomes={}
        for (code,_),outcome in group['samples'].items():outcomes.setdefault(code,[]).append(outcome)
        programs=sorted(outcomes)
        examples.append(dict(context=group['context'],programs=programs,
            targets=np.asarray([np.mean(outcomes[q]) for q in programs])))
    if not examples:raise ValueError('No absolute task observations were supplied.')
    return examples,sorted(actions),options,defenders,dict(datasets=counts,
        roots=len(examples),unique_outcomes=sum(len(g['samples']) for g in groups.values()))


def fit_ordinary_task_return_policy(base,datasets,output,*,width=91,updates=10000,seed=42,device='cpu'):
    """Retrain and deploy ordinary return learning without the M2 executor."""
    import copy
    from types import SimpleNamespace
    import develop_source_fusion as development
    model=copy.deepcopy(base)
    model.pop('successor_task_value',None)
    if development.joint_value_objectives(model):
        raise ValueError('Start ordinary return learning from the common source/M1 controller.')
    development.require_clean_mechanism_start(model,SimpleNamespace(
        joint_no_cf=True,without_peer=False,joint_no_interaction=False))
    examples,actions,options,defenders,data_fit=ordinary_task_return_examples(datasets)
    network,record,fit=fit_shared_program_potential(examples,width=width,updates=updates,
        paired=False,conditional=False,paired_likelihood=True,seed=seed,device=device,
        program_options=options,global_width=6)
    model['successor_task_value']=dict(kind='ordinary_joint_return',program_model=record,
        candidate_codes=[q for q,h in actions],candidate_spans=[h for q,h in actions],
        defenders=defenders,learning_objective='absolute')
    model['successor_task_fit']=dict(ordinary=True,**fit,**data_fit,
        deployed_parameters=sum(p.numel() for p in network.parameters()),
        candidate_program_windows=len(actions),inherited_stage_schedule=False,
        complete_mechanism_replacement=True,terminal_difference_labels=0,
        execution='one greedy absolute-return decision; global program/window vocabulary',
        continuation_objectives=[sorted(development.joint_value_objectives(data['model'])) for data in datasets],
        source='same fixed task observations; no predecessor controller is deployed')
    model['fixed_intervention_clock']=False
    model['whole_mechanism_replacement']=True
    model['comparison_scope']='fixed-data ordinary return replacement with independent execution'
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);torch.save(model,output)
    return model


def fit_shared_stage_task_policy(template,stage_data,output,*,ordinary=False,
                                 width=64,updates=5000,difference_weight=4.,
                                 seed=42,device='cpu',successor_model=False,informative_pair_sampling=False,
                                 peer_conditioned=True):
    """Learn one window-conditioned value surface across intervention stages.

    Conditional stage support and execution windows are kept as sampled. The
    value function shares parameters across stages instead of replacing
    learned values with per-context outcome tables. Both objectives receive
    the same unique terminal outcomes and complete-program representation;
    only conditional learning uses the direction of discordant paired
    outcomes. Ordinary learning uses an independent greedy controller and
    the global action vocabulary, without inheriting these stages. Window
    length is an input, never an unobserved future label.
    """
    import copy
    import numpy as np
    import develop_source_fusion as development
    if ordinary:
        if successor_model:
            raise ValueError('The paired successor objective is part of conditional learning.')
        return fit_ordinary_task_return_policy(template,stage_data,output,width=width,
            updates=updates,seed=seed,device=device)
    model=copy.deepcopy(template)
    stages=[];stage=model.get('successor_task_value')
    while stage is not None:
        stages.append(stage);stage=stage.get('prior')
    if len(stages)!=len(stage_data):
        raise ValueError('Provide the observed task data for every retained value stage.')
    if successor_model and ordinary:
        raise ValueError('The paired successor objective is part of conditional learning.')
    common=[];physical_examples=[];stage_counts=[];options=None
    for stage,data in zip(stages,stage_data):
        if not data.get('completed',False):
            raise ValueError('Every value stage requires completed paired continuations.')
        if int(stage['span'])!=int(data['span']):
            raise ValueError('The fitted window must match the measured intervention window.')
        if options is None:options=data.get('program_options')
        elif not np.array_equal(np.asarray(options),np.asarray(data.get('program_options'))):
            raise ValueError('Shared value stages require the same program encoding.')
        examples=successor_task_examples(data,absolute_frame=True)
        support_fit=align_task_stage_support(stage,examples)
        for example in examples:
            physical=copy.deepcopy(example)
            physical['context']=np.r_[example['context'],float(data['span'])/20.].astype(np.float32)
            physical['samples']={key:(code,np.r_[following,float(data['span'])/20.].astype(np.float32),outcome)
                for key,(code,following,outcome) in example['samples'].items()}
            physical_examples.append(physical)
            outcomes={};pairs={}
            for code,_,outcome in example['samples'].values():
                outcomes.setdefault(code,[]).append(outcome)
            for a,b in dict.fromkeys(example['pairs']) if not ordinary else ():
                ya=example['samples'][a][2];yb=example['samples'][b][2]
                pairs.setdefault((a[0],b[0]),[]).append(yb-ya)
            programs=sorted(outcomes);lookup={q:i for i,q in enumerate(programs)}
            edges=[(lookup[a],lookup[b],float(np.mean(values)),1.,
                    int(np.count_nonzero(np.asarray(values)>0.)),
                    int(np.count_nonzero(np.asarray(values)<0.)))
                   for (a,b),values in pairs.items()]
            common.append(dict(context=np.r_[example['context'],float(data['span'])/20.].astype(np.float32),
                programs=programs,targets=np.asarray([np.mean(outcomes[q]) for q in programs]),edges=edges))
        stage_counts.append(dict(roots=len(examples),window=int(data['span']),
            unique_outcomes=sum(len(e['samples']) for e in examples),
            paired_outcomes=sum(len(set(e['pairs'])) for e in examples),**support_fit))
    if successor_model:
        network,record,fit=fit_successor_task_potential(dict(program_options=options),
            width=width-5,response_width=24,updates=updates,seed=seed,device=device,
            decision_weight=2.,terminal_difference_weight=difference_weight,
            absolute_context=True,examples=physical_examples,global_width=6,conditional_ordering=True)
    else:
        trainer=fit_shared_program_potential if ordinary else fit_conditional_shared_program_potential
        settings={} if ordinary else dict(informative_pair_sampling=informative_pair_sampling,peer_conditioned=peer_conditioned)
        network,record,fit=trainer(common,width=width,updates=updates,paired=False,
            conditional=not ordinary,paired_likelihood=True,difference_weight=difference_weight,
            seed=seed,device=device,program_options=options,global_width=6,**settings)
    for stage in stages:
        stage.update(program_model=record,learning_objective='absolute' if ordinary else 'conditional',
            absolute_context=True,duration_conditioned=True,
            candidate_spans=[stage['span']]*len(stage['candidate_codes']),
            candidate_variance=None,confidence_scale=0.)
    fit.update(shared_stage_value=True,shared_successor_model=successor_model,
        stages=stage_counts,complete_mechanism_replacement=False,
        terminal_difference_labels=0 if ordinary else sum(s['paired_outcomes'] for s in stage_counts),
        deployed_parameters=sum(p.numel() for p in network.parameters()),
        source='identical stage observations, candidates, windows, support and optimizer budget',
        task_target='absolute terminal success' if ordinary else 'terminal success with conditional paired ordering')
    # This experiment freezes the sampled stage schedule and continuation
    # policies. Refitting every value does not remove that conditional-policy
    # scaffold, so it is an estimator comparison, not a whole-M2 replacement.
    fit['inherited_stage_schedule']=True
    fit['continuation_objectives']=[sorted(development.joint_value_objectives(data['model']))
        for data in stage_data]
    model['successor_task_fit']=dict(ordinary=ordinary,**fit)
    model['whole_mechanism_replacement']=False
    model['comparison_scope']='fixed-data estimator comparison with shared stage execution; not whole-mechanism replacement'
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True);torch.save(model,output)
    return model


@lru_cache(maxsize=1)
def _copula_quadrature():
    import numpy as np
    points,weights=np.polynomial.legendre.leggauss(24)
    return torch.as_tensor(points),torch.as_tensor(weights)


def paired_bernoulli_probabilities(probability,loading):
    """A paired outcome law whose marginals remain the task-success values."""
    probability=probability.clamp(1e-5,1.-1e-5)
    threshold=2.**.5*torch.erfinv(2.*probability-1.)
    correlation=(.95*loading.tanh()).prod(-1)
    points,weights=(value.to(probability) for value in _copula_quadrature())
    r=.5*correlation[:,None]*(1.+points)
    a,b=threshold[:,0,None],threshold[:,1,None]
    variance=1.-r.square()
    density=torch.exp(-(a.square()-2.*r*a*b+b.square())/(2.*variance))/(2.*torch.pi*variance.sqrt())
    joint=probability.prod(-1)+.5*correlation*(density*weights).sum(-1)
    lower=(probability.sum(-1)-1.).clamp_min(0.);upper=probability.amin(-1)
    joint=torch.minimum(torch.maximum(joint,lower),upper)
    left,right=probability.unbind(-1)
    return torch.stack((1.-left-right+joint,right-joint,left-joint,joint),-1).clamp_min(0.)


def fit_response_precision_potential(examples,width=48,updates=1500,seed=42,node_width=8,
                                     global_width=3,program_options=None,device='cpu'):
    """Estimate values once, resolving the correlated noise of paired futures.

    The intervention graph supplies joint observations, not extra difference
    labels. A root's outcome matrix estimates the covariance of its program
    means. Shrinkage keeps this positive definite with only a few futures;
    whitening then discounts shared outcome noise while retaining contrasts.
    Root sampling is proportional to the number of programs, so additional
    continuations improve covariance estimation without multiplying a
    program's influence. Neither future keys nor covariance are deployed.
    """
    import numpy as np
    contexts=[];codes=[];targets=[];blocks=[];precision=[];future_count=0
    for example in examples:
        count=len(example['programs']);start=len(codes)
        futures=list(dict.fromkeys(future for _,future,_ in example['outcomes']))
        lookup={future:i for i,future in enumerate(futures)}
        outcomes=np.full((count,len(futures)),np.nan)
        for program,future,outcome in example['outcomes']:
            previous=outcomes[program,lookup[future]]
            if np.isfinite(previous) and previous!=outcome:
                raise ValueError('A paired continuation has conflicting outcomes.')
            outcomes[program,lookup[future]]=outcome
        observed=np.isfinite(outcomes);samples=observed.sum(1)
        if (samples==0).any():raise ValueError('Every program needs an observed continuation.')
        mean=np.nanmean(outcomes,axis=1)
        residual=np.where(observed,outcomes-mean[:,None],0.)
        residual/=np.sqrt(np.maximum(samples*(samples-1),1.))[:,None]
        # A Bernoulli variance floor and two prior continuations stabilize
        # zero/one empirical outcomes and noisy two-future correlations.
        diagonal=np.maximum(np.square(residual).sum(1),.05/samples)
        standardized=residual/np.sqrt(diagonal[:,None])
        shrink=np.sqrt(samples/(samples+2.))
        factor=shrink[:,None]*standardized
        correlation=factor@factor.T
        correlation[np.diag_indices(count)]+=1.-np.square(factor).sum(1)
        inverse=np.linalg.solve(correlation,np.eye(count))
        inverse/=np.trace(inverse)/count
        contexts.extend([example['context']]*count);codes.extend(example['programs']);targets.extend(mean)
        blocks.append(np.arange(start,start+count));precision.append(torch.tensor(inverse,dtype=torch.float32,device=device))
        future_count+=len(futures)
    context=torch.as_tensor(np.asarray(contexts),dtype=torch.float32,device=device)
    code=torch.as_tensor(codes,dtype=torch.long,device=device)
    target=torch.as_tensor(targets,dtype=torch.float32,device=device)
    torch.manual_seed(seed);rng=np.random.default_rng(seed+104729)
    net=PairedResponsePotential(width,False,node_width=node_width,global_width=global_width,
        program_options=program_options,probabilistic=False).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=.005)
    probability=np.asarray([len(block) for block in blocks],dtype=float);probability/=probability.sum()
    last={}
    with torch.enable_grad():
        for step in range(updates):
            selected=rng.choice(len(blocks),4,p=probability)
            indices=torch.as_tensor(np.concatenate([blocks[b] for b in selected]),device=device)
            error=net(context[indices],code[indices])-target[indices]
            losses=[];offset=0
            for b in selected:
                length=len(blocks[b]);part=error[offset:offset+length];offset+=length
                losses.append((part@(precision[b]@part))/length)
            loss=torch.stack(losses).mean()
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            if (step+1)%250==0 or step+1==updates:
                last=dict(phase='paired_response_precision_fit',update=step+1,
                    joint_error=float(loss.detach()),node_mse=float(error.square().mean().detach()))
                print(last,flush=True)
    state=dict(width=width,paired=False,node_width=node_width,global_width=global_width,
        program_options=program_options,probabilistic=False,response_rank=0,outcome_scale=0.,
        state={key:value.detach().cpu() for key,value in net.state_dict().items()})
    return PairedResponsePotential(**state).eval(),state,dict(nodes=len(codes),contexts=len(blocks),
        paired_future_blocks=future_count,updates=updates,seed=seed,duplicate_contrast_labels=0,
        deployed_parameters=sum(p.numel() for p in net.parameters()),
        likelihood_factorization='joint_program_mean_covariance',last=last)


def fit_block_program_potential(examples,width=48,updates=1500,seed=42,node_width=8,
                                global_width=3,program_options=None,device='cpu',ordering_weight=1.):
    """Fit paired response distributions without counting an outcome twice.

    A Gaussian copula with plan-conditioned response loadings retains both
    positive and negative pairing dependence without changing the predicted
    marginal success probability. Each root/future
    uses a disjoint matching of its single-vessel intervention graph, plus
    marginal likelihoods for unmatched programs. This composite score keeps
    marginal task calibration while learning paired concordance/discordance;
    it never adds a loss on differences of the same outcome means. The latent
    disturbance is integrated out; future IDs are absent at deployment.
    """
    import numpy as np
    contexts=[];codes=[];targets=[];blocks=[];block_edges=[]
    for example in examples:
        groups={}
        for program,future,outcome in example['outcomes']:
            if outcome not in (0.,1.):
                raise ValueError('A unique paired continuation must have one binary terminal outcome.')
            groups.setdefault(future,{})[program]=outcome
        for group in groups.values():
            start=len(codes);lookup={}
            for program,outcome in group.items():
                lookup[program]=len(codes)
                contexts.append(example['context']);codes.append(example['programs'][program]);targets.append(outcome)
            blocks.append(np.arange(start,len(codes)))
            block_edges.append(np.asarray([(lookup[a],lookup[b]) for a,b,*_ in example['edges']
                if a != b and a in lookup and b in lookup],dtype=np.int64).reshape(-1,2))
    context=torch.as_tensor(np.asarray(contexts),dtype=torch.float32,device=device)
    code=torch.as_tensor(codes,dtype=torch.long,device=device)
    target=torch.as_tensor(targets,dtype=torch.float32,device=device)
    torch.manual_seed(seed);rng=np.random.default_rng(seed+104729)
    # The response head uses fewer total deployed parameters than the width-
    # matched ordinary potential; its extra signal comes from paired outcomes.
    width=max(8,width-1)
    net=PairedResponsePotential(width,False,node_width=node_width,global_width=global_width,
        program_options=program_options,probabilistic=True,response_rank=1).to(device)
    with torch.no_grad():net.readout[-1].bias[1:]=.25
    optimizer=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=.005)
    probability=np.asarray([len(block) for block in blocks],dtype=float);probability/=probability.sum()
    last={}
    with torch.enable_grad():
        for step in range(updates):
            selected=rng.choice(len(blocks),8,p=probability)
            factors=[];factor_blocks=[]
            for local,b in enumerate(selected):
                used=set();edges=block_edges[b]
                for a,c in edges[rng.permutation(len(edges))]:
                    if a in used or c in used:continue
                    used.update((a,c));factors.append((a,c));factor_blocks.append(local)
                for a in blocks[b]:
                    if a not in used:factors.append((a,));factor_blocks.append(local)
            indices=torch.as_tensor(np.concatenate(factors),device=device)
            lengths=np.asarray([len(factor) for factor in factors]);starts=np.r_[0,lengths.cumsum()[:-1]]
            parameters=net.response_parameters(context[indices],code[indices])
            logit=parameters[:,0];loading=parameters[:,1]
            paired=np.flatnonzero(lengths==2);single=np.flatnonzero(lengths==1)
            factor_loss=logit.new_zeros(len(factors))
            if len(paired):
                pair=torch.as_tensor(starts[paired,None]+np.arange(2),device=device)
                probabilities=paired_bernoulli_probabilities(logit[pair].sigmoid(),loading[pair])
                outcome=(2.*target[indices[pair[:,0]]]+target[indices[pair[:,1]]]).long()
                observed=probabilities.gather(1,outcome[:,None]).squeeze(1).clamp_min(1e-8)
                discordant=(outcome==1)|(outcome==2)
                discordance=probabilities[:,1:3].sum(-1).clamp_min(1e-8)
                # Factor the two observed bits into the success count and,
                # only for a discordant pair, its winning direction. The
                # conditional ordering score is not an additional endpoint
                # or difference label. At unit weight this is exactly the
                # original paired likelihood.
                cardinality=torch.where(discordant,discordance,observed)
                direction=torch.where(discordant,observed/discordance,torch.ones_like(observed))
                factor_loss[torch.as_tensor(paired,device=device)]=-cardinality.log()-ordering_weight*direction.clamp_min(1e-8).log()
            if len(single):
                position=torch.as_tensor(starts[single],device=device)
                factor_loss[torch.as_tensor(single,device=device)]=F.binary_cross_entropy_with_logits(
                    logit[position],target[indices[position]],reduction='none')
            block_loss=factor_loss.new_zeros(len(selected)).index_add(0,
                torch.as_tensor(factor_blocks,device=device),factor_loss)
            loss=(block_loss/torch.as_tensor([len(blocks[b]) for b in selected],device=device)).mean()
            optimizer.zero_grad();loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),5.);optimizer.step()
            if (step+1)%250==0 or step+1==updates:
                last=dict(phase='paired_block_likelihood_fit',update=step+1,
                    joint_nll=float(loss.detach()),mean_response_loading=float((.95*loading.tanh()).abs().mean().detach()))
                print(last,flush=True)
    state=dict(width=width,paired=False,node_width=node_width,global_width=global_width,
        program_options=program_options,probabilistic=True,response_rank=1,
        outcome_scale=0.,
        state={key:value.detach().cpu() for key,value in net.state_dict().items()})
    network=PairedResponsePotential(**state).eval()
    return network,state,dict(unique_outcomes=len(codes),paired_future_blocks=len(blocks),updates=updates,
        seed=seed,duplicate_contrast_labels=0,maximum_block_size=max(map(len,blocks)),
        deployed_parameters=sum(p.numel() for p in network.parameters()),response_rank=1,
        likelihood_factorization='disjoint_single_vessel_pairs',response_copula='gaussian',
        ordering_weight=ordering_weight,last=last)


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
        for edge in example.get('edges',()) if conditional else ():
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
            if conditional:
                nodes=torch.randint(len(code),(128,),generator=generator).to(device)
                edges=torch.randint(len(left),(128,),generator=generator).to(device)
                selected=torch.cat((nodes,left[edges],right[edges]))
                batch_weight=node_weight[selected]
            else:
                # Ordinary return regression consumes only (context, program,
                # return) observations. Its sampling must not use the local
                # intervention graph, even when edge labels are available.
                selected=torch.randint(len(code),(384,),generator=generator).to(device)
                batch_weight=torch.ones_like(target[selected])
            prediction=net.logits(context[selected],code[selected])
            if paired_likelihood:
                node_loss=(F.binary_cross_entropy_with_logits(prediction,target[selected],reduction='none')
                    *batch_weight).mean()
                log_odds=prediction[256:]-prediction[128:256] if conditional else None
                # Conditioning on discordance removes the common outcome of a
                # paired continuation. Concordant pairs supply node evidence.
                edge_loss=((wins[edges]*F.softplus(-log_odds)+losses[edges]*F.softplus(log_odds)).mean()
                    if conditional else prediction.new_zeros(()))
            else:
                node_loss=((prediction-target[selected]).square()*batch_weight).mean()
                if conditional:
                    contrast=(prediction[256:]-prediction[128:256]-delta[edges]).square()
                    edge_loss=(contrast*precision[edges]).mean()
                else:edge_loss=prediction.new_zeros(())
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
    return net.cpu().eval(),record,dict(nodes=len(code),edges=len(left),updates=updates,seed=seed,last=last,
        sampler='uniform nodes and intervention endpoints' if conditional else 'uniform return observations',
        paired_input_used=conditional)


def fit_conditional_shared_program_potential(examples,width=48,updates=1500,paired=True,conditional=True,
                                 difference_weight=4.,seed=42,node_width=8,global_width=3,
                                 program_options=None,device='cpu',paired_likelihood=False,
                                 bounded_success=False,informative_pair_sampling=False,peer_conditioned=True):
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
    edge_pool=torch.arange(len(left),device=device)
    edge_scale=1.
    if informative_pair_sampling and paired_likelihood and conditional:
        informative=torch.nonzero(wins+losses>0,as_tuple=False).flatten()
        if len(informative):
            edge_pool=informative
            # Concordant pairs contribute exactly zero to this contrast
            # objective. Concentrate its samples on nonzero contributions,
            # retaining the original loss scale and all absolute node data.
            edge_scale=len(informative)/len(left)
    degree=torch.bincount(torch.cat((left[edge_pool],right[edge_pool])),minlength=len(code)).to(torch.float32)
    node_weight=3./(1.+len(code)*degree/max(1,len(edge_pool)))
    torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed+104729)
    net=SharedProgramPotential(width,paired,node_width=node_width,global_width=global_width,
        program_options=program_options,probabilistic=paired_likelihood or bounded_success,
        peer_conditioned=peer_conditioned).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=.005)
    last={}
    with torch.enable_grad():
        for step in range(updates):
            nodes=torch.randint(len(code),(128,),generator=generator).to(device)
            edges=edge_pool[torch.randint(len(edge_pool),(128,),generator=generator).to(device)]
            selected=torch.cat((nodes,left[edges],right[edges]))
            prediction=net.logits(context[selected],code[selected])
            if bounded_success and not paired_likelihood:prediction=prediction.sigmoid()
            if paired_likelihood:
                node_loss=(F.binary_cross_entropy_with_logits(prediction,target[selected],reduction='none')
                    *node_weight[selected]).mean()
                log_odds=prediction[256:]-prediction[128:256]
                # Conditioning on discordance removes the common outcome of a
                # paired continuation. Concordant pairs supply node evidence.
                edge_loss=edge_scale*(wins[edges]*F.softplus(-log_odds)+losses[edges]*F.softplus(log_odds)).mean()
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
        program_options=program_options,probabilistic=paired_likelihood or bounded_success,peer_conditioned=peer_conditioned,
        state={k:v.detach().cpu() for k,v in net.state_dict().items()})
    return net.cpu().eval(),record,dict(nodes=len(code),edges=len(left),updates=updates,seed=seed,last=last,
        informative_pair_sampling=informative_pair_sampling,sampled_edges=len(edge_pool),edge_loss_scale=edge_scale)


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


def continuous_program_proposals(reference,elites,count,rng,generation=0,spans=(3,5,8,12,20),asynchronous=False):
    """Refine complete fusion sequences jointly with their switching times.

    Keep the predecessor and the best measured elites, search around those
    elites, and retain a quarter of global proposals. Each candidate blends the unchanged
    neural and Boids controls; the proposal search runs only in development.
    """
    import numpy as np
    n=len(reference)
    original=(reference.copy() if reference.shape[-1]==9 else
        np.concatenate((reference[:,:6],np.zeros((n,2),np.float32),
            np.full((n,1),int(reference[0,4])//2,np.float32)),-1))
    retained=min(len(elites),max(1,count//8),max(0,count-1))
    programs=[original]+[action.copy() for action in elites[:retained]]
    local_count=3*count//4-len(programs)
    for j in range(count-len(programs)):
        action=elites[j%len(elites)].copy() if elites and j<local_count else original.copy()
        if j>=local_count or (not elites and j>=count//2):
            first=rng.choice([-1.,0.,.25,.5,.75,1.],n)
            second=rng.choice([-1.,0.,.25,.5,.75,1.],n)
        else:
            first=action[:,3].copy();second=action[:,5].copy()
            for values in (first,second):
                change=rng.random(n)<.6;held=(values>=0.)&change
                values[held]=np.clip(values[held]+rng.normal(0.,.35/(1.+.5*generation),held.sum()),0.,1.)
                release=(rng.random(n)<.12)&change
                values[release]=rng.choice([-1.,0.,1.],release.sum())
        span=(int(rng.choice(spans)) if not elites or j>=local_count or rng.random()<.25 else int(action[0,4]))
        boundary=(rng.integers(1,span,size=n) if asynchronous else int(rng.integers(1,span))) if (
            not elites or j>=local_count or rng.random()<.3) else np.minimum(span-1,action[:,8])
        action[:,3]=first;action[:,5]=second;action[:,4]=span;action[:,8]=boundary
        programs.append(action)
    return programs


def response_program_proposals(reference,coefficients,spans=(3,5,8,12,20)):
    """Compress observed joint fusion responses into timed program proposals.

    Each vessel keeps the least-squares two-phase approximation of its own
    response. These are proposals only: paired environment outcomes, including
    losses on other futures, determine their learned intervention authority.
    """
    import numpy as np
    programs=[];coefficients=np.asarray(coefficients)
    for span in spans:
        if span>len(coefficients) or span<2:continue
        action=np.concatenate((reference[:,:6],np.zeros((len(reference),3),np.float32)),-1)
        action[:,4]=span
        for vessel in range(len(reference)):
            sequence=coefficients[:span,vessel];best=None
            for boundary in range(1,span):
                first=sequence[:boundary].mean();second=sequence[boundary:].mean()
                error=np.square(sequence[:boundary]-first).sum()+np.square(sequence[boundary:]-second).sum()
                if best is None or error<best[0]:best=(error,boundary,first,second)
            action[vessel,3]=best[2];action[vessel,5]=best[3];action[vessel,8]=best[1]
        programs.append(action)
    return programs


def continuous_program_neighborhood(reference,parent,step=.2):
    """Single-vessel edges around a continuous, timed fusion program.

    Both endpoint programs retain the same switch and end time. Reverting a
    vessel restores its predecessor coefficients, not its predecessor clock.
    The factual continuation is consequently evaluated separately.
    """
    import numpy as np
    arms=[parent.copy()];edges=[];known={parent[:,3:].tobytes():0}
    for subject in range(len(parent)):
        alternatives=[]
        deletion=parent.copy();deletion[subject,[3,5]]=reference[subject,[3,5]]
        alternatives.append(deletion)
        for column in (3,5):
            if parent[subject,column]<0.:continue
            for change in (-step,step):
                action=parent.copy();action[subject,column]=np.clip(action[subject,column]+change,0.,1.)
                alternatives.append(action)
        release=parent.copy();release[subject,[3,5]]=-1.;alternatives.append(release)
        for action in alternatives:
            key=action[:,3:].tobytes()
            if key not in known:known[key]=len(arms);arms.append(action)
            if known[key]:edges.append((0,known[key]))
    return arms,list(dict.fromkeys(edges))


def fit_continuous_program_values(groups,difference_weight=4.,ridge=.25,bandwidth=.03,margin=.05):
    """Learn sparse complete-program potentials from paired physical outcomes.

    Programs retain real-valued mixtures and their learned phase boundary.
    The representation scales with observed programs, rather than an
    exponential Cartesian product of a quantized per-vessel alphabet.
    """
    import numpy as np
    contexts=[];programs=[];indices=[];values=[];variances=[]
    for group in groups:
        outcomes=np.asarray(group['outcomes'],dtype=float)
        factual=np.asarray(group['factual'],dtype=float)
        candidates=group['programs'];count=len(candidates)
        gain=outcomes-factual[:,None]
        normal=(1.+ridge)*np.eye(count);target=gain.mean(0)
        for left,right in group['edges']:
            a,b=candidates[left],candidates[right]
            changed=np.any(np.abs(a[:,[3,5]]-b[:,[3,5]])>1e-6,axis=-1)
            if changed.sum()!=1 or not np.array_equal(a[:,[4,8]],b[:,[4,8]]):
                raise ValueError('Conditional edges must change one vessel and preserve both decision times.')
            contrast=outcomes[:,right]-outcomes[:,left]
            precision=difference_weight/(.25+contrast.var())
            normal[left,left]+=precision;normal[right,right]+=precision
            normal[left,right]-=precision;normal[right,left]-=precision
            target[left]-=precision*contrast.mean();target[right]+=precision*contrast.mean()
        fitted=np.linalg.solve(normal,target).clip(-1.,1.)
        packet={k:torch.as_tensor(v[None]) for k,v in group['packet'].items()}
        context,order=public_return_context(packet,True,5,torch.as_tensor(group['reference'][None]))
        root=len(contexts);contexts.append(context[0].numpy());order=order[0].numpy()
        for index,action in enumerate(candidates):
            programs.append(action[order][:,[3,5,4,8]])
            indices.append(root);values.append(fitted[index])
            variances.append(gain[:,index].var()/max(1,len(gain)))
    return dict(contexts=np.asarray(contexts),programs=np.asarray(programs),roots=indices,
        values=values,variances=variances,bandwidth=bandwidth,margin=margin,
        learning_objective='paired continuous-program differences' if difference_weight else 'paired program means')


def response_feedback_context(packet, water_relative=False):
    context, order = public_return_context(packet, True, 2)
    if water_relative:
        motion = packet['motion']; n = motion.shape[1]
        h = motion[:, 0, 2]; body = 30.*packet['node'][:, 0, 2:4]
        target = motion[:, 0, :2] + torch.stack((h.cos()*body[:, 0]-h.sin()*body[:, 1],
                                               h.sin()*body[:, 0]+h.cos()*body[:, 1]), -1)
        axis = -target/target.norm(dim=-1, keepdim=True).clamp_min(1.)
        velocity = 5.*packet['water']
        rotated = torch.stack(((velocity[..., :2]*axis[:, None]).sum(-1),
            axis[:, None, 0]*velocity[..., 1]-axis[:, None, 1]*velocity[..., 0]), -1)/3.
        rows = context[:, :8*n].reshape(-1, n, 8).clone()
        rows[..., 2:4] = rotated.gather(1, order[..., None].expand(-1, -1, 2))
        rows[..., 6] = velocity[..., 2].gather(1, order)
        context = torch.cat((rows.flatten(1), context[:, 8*n:]), -1)
    return context, order


@torch.no_grad()
def actuator_joint_design(n,device,interaction_order=2):
    """Balanced complete plans for pair or three-vessel feedback effects."""
    if interaction_order not in (2,3):
        raise ValueError('Actuator designs support pair or three-vessel interactions.')
    if n>10:raise ValueError('The balanced actuator design supports at most ten vessels.')
    index=torch.arange(9**interaction_order,device=device)
    first=index//(9**(interaction_order-1))
    second=(index//(9**(interaction_order-2)))%9
    third=index%9 if interaction_order==3 else None
    columns=[]
    # Evaluate degree-one/two polynomials over GF(9)=GF(3)[a]/(a*a+1).
    # Any selected two/three vessel columns cover every joint assignment.
    def multiply(left,right):
        return (((left%3)*(right%3)+2*(left//3)*(right//3))%3
            +3*(((left%3)*(right//3)+(left//3)*(right%3))%3))
    for vessel in range(n):
        slope=vessel
        if slope==9:columns.append(second if third is None else third);continue
        product=multiply(slope,second)
        real=first%3+product%3;imaginary=first//3+product//3
        if third is not None:
            quadratic=multiply(multiply(slope,slope),third)
            real=real+quadratic%3;imaginary=imaginary+quadratic//3
        columns.append(real%3+3*(imaginary%3))
    return torch.stack(columns,-1)


@torch.no_grad()
def coupled_actuator_risk(actor,packet,source,plans,return_clearance=False,synchronized=False,horizon=None,
                          tail_horizon=0.,target_paths=None,target_response=None):
    """Forecast unchanged source feedback inside each complete fleet plan."""
    b,m,n=plans.shape[:3];dt=min(.05,actor.prediction_step) if synchronized else actor.prediction_step
    initial=(torch.cat((packet['motion'][...,:3],5.*packet['water']),-1)
        if synchronized else actor.prediction_motion(packet))
    state=initial[:,None].expand(-1,m,-1,-1).clone()
    measured_velocity=packet['motion'][:,None,:,3:].expand(-1,m,-1,-1)
    force=750.*(plans*source[:,None]+(1.-plans)*packet['node'][:,None,:,11:13])+250.
    minimum=None;feedback_steps=max(1,round(.2/dt))
    trajectories=[] if return_clearance in ('paths','task_paths') else None
    target_trajectories=[] if return_clearance=='task_paths' else None
    steps=round((actor.risk_horizon if horizon is None else horizon)/dt)
    if target_paths is not None and (target_paths.shape!=(b,steps,2)
                                     or not bool(torch.isfinite(target_paths).all())):
        raise ValueError('Public target scenarios must align with the complete feedback forecast.')
    if trajectories is not None and tail_horizon:
        raise ValueError('Complete feedback paths require an explicit simulated horizon.')
    if target_response is not None:
        if target_paths is not None or target_trajectories is None or abs(feedback_steps*dt-.2)>1e-9:
            raise ValueError('Reactive target prediction needs complete fleet paths at the source control interval.')
        public_target,public_velocity=actor.public_target_motion(packet)
        response_target=public_target[:,None].expand(-1,m,-1).clone()
        response_velocity=public_velocity[:,None].expand(-1,m,-1).clone()
        response_boids=packet['node'][:,None,:,11:13].expand(-1,m,-1,-1)
    task_guard=return_clearance=='task'
    if task_guard:
        target,target_velocity=actor.public_target_motion(packet)
        captured=torch.zeros((b,m),dtype=torch.bool,device=source.device)
        exposure=None
    for step in range(steps):
        response_boundary=target_response is not None and step%feedback_steps==0
        if response_boundary and step:
            response_target=next_target;response_velocity=next_velocity
        if actor.risk_feedback and step and step%feedback_steps==0:
            observed=torch.cat((state[...,:3],measured_velocity),-1) if synchronized else state
            if target_response is not None:
                obs,boids=actor.feedback_observation(packet,observed,step*dt,joint=True,
                    target_override=(response_target,response_velocity))
                response_boids=boids
            elif target_paths is None:
                obs,boids=actor.feedback_observation(packet,observed,step*dt,joint=True)
            else:
                preceding=(target_paths[:,step-2] if step>1 else actor.public_target_motion(packet)[0])
                target_state=(target_paths[:,step-1],(target_paths[:,step-1]-preceding)/dt)
                obs,boids=actor.feedback_observation(packet,observed,step*dt,joint=True,
                    target_override=target_state)
            future=actor.proposal_distribution(obs.reshape(b*m,n,-1))[0].tanh().reshape(b,m,n,2)
            force=750.*(plans*future+(1.-plans)*boids)+250.
        if response_boundary:
            observed=torch.cat((state[...,:3],measured_velocity),-1) if synchronized else state
            if step and not actor.risk_feedback:
                _,response_boids=actor.feedback_observation(packet,observed,step*dt,joint=True,
                    target_override=(response_target,response_velocity))
            next_target,next_velocity=target_response.predict_state(observed.reshape(b*m,n,6),
                response_boids.reshape(b*m,n,2),response_target.reshape(b*m,2),response_velocity.reshape(b*m,2))
            next_target=next_target.reshape(b,m,2);next_velocity=next_velocity.reshape(b,m,2)
        x,y,heading,vx,vy,yaw_rate=state.unbind(-1)
        measured_velocity=state[...,3:]
        x=x+dt*vx;y=y+dt*vy;heading=heading+dt*yaw_rate
        cosine,sine=heading.cos(),heading.sin();u=cosine*vx+sine*vy;v=-sine*vx+cosine*vy
        du=(400.*yaw_rate*v-(100.+150.*u.abs())*u+force.sum(-1))/380.
        dv=(-380.*yaw_rate*u-(100.+150.*v.abs())*v)/400.
        dr=(-20.*u*v-(980.+950.*yaw_rate.abs())*yaw_rate+1.25*(force[...,0]-force[...,1]))/1430.
        u=u+dt*du;v=v+dt*dv
        state=torch.stack((x,y,heading,cosine*u-sine*v,sine*u+cosine*v,yaw_rate+dt*dr),-1)
        if trajectories is not None:
            trajectories.append(state[...,:2])
            if target_trajectories is not None:
                if target_response is not None:
                    fraction=(step%feedback_steps+1)/feedback_steps
                    target_trajectories.append(response_target+fraction*(next_target-response_target))
                else:
                    target,velocity=actor.public_target_motion(packet)
                    predicted=(target_paths[:,step] if target_paths is not None else
                        actor.target_motion.predict(packet,(step+1)*dt)[0]
                        if getattr(actor,'target_motion',None) is not None and actor.interaction_enabled
                        else target+(step+1)*dt*velocity)
                    target_trajectories.append(predicted[:,None].expand(-1,m,-1))
            continue
        distance=(state[:,:,:,:2][:,:,:,None]-state[:,:,:,:2][:,:,None]).norm(dim=-1)
        minimum=distance if minimum is None else torch.minimum(minimum,distance)
        if task_guard:
            time=(step+1)*dt
            predicted_target=(target_paths[:,step] if target_paths is not None else
                (actor.target_motion.predict(packet,time)[0]
                 if getattr(actor,'target_motion',None) is not None and actor.interaction_enabled
                 else target+time*target_velocity))
            reach=(state[...,:2]-predicted_target[:,None,None]).norm(dim=-1).amin(-1)
            if (step+1)%feedback_steps==0:
                captured|=reach<5.
            # The source task fails when the target passes every defender
            # relative to the protected asset. Preserve its best interceptor,
            # rather than summing pursuit distances and encouraging crowding.
            outside=state[...,:2].norm(dim=-1).amin(-1)-predicted_target.norm(dim=-1)[:,None]
            exposure=outside if exposure is None else torch.maximum(exposure,outside)
    if trajectories is not None:
        paths=torch.stack(trajectories,-2)
        return (paths,torch.stack(target_trajectories,-2)) if target_trajectories is not None else paths
    if tail_horizon:
        # Short feedback forecasts can look safe while leaving two boats
        # rapidly closing at their endpoint. Continue their predicted relative
        # velocity to its closest approach, without compounding the source
        # policy's long-horizon feedback prediction error.
        displacement=state[:,:,:,:2][:,:,:,None]-state[:,:,:,:2][:,:,None]
        velocity=state[:,:,:,3:5][:,:,:,None]-state[:,:,:,3:5][:,:,None]
        closest=(-(displacement*velocity).sum(-1)/velocity.square().sum(-1).clamp_min(1e-8)).clamp(0.,tail_horizon)
        continuation=(displacement+closest[...,None]*velocity).norm(dim=-1)
        minimum=torch.minimum(minimum,continuation)
    diagonal=torch.eye(n,dtype=torch.bool,device=source.device)[None,None]
    clearance=minimum.masked_fill(diagonal,float('inf')).amin((-1,-2))
    cost=(actor.distance_risk(minimum)*torch.triu(torch.ones((n,n),device=source.device),1)).sum((-1,-2))
    if task_guard:
        reach=torch.where(captured,reach.new_full((),5.),reach.clamp_min(5.))
        return cost,clearance,torch.stack((reach,exposure),-1)
    if return_clearance=='both':return cost,clearance
    return clearance if return_clearance else cost


def apply_actuator_preferences(packet,source_action,preferred_action,preferences):
    """Decode task preferences inside the unchanged source-command hull."""
    rule=packet['node'][...,11:13]
    command=preferred_action[...,:2]*preferred_action[...,2:3]+rule*(1.-preferred_action[...,2:3])
    proposed=preferences*source_action[...,:2]+(1.-preferences)*rule
    command=torch.where(preferences>=0.,proposed,command)
    return torch.cat((command,torch.ones_like(source_action[...,:1])),-1)


def refine_pair_blocks(unary, risk, plans, risk_weight, sweeps=1):
    """Optimize two vessels together while retaining every other fleet choice."""
    plans = plans.clone()
    b, n, k = unary.shape
    for _ in range(sweeps):
        previous = plans.clone()
        for first in range(n):
            for second in range(first+1, n):
                first_cost = unary[:, first, None].expand(-1, plans.shape[1], -1).clone()
                second_cost = unary[:, second, None].expand(-1, plans.shape[1], -1).clone()
                for other in range(n):
                    if other in (first, second):
                        continue
                    choices = plans[:, None, :, other].expand(-1, k, -1)
                    first_cost += risk_weight*risk[:, first, other].gather(2, choices).transpose(1, 2)
                    second_cost += risk_weight*risk[:, second, other].gather(2, choices).transpose(1, 2)
                cost = (first_cost[..., None]+second_cost[..., None, :]
                        +risk_weight*risk[:, first, second, None])
                flat = cost.flatten(-2)
                choice = flat.argmin(-1)
                current = plans[:, :, first]*k+plans[:, :, second]
                # Preserve the existing maneuver on an exact tie. A block
                # change must improve this fleet's interaction objective.
                better = flat.gather(2, choice[..., None]).squeeze(-1) < flat.gather(
                    2, current[..., None]).squeeze(-1)
                plans[:, :, first] = torch.where(better, choice//k, plans[:, :, first])
                plans[:, :, second] = torch.where(better, choice % k, plans[:, :, second])
        if torch.equal(previous, plans):
            break
    return plans


def distinct_fleet_shortlist(plans, scores, limit):
    """Retain distinct fleet maneuvers in increasing proposal-cost order."""
    if limit < 1:
        raise ValueError('A fleet shortlist must retain at least one plan.')
    count = min(limit, plans.shape[1])
    indices = []
    for rows, ranking in zip(plans.detach().cpu().tolist(), scores.argsort(
            dim=1, stable=True).detach().cpu().tolist()):
        selected = []; seen = set()
        for index in ranking:
            code = tuple(rows[index])
            if code in seen:
                continue
            seen.add(code); selected.append(index)
            if len(selected) == count:
                break
        indices.append(selected+[selected[-1]]*(count-len(selected)))
    indices = torch.tensor(indices, dtype=torch.long, device=plans.device)
    return plans.gather(1, indices[..., None].expand(-1, -1, plans.shape[-1]))


def pareto_fleet_refinement(unary, risk, plans, reference, risk_weight, componentwise=False):
    """Replace a fleet plan only when both control deviation and risk improve."""
    b, n, k = unary.shape
    catalog = torch.cat((reference[:, None], plans), 1)
    deviation = unary.new_zeros(catalog.shape[:2])
    interaction = torch.zeros_like(deviation)
    protected = torch.ones_like(deviation, dtype=torch.bool)
    for vessel in range(n):
        own = unary[:, vessel].gather(1, catalog[:, :, vessel])
        deviation += own
        if componentwise:
            protected &= own <= own[:, :1]
        for other in range(vessel):
            code = catalog[:, :, other]*k+catalog[:, :, vessel]
            pair = risk[:, other, vessel].flatten(1).gather(1, code)
            interaction += pair
            if componentwise:
                protected &= pair <= pair[:, :1]
    score = deviation+risk_weight*interaction
    admissible = ((deviation <= deviation[:, :1]) & (interaction <= interaction[:, :1])
                  & (score < score[:, :1]) & protected)
    admissible[:, 0] = True
    selected = score.masked_fill(~admissible, float('inf')).argmin(-1)
    return catalog[torch.arange(b, device=unary.device), selected]


def feedback_guarded_refinement(actor, packet, source, candidates, order, reference, proposed, horizons,
                                tail_horizon=0., task_guard=False):
    """Keep a proposal only when shared fleet feedback agrees across horizons."""
    if not horizons or any(not isfinite(horizon) or horizon <= 0 for horizon in horizons):
        raise ValueError('Feedback refinement requires positive forecast horizons.')
    if not isfinite(tail_horizon) or tail_horizon < 0:
        raise ValueError('Feedback refinement requires a finite nonnegative tail horizon.')
    changed = (reference != proposed).any(-1)
    if not bool(changed.any()):
        return reference
    subset = {key: (value[changed] if torch.is_tensor(value) and value.ndim
                   and value.shape[0] == len(source) else value) for key, value in packet.items()}
    choices = torch.stack((reference[changed], proposed[changed]), 1)
    catalog = candidates[changed, None].expand(-1, 2, -1, -1, -1)
    gates = catalog.gather(3, choices[..., None, None].expand(-1, -1, -1, 1, 2)).squeeze(3)
    physical = gates.gather(2, order[changed].argsort(-1)[:, None, :, None].expand(-1, 2, -1, 2))
    admissible = torch.ones(choices.shape[0], dtype=torch.bool, device=source.device)
    for horizon in horizons:
        settings = dict(synchronized=True, horizon=horizon)
        if tail_horizon:
            settings['tail_horizon'] = tail_horizon
        forecast = coupled_actuator_risk(
            actor, subset, source[changed], physical, 'task' if task_guard else 'both', **settings)
        risk, clearance = forecast[:2]
        admissible &= ((risk[:, 1] <= risk[:, 0]) & (clearance[:, 1] >= clearance[:, 0])
                       & torch.isfinite(risk).all(-1) & torch.isfinite(clearance).all(-1))
        if task_guard:
            progress = forecast[2]
            admissible &= ((progress[:, 1] <= progress[:, 0]).all(-1)
                           & torch.isfinite(progress).all((-1, -2)))
    result = reference.clone()
    result[changed] = torch.where(admissible[:, None], proposed[changed], reference[changed])
    return result


def actuator_candidate_grid(preferred, subdivisions=1, *, include_uniform_midpoint=False):
    """Resolve each side of the source preference without changing its hull."""
    if isinstance(subdivisions, bool) or not isinstance(subdivisions, int) or subdivisions < 1:
        raise ValueError('Candidate subdivisions must be a positive integer.')
    if not isinstance(include_uniform_midpoint, bool):
        raise ValueError('The common midpoint option must be boolean.')
    if subdivisions == 1:
        values = torch.stack((torch.zeros_like(preferred), preferred, torch.ones_like(preferred)), -2)
    else:
        fraction = torch.arange(subdivisions, device=preferred.device, dtype=preferred.dtype)/subdivisions
        lower = preferred[..., None, :]*fraction[:, None]
        upper = preferred[..., None, :] + (1.-preferred[..., None, :])*fraction[:, None]
        values = torch.cat((lower, upper, torch.ones_like(preferred[..., None, :])), -2)
    if include_uniform_midpoint:
        values = torch.cat((values, torch.full_like(preferred[..., None, :], .5)), -2)
    levels = values.shape[-2]
    digits = torch.arange(levels*levels, device=preferred.device)
    return torch.stack((values[..., digits//levels, 0], values[..., digits%levels, 1]), -1)


def independent_fleet_minimum_cost(values, probabilities):
    """Exact expected fleet minimum when only one candidate choice is free."""
    if values.shape != probabilities.shape or values.ndim != 3:
        raise ValueError('Fleet candidate values and probabilities must have shape batch, vessel, candidate.')
    support = torch.cat((values.new_zeros((len(values), 1)), values.flatten(1)), -1).sort(-1).values
    lower, intervals = support[:, :-1], support.diff(dim=-1)
    survives = ((values[..., None] > lower[:, None, None])*probabilities[..., None]).sum(2)
    costs = []
    for vessel in range(values.shape[1]):
        peers = [other for other in range(values.shape[1]) if other != vessel]
        survival = survives[:, peers].prod(1)
        costs.append((intervals[:, None]*survival[:, None]*
                      (values[:, vessel, :, None] > lower[:, None])).sum(-1))
    return torch.stack(costs, 1)


def task_stopped_fusion_factors(paths, target_path, unary, distance_risk, *, control_stride=2,
                               capture_radius=5.):
    """Reuse candidate paths as a fleet task that ends at its first capture.

    A collision or radial escape at the capture tick still has a cost, matching
    the source environment's termination priority. Later predicted encounters
    have no cost. All factors are shared by joint and independent selection.
    """
    b, n, k, horizon, coordinates = paths.shape
    if coordinates != 2 or target_path.shape != (b, horizon, 2) or unary.shape != (b, n, k):
        raise ValueError('Task-stopped fusion needs aligned candidate and target paths.')
    if control_stride < 1 or capture_radius <= 0:
        raise ValueError('Task stopping needs a positive control stride and capture radius.')
    ticks = torch.arange(horizon, device=paths.device)
    observed = (ticks+1) % control_stride == 0
    capture = ((paths-target_path[:, None, None]).norm(dim=-1) < capture_radius) & observed
    first = torch.where(capture, ticks, horizon-1).amin(-1)
    alive = torch.cat((torch.ones_like(capture[..., :1]),
                       ~capture[..., :-1].cummax(-1).values), -1)
    pairs, cumulative, increments = [], [], []
    for left in range(n):
        for right in range(left+1, n):
            distance = (paths[:, left, :, None]-paths[:, right, None]).norm(dim=-1)
            risk = distance_risk(distance).cummax(-1).values
            pairs.append((left, right)); cumulative.append(risk)
            increments.append(torch.cat((risk[..., :1], risk.diff(dim=-1)), -1))
    # The source task fails when every defender is farther from the asset
    # than the attacker. The fleet minimum preserves a remaining interceptor;
    # adding individual pursuit distances would instead encourage crowding.
    exposure = (paths.norm(dim=-1)-target_path.norm(dim=-1)[:, None, None]).clamp_min(0.).square()
    exposure = exposure*observed
    return dict(unary=unary, first_capture=first, alive=alive, pairs=pairs,
                pair_indices=torch.tensor(pairs, dtype=torch.long, device=paths.device),
                cumulative_risk=torch.stack(cumulative, 1), risk_increments=increments,
                exposure=exposure, observed_ticks=max(1, int(observed.sum().item())))


def target_turning_envelope(position, central_path, degrees):
    """Construct public target futures with a common observed motion prefix.

    Deflection grows from zero with forecast time; an instantaneous rotation
    of the measured velocity would otherwise invent an unobserved maneuver.
    The center is unchanged and no controller, agility, or future is queried.
    """
    if not isfinite(degrees) or not 0. <= degrees <= 90.:
        raise ValueError('The public target turning envelope must be within zero to ninety degrees.')
    if central_path.ndim != 3 or position.shape != central_path[:, 0].shape:
        raise ValueError('The target envelope requires aligned public positions and forecast paths.')
    if not degrees:
        return central_path[:, None], central_path.new_ones(1)
    fractions = torch.arange(1, central_path.shape[1]+1, device=central_path.device,
                             dtype=central_path.dtype)/central_path.shape[1]
    branches = central_path.new_tensor([-1., -.5, 0., .5, 1.])
    angle = branches[:, None]*fractions[None]*(degrees*torch.pi/180.)
    displacement = central_path-position[:, None]
    x, y = displacement[:, None, :, 0], displacement[:, None, :, 1]
    rotated = torch.stack((angle.cos()*x-angle.sin()*y,
                           angle.sin()*x+angle.cos()*y), -1)
    return position[:, None, None]+rotated, central_path.new_tensor([.1, .2, .4, .2, .1])


def task_envelope_fusion_factors(paths, target_paths, unary, distance_risk, *, weights,
                                 interception_weight, control_stride=2, capture_radius=5.):
    """Score complementary interception of shared public target futures.

    Both selectors reuse precisely the same forecasts. Five-dimensional paths
    share each vessel forecast across scenarios; six-dimensional paths carry
    the separately simulated feedback response to each shared target future.
    Each future contributes the fleet's best interception and its own first
    capture stopping time, allowing different vessels to cover different turns.
    """
    if paths.ndim not in (5,6):
        raise ValueError('Candidate paths need an optional target-scenario axis.')
    b=paths.shape[0];n,k,horizon,coordinates=paths.shape[-4:]
    if target_paths.ndim != 4 or target_paths.shape[0] != b or target_paths.shape[2:] != (horizon, 2):
        raise ValueError('Target futures must align with the common candidate forecast horizon.')
    scenarios = target_paths.shape[1]
    weights = torch.as_tensor(weights, device=paths.device, dtype=paths.dtype)
    if weights.shape != (scenarios,) or not torch.isfinite(weights).all() or not (weights > 0.).all():
        raise ValueError('Every shared target future needs a finite positive weight.')
    if not isfinite(interception_weight) or interception_weight < 0.:
        raise ValueError('The fleet interception weight must be finite and nonnegative.')
    if paths.ndim==6 and paths.shape[1]!=scenarios:
        raise ValueError('Each target future needs its matching fleet feedback response.')
    repeated = (paths[:, None].expand(-1, scenarios, -1, -1, -1, -1)
                if paths.ndim==5 else paths).reshape(b*scenarios, n, k, horizon, coordinates)
    targets = target_paths.reshape(b*scenarios, horizon, 2)
    repeated_unary = unary[:, None].expand(-1, scenarios, -1, -1).reshape(b*scenarios, n, k)
    components = task_stopped_fusion_factors(repeated, targets, repeated_unary, distance_risk,
        control_stride=control_stride, capture_radius=capture_radius)
    observed = (torch.arange(horizon, device=paths.device)+1) % control_stride == 0
    if not bool(observed.any()):
        raise ValueError('Fleet interception needs at least one forecast control tick.')
    distance = (repeated-targets[:, None, None]).norm(dim=-1).masked_fill(~observed, float('inf'))
    components['interception_deficit'] = (distance.amin(-1)-capture_radius).clamp_min(0.)/capture_radius
    components['interception_weight'] = interception_weight
    return dict(unary=unary, scenario_factors=components, scenario_weights=weights/weights.sum())


def task_stopped_fleet_cost(factors, plans, risk_weight=32., escape_weight=0.):
    """Evaluate complete fleet plans without making any extra motion forecast."""
    unary = factors['unary']; b, n, k = unary.shape
    if plans.ndim != 3 or plans.shape[0] != b or plans.shape[-1] != n:
        raise ValueError('Complete candidate plans must have shape batch, plan, vessel.')
    if 'scenario_factors' in factors:
        weights = factors['scenario_weights']; scenarios = len(weights)
        repeated = plans[:, None].expand(-1, scenarios, -1, -1).reshape(b*scenarios, -1, n)
        costs = task_stopped_fleet_cost(factors['scenario_factors'], repeated, risk_weight,
                                        escape_weight).reshape(b, scenarios, -1)
        return (costs*weights[None, :, None]).sum(1)
    index = plans.transpose(1, 2)
    stop = factors['first_capture'].gather(2, index).amin(1)
    cost = unary.gather(2, index).sum(1)
    horizon = factors['alive'].shape[-1]
    pairs = factors['pair_indices']
    pair_index = ((plans[:, :, pairs[:, 0]]*k+plans[:, :, pairs[:, 1]])*horizon
                  +stop[..., None]).transpose(1, 2)
    cost = cost+risk_weight*factors['cumulative_risk'].flatten(2).gather(2, pair_index).sum(1)
    if escape_weight:
        exposure = factors['exposure'].gather(2, index[..., None].expand(-1, -1, -1, horizon)).amin(1)
        alive = torch.arange(horizon, device=plans.device) <= stop[..., None]
        cost = cost+escape_weight*(exposure*alive).sum(-1)/factors['observed_ticks']
    if factors.get('interception_weight', 0.):
        cost = cost+factors['interception_weight']*factors['interception_deficit'].gather(2, index).amin(1)
    return cost


def task_stopped_independent_cost(factors, probabilities, risk_weight=32., escape_weight=0.):
    """Exact conditional expectation of the SAME fleet cost for each vessel.

    Each subject optimizes independently against the peers' unchanged source
    distribution. Products of capture-survival probabilities integrate every
    peer assignment exactly; no Monte Carlo or coupled choice is hidden here.
    """
    unary = factors['unary']; b, n, k = unary.shape
    if probabilities.shape != unary.shape:
        raise ValueError('Independent task stopping needs the shared candidate distribution.')
    if 'scenario_factors' in factors:
        weights = factors['scenario_weights']; scenarios = len(weights)
        repeated = probabilities[:, None].expand(-1, scenarios, -1, -1).reshape(b*scenarios, n, k)
        costs = task_stopped_independent_cost(factors['scenario_factors'], repeated, risk_weight,
                                              escape_weight).reshape(b, scenarios, n, k)
        return (costs*weights[None, :, None, None]).sum(1)
    alive = factors['alive'].to(unary.dtype)
    weighted_alive = probabilities[..., None]*alive
    survival = weighted_alive.sum(2)
    expected_unary = (unary.masked_fill(probabilities == 0., 0.)*probabilities).sum(-1)
    cost = unary+expected_unary.sum(-1)[:, None, None]-expected_unary[..., None]
    for (left, right), increment in zip(factors['pairs'], factors['risk_increments']):
        pair = torch.einsum('bgt,bht,bght->bt',
            weighted_alive[:, left], weighted_alive[:, right], increment)
        for subject in range(n):
            if subject == left:
                term = torch.einsum('bht,bght->bgt', weighted_alive[:, right], increment)*alive[:, left]
            elif subject == right:
                term = torch.einsum('bgt,bght->bht', weighted_alive[:, left], increment)*alive[:, right]
            else:
                term = pair[:, None]*alive[:, subject]
            others = [vessel for vessel in range(n) if vessel not in (left, right, subject)]
            if others:
                term = term*survival[:, others].prod(1)[:, None]
            cost[:, subject] += risk_weight*term.sum(-1)
    if escape_weight:
        horizon = alive.shape[-1]
        # Captured candidates carry an absorbing zero exposure after capture.
        # Taking the fleet minimum therefore stops the escape cost at exactly
        # the same first capture as the joint objective.
        values = (factors['exposure']*alive).permute(0, 3, 1, 2).reshape(b*horizon, n, k)
        probability = probabilities[:, None].expand(-1, horizon, -1, -1).reshape(b*horizon, n, k)
        expected = independent_fleet_minimum_cost(values, probability).reshape(b, horizon, n, k)
        cost += escape_weight*expected.sum(1)/factors['observed_ticks']
    if factors.get('interception_weight', 0.):
        cost += factors['interception_weight']*independent_fleet_minimum_cost(
            factors['interception_deficit'], probabilities)
    return cost


def select_task_stopped_fusion(factors, *, selection, beam_width, risk_weight, escape_weight):
    unary = factors['unary']; b, n, k = unary.shape
    reference = unary.argmin(-1)
    if selection in ('independent', 'single_reference'):
        probability = ((-unary).softmax(-1) if selection == 'independent' else
                       F.one_hot(reference, k).to(unary.dtype))
        return task_stopped_independent_cost(factors, probability, risk_weight, escape_weight).argmin(-1)
    if selection != 'joint':
        raise ValueError('Unknown task-stopped fusion selection rule.')
    choices = torch.arange(k, device=unary.device)
    plans = reference[:, None]
    for vessel in range(n):
        expanded = plans[:, :, None].expand(-1, -1, k, -1).clone()
        expanded[..., vessel] = choices
        expanded = expanded.flatten(1, 2)
        scores = task_stopped_fleet_cost(factors, expanded, risk_weight, escape_weight)
        index = scores.topk(min(beam_width, scores.shape[1]), largest=False).indices
        plans = expanded.gather(1, index[..., None].expand(-1, -1, n))
    uniform = choices[None, :, None].expand(b, -1, n)
    plans = torch.cat((plans, reference[:, None], uniform), 1)
    initial = plans.clone()
    for _ in range(3):
        previous = plans.clone()
        for vessel in range(n):
            alternatives = plans[:, :, None].expand(-1, -1, k, -1).clone()
            alternatives[..., vessel] = choices
            scores = task_stopped_fleet_cost(factors, alternatives.flatten(1, 2), risk_weight,
                                             escape_weight).reshape(b, plans.shape[1], k)
            best = scores.argmin(-1)
            current = scores.gather(-1, plans[:, :, vessel, None]).squeeze(-1)
            improve = scores.gather(-1, best[..., None]).squeeze(-1) < current
            plans[:, :, vessel] = torch.where(improve, best, plans[:, :, vessel])
        if torch.equal(previous, plans):
            break
    plans = torch.cat((initial, plans), 1)
    scores = task_stopped_fleet_cost(factors, plans, risk_weight, escape_weight)
    return plans[torch.arange(b, device=plans.device), scores.argmin(-1)]


class FeedbackFleetPolicy(nn.Module):
    """Learn whole-fleet candidate consequences from real closed-loop returns.

    Each graph is one complete candidate plan, including the response of its
    unchanged teammates. The independent control queries this exact same
    scorer on the reference-change slice of the common forecast library.
    """
    def __init__(self, width=48, analytic_temperature=2., state=None, candidate_contrast=False,
                 sampling_temperature=1., representation='candidate_graph', scenario_evaluation=False,
                 scenario_regret_weight=0.):
        super().__init__()
        if width < 8 or not isfinite(analytic_temperature) or analytic_temperature <= 0:
            raise ValueError('The fleet policy needs a positive width and analytic temperature.')
        self.width=width;self.analytic_temperature=float(analytic_temperature)
        if not isfinite(sampling_temperature) or sampling_temperature<=0:
            raise ValueError('Fleet exploration requires a positive finite sampling temperature.')
        self.sampling_temperature=float(sampling_temperature)
        self.candidate_contrast=bool(candidate_contrast)
        if not isinstance(scenario_evaluation,bool):
            raise ValueError('Separate scenario evaluation must be an explicit Boolean.')
        self.scenario_evaluation=scenario_evaluation
        if (not isfinite(scenario_regret_weight) or not 0.<=scenario_regret_weight<=1.
                or scenario_regret_weight and not scenario_evaluation):
            raise ValueError('Scenario regret requires complete futures and a weight between zero and one.')
        self.scenario_regret_weight=float(scenario_regret_weight)
        if representation not in ('candidate_graph','candidate_attention'):
            raise ValueError('Unknown complete-candidate representation.')
        if representation=='candidate_attention' and width%4:
            raise ValueError('Ordinary candidate attention needs a width divisible by four.')
        self.representation=representation
        self.node=nn.Sequential(nn.Linear(37,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
        if representation=='candidate_graph':
            self.message=nn.Sequential(nn.Linear(2*width+17,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU())
            self.update=nn.Sequential(nn.Linear(3*width,width),nn.SiLU())
        else:
            # The control receives the same complete-candidate node features.
            # Ordinary attention replaces explicit pair geometry and messages;
            # at width 48 its total capacity differs by only five parameters.
            self.attention=nn.TransformerEncoderLayer(width,4,width+5,dropout=0.,
                activation='gelu',batch_first=True,norm_first=True)
        self.plan=nn.Sequential(nn.Linear(2*width+1,width),nn.SiLU())
        self.actor=nn.Linear(width,1)
        self.critic=nn.Sequential(nn.Linear(2*width,width),nn.SiLU(),nn.Linear(width,1))
        nn.init.zeros_(self.actor.weight);nn.init.zeros_(self.actor.bias)
        self.sample=False;self.collect=False;self.last_decision=None
        self.collection_dtype=torch.float16
        if state is not None:self.load_state_dict(state)

    @staticmethod
    def features(packet,source,preferred,plans,paths,futures,weights,order,target):
        b,count,_,n,horizon,_=paths.shape
        context,_=public_return_context(packet,True,2)
        rows=context[:,:8*n].reshape(b,n,8)
        source=source.gather(1,order[...,None].expand_as(source))
        rule=packet['node'][...,11:13].gather(1,order[...,None].expand(-1,-1,2))
        indices=torch.tensor([0,(horizon-1)//3,2*(horizon-1)//3,horizon-1],device=paths.device)
        future=futures.index_select(-2,indices)
        relative=paths.index_select(-2,indices)-(future[:,None,:,None] if future.ndim==4 else future[:,:,:,None])
        direction=-target/target.norm(dim=-1,keepdim=True).clamp_min(1.)
        forward=direction[:,None,None,None,None]
        relative=torch.stack(((relative*forward).sum(-1),
            forward[...,0]*relative[...,1]-forward[...,1]*relative[...,0]),-1)/30.
        mass=weights.to(paths).reshape(1,1,-1,1,1,1)
        mean=(relative*mass).sum(2)
        spread=(((relative-mean[:,:,None]).square().sum(-1)*mass[...,0]).sum(2)).sqrt()
        distance=(relative.norm(dim=-1)*mass[...,0]).sum(2)
        global_state=torch.cat((context[:,-3:],target.norm(dim=-1,keepdim=True)/30.,
            context.new_full((b,1),n/6.)),-1)
        shared=torch.cat((rows,source,rule,preferred),-1)[:,None].expand(-1,count,-1,-1)
        return torch.cat((shared,plans,mean.flatten(-2),spread,distance,
            global_state[:,None,None].expand(-1,count,n,-1)),-1)

    @classmethod
    def scenario_features(cls,packet,source,preferred,plans,paths,futures,weights,order,target,costs):
        """Keep each complete future intact through the nonlinear fleet scorer."""
        b,count,scenarios,n=paths.shape[:4]
        if costs.shape!=(b,count,scenarios) or weights.shape!=(scenarios,):
            raise ValueError('Each complete future needs its matching fleet cost and probability.')
        pieces=[cls.features(packet,source,preferred,plans,paths[:,:,index:index+1],
            futures[:,:,index:index+1],weights.new_ones(1),order,target) for index in range(scenarios)]
        cost=costs[...,None,None].expand(-1,-1,-1,n,1)
        mass=weights[None,None,:,None,None].expand(b,count,-1,n,1)
        return torch.cat((torch.stack(pieces,2),cost,mass),-1)

    def forward(self,features,costs):
        if features.ndim==5:
            if not self.scenario_evaluation or features.shape[-1]!=39 or not features.shape[2]:
                raise ValueError('Scenario scoring needs complete public features, costs and probabilities.')
            logits=[];values=[];scenario_logits=[]
            for index in range(features.shape[2]):
                logit,value=self(features[:,:,index,:,:37],features[:,:,index,0,37])
                scenario_logits.append(logit)
                logits.append(logit*features[:,:,index,0,38])
                values.append(value*features[:,0,index,0,38])
            if self.scenario_regret_weight:
                scores=torch.stack(scenario_logits,-1)
                probabilities=features[:,:,:,0,38]
                regret=scores.amax(1,keepdim=True)-scores
                expected=(regret*probabilities).sum(-1)
                worst=regret.masked_fill(probabilities<=0.,-torch.inf).amax(-1)
                weight=self.scenario_regret_weight
                return -((1.-weight)*expected+weight*worst),torch.stack(values).sum(0)
            return torch.stack(logits).sum(0),torch.stack(values).sum(0)
        b,count,n,_=features.shape
        x=features.reshape(b*count,n,-1)
        if self.representation=='candidate_graph':
            difference=x[:,:,None]-x[:,None]
            trajectory=difference[...,16:24].reshape(b*count,n,n,4,2)
            geometry=torch.cat((difference[...,:4],difference[...,:2].norm(dim=-1,keepdim=True),
                trajectory.flatten(-2),trajectory.norm(dim=-1)),-1)
            own=self.node(x)
            mask=~torch.eye(n,dtype=torch.bool,device=x.device)[None,:,:,None]
            for _ in range(2):
                message=self.message(torch.cat((own[:,:,None].expand(-1,-1,n,-1),
                    own[:,None].expand(-1,n,-1,-1),geometry),-1))
                own=own+self.update(torch.cat((own,(message*mask).sum(2)/max(n-1,1),
                    message.masked_fill(~mask,-1e4).amax(2)),-1))
        else:
            own=self.node(x)
            for _ in range(2):own=self.attention(own)
        raw_cost=costs.clamp(0.,1e4)
        plans=self.plan(torch.cat((own.mean(1),own.amax(1),raw_cost.log1p().reshape(-1,1)),-1))
        plans=plans.reshape(b,count,-1)
        prior=-(raw_cost-raw_cost.amin(-1,keepdim=True)).clamp(max=80.)/self.analytic_temperature
        policy_features=plans
        if self.candidate_contrast:
            # Shared scene features can dominate the much smaller differences
            # between complete fleet plans. Assign policy credit to those
            # within-state differences while retaining absolute scene features
            # for the return baseline. The scale is a detached statistic of
            # this common candidate library, not a learned target or label.
            reference=plans[:,:81+9*n]
            center=reference.mean(1,keepdim=True)
            contrast=plans-center
            scale=(reference-center).square().mean(1,keepdim=True).sqrt().detach().clamp_min(.02)
            policy_features=contrast/scale
        logits=prior+self.actor(policy_features).squeeze(-1)
        value=self.critic(torch.cat((plans.mean(1),plans.amax(1)),-1)).squeeze(-1)
        return logits,value

    @staticmethod
    def policy_distribution(logits,selection,defenders,reference_start=81,temperature=1.):
        if not isfinite(temperature) or temperature<=0:
            raise ValueError('Fleet sampling temperature must be positive and finite.')
        logits=logits/temperature
        if selection=='joint':
            return torch.distributions.Categorical(logits=logits)
        if selection=='single_reference':
            stop=reference_start+9*defenders
            if logits.shape[-1]<stop:
                raise ValueError('Independent learning requires all single-vessel reference queries.')
            local=logits[:,reference_start:stop].reshape(len(logits),defenders,9)
            return torch.distributions.Independent(torch.distributions.Categorical(logits=local),1)
        raise ValueError('Fleet learning supports joint or single-reference independent selection.')

    def choose(self,features,costs,library,reference_start,selection):
        logits,value=self(features,costs)
        if not self.sample and not self.collect:
            return select_feedback_fleet_plan(-logits,library,reference_start,selection)
        distribution=self.policy_distribution(logits,selection,library.shape[-1],reference_start,
            self.sampling_temperature)
        index=(distribution.sample() if self.sample else logits.argmax(-1) if selection=='joint' else
            select_feedback_fleet_plan(-logits,library,reference_start,selection))
        if self.collect:
            self.last_decision=dict(features=features.detach().to('cpu',self.collection_dtype),
                costs=costs.detach().cpu(),index=index.detach().cpu(),
                logp=distribution.log_prob(index).detach().cpu(),value=value.detach().cpu())
        return (library[torch.arange(len(library),device=library.device),index]
            if selection=='joint' else index)


def feedback_fleet_library(unary):
    """Share balanced fleet plans and every single-vessel reference change.

    The same complete feedback forecasts serve both selectors. The reference
    control changes each vessel independently while fixing every teammate at
    its source-preferred candidate; it never uses the selected joint plan.
    """
    if unary.ndim != 3 or unary.shape[-1] != 9:
        raise ValueError('The balanced fleet feedback library requires nine actuator candidates.')
    b,n,k=unary.shape
    reference=unary.argmin(-1)
    balanced=actuator_joint_design(n,unary.device)[None].expand(b,-1,-1)
    local=reference[:,None,None].expand(-1,n,k,-1).clone()
    for vessel in range(n):
        local[:,vessel,:,vessel]=torch.arange(k,device=unary.device)
    return torch.cat((balanced,local.flatten(1,2)),1),balanced.shape[1]


def select_feedback_fleet_plan(costs,library,reference_start,selection):
    """Choose jointly or select each vessel against a fixed teammate population."""
    b,count,n=library.shape
    reference_stop=reference_start+n*9
    if costs.shape!=(b,count) or count<reference_stop:
        raise ValueError('Feedback costs and the common fleet candidate library must align.')
    if selection=='joint':
        return library[torch.arange(b,device=costs.device),costs.argmin(-1)]
    if selection=='single_reference':
        return costs[:,reference_start:reference_stop].reshape(b,n,9).argmin(-1)
    if selection=='balanced_independent':
        # Every candidate is compared against its teammates in the same
        # balanced fleet design. No vessel conditions on another vessel's
        # selected action, and all choices are combined only after scoring.
        design=library[:,:reference_start]
        membership=F.one_hot(design,9).to(costs.dtype)
        count=membership.sum(1)
        if not reference_start or bool((count==0).any()):
            raise ValueError('Marginal independent selection needs every candidate in the balanced design.')
        marginal=(costs[:,:reference_start,None,None]*membership).sum(1)/count
        return marginal.argmin(-1)
    raise ValueError('Complete task-feedback prediction needs joint, reference or balanced-independent selection.')


@torch.no_grad()
def refine_feedback_relative_intervention(actor, packet, source_action, offsets, *,
                                         relative_scale=None, source_feedback=False, **configuration):
    """Keep recorded fusion displacements relative to live predecessor feedback.

    The requested coefficients define the first intervention command. Its
    displacement from the factual joint choice is retained for this window;
    every following command uses the newly observed state's joint choice.
    Unspecified peers receive that current factual choice, so their reaction is
    through the changed physical state rather than an altered activation plan.
    With a relative scale, the requested grid instead represents a bounded
    signed displacement; its midpoint is exactly the factual continuation.
    """
    if relative_scale is not None and (not isfinite(relative_scale) or not 0. < relative_scale <= 1.):
        raise ValueError('A relative fusion scale must be finite and lie in (0,1].')
    held = packet.get('_committed_actuator_preferences')
    shape = source_action[..., :2].shape
    if held is None or held.shape != shape or offsets.shape != shape:
        raise ValueError('Relative feedback needs current preferences and one two-actuator offset per vessel.')
    active = (held >= 0.).any(-1)
    neighborhood = packet.get('_policy_neighborhood')
    if neighborhood is not None and (neighborhood.shape != (len(source_action),)
            or neighborhood.dtype != torch.bool):
        raise ValueError('A policy-local execution flag must identify each fleet state.')
    maximum_active = 1 if neighborhood is None else 1+neighborhood.long()
    # A single-coordinate change to a one-vessel reference can retain that
    # reference component and activate one additional vessel. The collector
    # and decision catalog validate the one-coordinate comparison itself.
    if (bool((active.sum(-1) > maximum_active).any()) or
            bool((active & ~(held >= 0.).all(-1)).any()) or
            not bool(torch.isfinite(held).all()) or bool((held > 1.).any())):
        raise ValueError('Relative feedback may change one vessel with two bounded fusion coefficients.'
            if neighborhood is None else
            'Relative feedback must respect its recorded policy-local plan and bounded fusion coefficients.')
    initialized = torch.isfinite(offsets).all(-1)
    if (bool((~initialized & ~torch.isnan(offsets).all(-1)).any()) or
            bool((offsets[initialized].abs() > 1.).any())):
        raise ValueError('Relative offsets must be uninitialized pairs or finite displacements in [-1,1].')
    if source_feedback:
        if (configuration or getattr(actor, 'interaction_enabled', False)
                or getattr(actor, 'actuator_fusion_refinement', None)
                or getattr(actor, 'generic_fusion_adapter', None) is not None
                or getattr(actor, 'response_feedback_value', None) is not None):
            raise ValueError('Source-relative feedback requires the native Adapter without mechanism one.')
        # F01 uses the same intervention rule on its own source policy. Neither
        # candidate forecasts nor the joint selector participate in this path.
        factual = source_action[..., :3]
        coefficients = source_action[..., 2:3].expand(-1, -1, 2)
    else:
        factual_packet = dict(packet)
        factual_packet.pop('_committed_actuator_preferences', None)
        factual_packet.pop('_isolate_factual_actuator_start', None)
        factual_packet['_preferred_actuator_coefficients'] = source_action[..., 2:3].expand(-1, -1, 2)
        factual_preference = source_action
        if getattr(actor, 'generic_fusion_adapter', None) is not None:
            factual_preference = actor.generic_fusion_adapter.control(actor, factual_packet, source_action)
        factual, coefficients = _refine_actuator_interactions(actor, factual_packet,
            source_action, factual_preference, return_coefficients=True, **configuration)
    displacement = held-coefficients if relative_scale is None else relative_scale*(2.*held-1.)
    offsets = torch.where((active & ~initialized)[..., None], displacement, offsets)
    resolved = (coefficients+torch.nan_to_num(offsets)).clamp(0., 1.)
    if relative_scale is None:
        resolved = torch.where((active & ~initialized)[..., None], held, resolved)
    rule = packet['node'][..., 11:13]
    command = resolved*source_action[..., :2]+(1.-resolved)*rule
    replacement = torch.cat((command, torch.ones_like(source_action[..., 2:3])), -1)
    # Preserve the exact factual path for a null displacement, including
    # roundoff and all peer controls. An inactive vessel never inherits NaNs.
    changed = active & (torch.nan_to_num(offsets).abs().amax(-1) > 1e-7)
    return torch.where(changed[..., None], replacement, factual), offsets


@torch.no_grad()
def refine_actuator_interactions(actor,packet,source_action,preferred_action,**configuration):
    """Apply a local intervention after the already chosen factual fleet plan.

    At its first active control tick, exactly one vessel changes its mixture.
    The other controls equal the factual joint decision at this same state.
    Later ticks use the ordinary coupled feedback under the held intervention.
    """
    isolated=packet.get('_isolate_factual_actuator_start')
    if isolated is None or not bool(isolated.any()):
        return _refine_actuator_interactions(actor,packet,source_action,preferred_action,**configuration)
    if isolated.shape!=(len(source_action),) or isolated.dtype!=torch.bool:
        raise ValueError('Local intervention activation needs one boolean per fleet state.')
    held=packet.get('_committed_actuator_preferences')
    if held is None or held.shape!=source_action[...,:2].shape:
        raise ValueError('Local intervention activation needs its actual held actuator coefficients.')
    active=(held>=0.).any(-1)
    if not bool((active[isolated].sum(-1)==1).all()):
        raise ValueError('An isolated intervention must change exactly one vessel.')
    def subset(mask):
        return {key:(value[mask] if torch.is_tensor(value) and value.ndim
                     and value.shape[0]==len(source_action) else value)
                for key,value in packet.items() if key!='_isolate_factual_actuator_start'}
    result=torch.empty_like(source_action[...,:3])
    if bool((~isolated).any()):
        result[~isolated]=_refine_actuator_interactions(actor,subset(~isolated),
            source_action[~isolated],preferred_action[~isolated],**configuration)
    factual_packet=subset(isolated)
    factual_packet.pop('_committed_actuator_preferences',None)
    factual_source=source_action[isolated]
    factual_packet['_preferred_actuator_coefficients']=factual_source[...,2:3].expand(-1,-1,2)
    factual_preference=factual_source
    if getattr(actor,'generic_fusion_adapter',None) is not None:
        factual_preference=actor.generic_fusion_adapter.control(actor,factual_packet,factual_source)
    factual=_refine_actuator_interactions(actor,factual_packet,factual_source,factual_preference,**configuration)
    rule=factual_packet['node'][...,11:13]
    command=factual[...,:2]*factual[...,2:3]+rule*(1.-factual[...,2:3])
    replacement=held[isolated]*factual_source[...,:2]+(1.-held[isolated])*rule
    command=torch.where(active[isolated,...,None],replacement,command)
    result[isolated]=torch.cat((command,torch.ones_like(factual_source[...,2:3])),-1)
    return result


def _refine_actuator_interactions(actor,packet,source_action,preferred_action,*,beam_width=128,
                                 risk_weight=32.,preference_width=.4,selection='joint',
                                 source_preference=False,prior_map=None,coupled_prediction=False,
                                 contact_distance=5.,synchronized_prediction=False,
                                 commit_interventions=False,refine_all_plans=False,pair_refinements=0,
                                 joint_feedback_topk=0,joint_feedback_horizon=None,
                                 pareto_refinement=False,pareto_feedback_horizons=(),
                                 pareto_componentwise=False,pareto_feedback_tail=0.,
                                 pareto_task_guard=False,
                                 coupled_horizon=None,coupled_recovery=False,recovery_horizon=.6,
                                 recovery_distance=None,interaction_order=2,expand_interactions=False,
                                 coupled_tail=0.,coupled_refinements=0,interior_fusion=None,
                                 candidate_subdivisions=1,capture_weight=0.,include_uniform_midpoint=False,
                                 task_stopping=False,escape_weight=0.,
                                 target_envelope_degrees=0.,target_envelope_weight=0.,
                                 task_feedback_prediction=False,independent_feedback_proposal=False,
                                 nominal_feedback_proposals=False,feedback_refinement_rounds=0,
                                 return_coefficients=False,candidate_output=None):
    """Jointly reconcile task preferences with two-actuator candidate motion.

    The conditional value proposes a fusion preference. Candidate interaction
    remains active during its execution window, with both actuator mixtures
    inside the original SAC/Boids hull. Beam search retains complete fleet
    combinations; subsequent coordinate minimization removes simple pruning
    errors. The independent selector receives exactly the same candidates.
    """
    if candidate_output is not None:
        if not isinstance(candidate_output,dict) or candidate_output:
            raise ValueError('The joint candidate output must be an empty dictionary.')
        if (selection not in ('joint','independent') or task_stopping or coupled_prediction
                or coupled_recovery or capture_weight or joint_feedback_topk
                or pareto_feedback_horizons or packet.get('_committed_actuator_preferences') is not None):
            raise ValueError('Unified candidates require the uncommitted factorized CI planner.')
    if return_coefficients and not (task_stopping and task_feedback_prediction):
        raise ValueError('Exact fusion coefficients require the complete task-feedback planner.')
    if independent_feedback_proposal and not (task_stopping and task_feedback_prediction):
        raise ValueError('Independent completion requires the common complete-feedback candidate library.')
    if (nominal_feedback_proposals or feedback_refinement_rounds) and not (task_stopping and task_feedback_prediction):
        raise ValueError('Feedback proposal search requires complete task-feedback prediction.')
    if not isinstance(feedback_refinement_rounds,int) or feedback_refinement_rounds<0:
        raise ValueError('Feedback refinement rounds must be a nonnegative integer.')
    if (pareto_feedback_horizons or pareto_componentwise) and (not pareto_refinement or not refine_all_plans
                                    or selection != 'joint' or coupled_prediction or joint_feedback_topk):
        raise ValueError('Protected refinement requires the Pareto fleet proposal and its exact reference.')
    if (pareto_feedback_tail or pareto_task_guard) and not pareto_feedback_horizons:
        raise ValueError('Tail and task protection require explicit feedback forecast horizons.')
    if (candidate_subdivisions != 1 or include_uniform_midpoint) and (coupled_prediction or coupled_recovery):
        raise ValueError('The balanced coupled design uses the original three-level candidates.')
    if not isfinite(capture_weight) or capture_weight < 0:
        raise ValueError('Fleet interception weight must be finite and nonnegative.')
    if not isfinite(escape_weight) or escape_weight < 0 or (escape_weight and not task_stopping):
        raise ValueError('Radial escape protection requires the task-stopped objective and a nonnegative weight.')
    if (not isfinite(target_envelope_weight) or target_envelope_weight < 0.
            or (target_envelope_weight and not task_stopping)
            or (target_envelope_degrees and not target_envelope_weight)):
        raise ValueError('Target-future coverage requires task stopping and a positive interception weight.')
    if not isfinite(target_envelope_degrees) or not 0. <= target_envelope_degrees <= 90.:
        raise ValueError('The public target turning envelope must be within zero to ninety degrees.')
    if task_feedback_prediction and (not task_stopping or candidate_subdivisions!=1
                                    or include_uniform_midpoint
                                    or selection not in ('joint','single_reference','balanced_independent')):
        raise ValueError('Task feedback needs the shared nine-candidate joint or independent comparison.')
    if task_stopping and (coupled_prediction or coupled_recovery or capture_weight or joint_feedback_topk
                          or pareto_feedback_horizons):
        raise ValueError('Task-stopped fusion reuses the common factorized candidate paths.')
    if capture_weight and (coupled_prediction or coupled_recovery or pair_refinements
                           or joint_feedback_topk or pareto_refinement):
        raise ValueError('Fleet interception uses the shared factorized candidate forecast and objective.')
    if (selection in ('independent','single_reference','balanced_independent') and not source_preference
            and getattr(actor,'fusion_selection','joint')!=selection):
        raise ValueError('Independent actuator selection requires source_preference=True; '
            'or the preceding fusion selector must also be independent.')
    source=source_action[...,:2];rule=packet['node'][...,11:13]
    command=preferred_action[...,:2]*preferred_action[...,2:3]+rule*(1.-preferred_action[...,2:3])
    difference=source-rule
    preferred=torch.where(difference.abs()>1e-5,(command-rule)/difference.masked_fill(difference.abs()<=1e-5,1.),
                          source_action[...,2:3]).clamp(0.,1.)
    explicit=packet.get('_preferred_actuator_coefficients')
    if explicit is not None:
        # These coefficients are already the decision variables. Recovering
        # them from a mixed force divides by source minus rule, which is
        # ill-conditioned when the source policies produce similar commands.
        preferred=explicit
    else:
        native=(preferred_action[...,:2]==source).all(-1,keepdim=True)
        preferred=torch.where(native,preferred_action[...,2:3],preferred)
    interior=preferred if interior_fusion is None else torch.full_like(preferred,float(interior_fusion))
    candidates=actuator_candidate_grid(interior, candidate_subdivisions,
        include_uniform_midpoint=include_uniform_midpoint)
    committed=packet.get('_committed_actuator_preferences') if commit_interventions else None
    if committed is not None:
        fixed=(committed>=0.).all((-1,-2))
        if bool(fixed.any()):
            # Eliminate decisions with no free fusion variables exactly.
            # Their two source policies still refresh at every control tick.
            command=committed*source+(1.-committed)*rule
            result=torch.cat((command,torch.ones_like(source_action[...,2:3])),-1)
            free=~fixed
            if bool(free.any()):
                subset={key:(value[free] if torch.is_tensor(value) and value.ndim
                    and value.shape[0]==len(source) else value) for key,value in packet.items()}
                result[free]=refine_actuator_interactions(actor,subset,source_action[free],preferred_action[free],
                    beam_width=beam_width,risk_weight=risk_weight,preference_width=preference_width,
                    selection=selection,source_preference=source_preference,prior_map=prior_map,
                    coupled_prediction=coupled_prediction,contact_distance=contact_distance,
                    synchronized_prediction=synchronized_prediction,commit_interventions=True,
                    refine_all_plans=refine_all_plans,pair_refinements=pair_refinements,
                    joint_feedback_topk=joint_feedback_topk,
                    joint_feedback_horizon=joint_feedback_horizon,
                    pareto_refinement=pareto_refinement,
                    pareto_feedback_horizons=pareto_feedback_horizons,
                    pareto_componentwise=pareto_componentwise,
                    pareto_feedback_tail=pareto_feedback_tail,
                    pareto_task_guard=pareto_task_guard,
                    coupled_horizon=coupled_horizon,
                    coupled_recovery=coupled_recovery,recovery_horizon=recovery_horizon,
                    recovery_distance=recovery_distance,interaction_order=interaction_order,
                    expand_interactions=expand_interactions,coupled_tail=coupled_tail,
                    coupled_refinements=coupled_refinements,interior_fusion=interior_fusion,
                    candidate_subdivisions=candidate_subdivisions,capture_weight=capture_weight,
                    include_uniform_midpoint=include_uniform_midpoint,
                    task_stopping=task_stopping,escape_weight=escape_weight,
                    target_envelope_degrees=target_envelope_degrees,target_envelope_weight=target_envelope_weight,
                    task_feedback_prediction=task_feedback_prediction,
                    independent_feedback_proposal=independent_feedback_proposal,
                    nominal_feedback_proposals=nominal_feedback_proposals,
                    feedback_refinement_rounds=feedback_refinement_rounds)
            return result
        # A sampled intervention fixes these coefficients for its window.
        # Optimize the remaining coefficients against those actual partner
        # maneuvers, instead of projecting away the intervention afterwards.
        candidates=torch.where((committed>=0.)[:,:,None],committed[:,:,None],candidates)
        preferred=torch.where(committed>=0.,committed,preferred)
    if task_stopping:
        if task_feedback_prediction:
            _,order=public_return_context(packet,True,2)
            b,n=order.shape
            candidates=candidates.gather(1,order[...,None,None].expand_as(candidates))
            preferred=preferred.gather(1,order[...,None].expand_as(preferred))
            unary=((candidates-preferred[:,:,None])/preference_width).square().mean(-1)
            library,reference_start=feedback_fleet_library(unary)
            dt=min(.05,actor.prediction_step) if synchronized_prediction else actor.prediction_step
            target,velocity=actor.public_target_motion(packet)
            horizon=round((actor.risk_horizon if coupled_horizon is None else coupled_horizon)/dt)
            times=torch.arange(1,horizon+1,device=source.device)*dt
            central=(torch.stack([actor.target_motion.predict(packet,float(time))[0] for time in times],1)
                if actor.target_motion is not None else target[:,None]+times[None,:,None]*velocity[:,None])
            futures,weights=target_turning_envelope(target,central,target_envelope_degrees)
            scenarios=futures.shape[1]
            learner=getattr(actor,'feedback_fleet_policy',None)
            target_response=getattr(actor,'target_feedback',None)
            if target_response is not None and scenarios!=1:
                raise ValueError('Learned reactive target motion replaces the fixed turning envelope.')
            response_components=(tuple(target_response.component(index) for index in range(len(target_response.models)))
                if target_response is not None and getattr(target_response,'scenario_mode','mean')=='members'
                else (target_response,))
            if target_response is not None:
                if hasattr(target_response,'trajectory_components'):
                    response_components=target_response.trajectory_components()
                scenarios=len(response_components)
                probabilities=getattr(target_response,'scenario_weights',None)
                weights=(source.new_full((scenarios,),1./scenarios) if probabilities is None
                    else source.new_tensor(probabilities))
                kinematic_weight=getattr(target_response,'kinematic_weight',0.)
                if kinematic_weight:
                    # Preserve distinct fleet responses instead of averaging
                    # incompatible target accelerations before simulation.
                    learned=(response_components if kinematic_weight<1. else ())
                    response_components=(*learned,target_response.kinematic_component())
                    weights=torch.cat((weights[:len(learned)]*(1.-kinematic_weight),
                        source.new_tensor([kinematic_weight])))
                    scenarios=len(response_components)
            def forecast(choices):
                count=choices.shape[1]
                plans=candidates[:,None].expand(-1,count,-1,-1,-1).gather(3,
                    choices[...,None,None].expand(-1,-1,-1,1,2)).squeeze(3)
                physical=plans.gather(2,order.argsort(-1)[:,None,:,None].expand(-1,count,-1,2))
                if target_response is not None:
                    responses=[coupled_actuator_risk(actor,packet,source,physical,'task_paths',
                        synchronized_prediction,coupled_horizon,target_response=component)
                        for component in response_components]
                    paths=torch.stack([response[0] for response in responses],2)
                    plan_futures=torch.stack([response[1] for response in responses],2)
                elif scenarios==1:
                    paths=coupled_actuator_risk(actor,packet,source,physical,'paths',
                        synchronized_prediction,coupled_horizon)[:,:,None]
                else:
                    expanded={key:(value[:,None].expand(-1,scenarios,*value.shape[1:]).reshape(
                        b*scenarios,*value.shape[1:]) if torch.is_tensor(value) and value.ndim
                        and value.shape[0]==b else value) for key,value in packet.items()}
                    expanded_source=source[:,None].expand(-1,scenarios,-1,-1).reshape(b*scenarios,n,2)
                    expanded_plans=physical[:,None].expand(-1,scenarios,-1,-1,-1).reshape(b*scenarios,count,n,2)
                    paths=coupled_actuator_risk(actor,expanded,expanded_source,expanded_plans,'paths',
                        synchronized_prediction,coupled_horizon,
                        target_paths=futures.reshape(b*scenarios,horizon,2))
                    paths=paths.reshape(b,scenarios,count,n,horizon,2).permute(0,2,1,3,4,5)
                if target_response is None:
                    plan_futures=futures[:,None].expand(-1,count,-1,-1,-1)
                paths=paths.gather(3,order[:,None,None,:,None,None].expand_as(paths))
                factors=task_envelope_fusion_factors(paths.reshape(b*count,scenarios,n,1,horizon,2),
                    plan_futures.reshape(b*count,scenarios,horizon,2),
                    ((plans-preferred[:,None])/preference_width).square().mean(-1).reshape(b*count,n,1),
                    actor.distance_risk,weights=weights,interception_weight=target_envelope_weight,
                    control_stride=max(1,round(.2/dt)))
                if learner is not None and learner.scenario_evaluation:
                    scenario_costs=task_stopped_fleet_cost(factors['scenario_factors'],
                        torch.zeros((b*count*scenarios,1,n),dtype=torch.long,device=source.device),
                        risk_weight,escape_weight).reshape(b,count,scenarios)
                    mass=factors['scenario_weights']
                    costs=(scenario_costs*mass[None,None]).sum(-1)
                    features=learner.scenario_features(packet,source,preferred,plans,paths,
                        plan_futures,mass,order,target,scenario_costs)
                else:
                    costs=task_stopped_fleet_cost(factors,torch.zeros((b*count,1,n),
                        dtype=torch.long,device=source.device),risk_weight,escape_weight).reshape(b,count)
                    features=(learner.features(packet,source,preferred,plans,paths,plan_futures,weights,order,target)
                        if learner is not None else None)
                return costs,features
            scores,features=forecast(library)
            extra_search=independent_feedback_proposal or nominal_feedback_proposals or feedback_refinement_rounds
            if extra_search:
                local_scores=scores if learner is None else -learner(features,scores)[0]
                proposal=select_feedback_fleet_plan(local_scores,library,reference_start,'single_reference')
            proposals=[]
            if independent_feedback_proposal:
                # A simultaneous independent recombination need not occur in
                # the balanced library. Complete it with an actual feedback
                # forecast, available to both selectors at the same cost.
                # Contrast normalization uses the original common library so
                # adding this proposal cannot change the local query scores.
                proposals.append(proposal[:,None])
            if nominal_feedback_proposals:
                # Retain the nominal planner's coordination basins before
                # evaluating their complete teammate feedback. Nominal costs
                # propose plans but never select the feedback action.
                physical_candidates=candidates.gather(1,
                    order.argsort(-1)[...,None,None].expand_as(candidates))
                nominal_paths=actor.candidate_risk(dict(packet),source,
                    candidate_grid=physical_candidates,return_paths=True)
                nominal_paths=nominal_paths.gather(1,order[...,None,None,None].expand_as(nominal_paths))
                nominal_times=torch.arange(1,nominal_paths.shape[-2]+1,device=source.device)*actor.prediction_step
                nominal_target=(torch.stack([actor.target_motion.predict(packet,float(t))[0] for t in nominal_times],1)
                    if actor.target_motion is not None else target[:,None]+nominal_times[None,:,None]*velocity[:,None])
                nominal_futures,nominal_weights=target_turning_envelope(target,nominal_target,target_envelope_degrees)
                identical=(candidates[:,:,:,None]-candidates[:,:,None,:]).abs().amax(-1)<1e-6
                duplicate=(identical & torch.tril(torch.ones((9,9),dtype=torch.bool,device=source.device),-1)).any(-1)
                nominal_factors=task_envelope_fusion_factors(nominal_paths,nominal_futures,
                    unary.masked_fill(duplicate,float('inf')),actor.distance_risk,
                    weights=nominal_weights,interception_weight=target_envelope_weight,
                    control_stride=max(1,round(.2/actor.prediction_step)))
                for nominal_selection in ('joint','independent','single_reference'):
                    proposals.append(select_task_stopped_fusion(nominal_factors,selection=nominal_selection,
                        beam_width=beam_width,risk_weight=risk_weight,escape_weight=escape_weight)[:,None])
            if proposals:
                extra=torch.cat(proposals,1)
                extra_scores,extra_features=forecast(extra)
                library=torch.cat((library,extra),1)
                scores=torch.cat((scores,extra_scores),1)
                if learner is not None:features=torch.cat((features,extra_features),1)
            for _ in range(feedback_refinement_rounds):
                ranked=scores if learner is None else -learner(features,scores)[0]
                current=library[torch.arange(b,device=source.device),ranked.argmin(-1)]
                local=current[:,None,None].expand(-1,n,9,-1).clone()
                for vessel in range(n):local[:,vessel,:,vessel]=torch.arange(9,device=source.device)
                extra=local.flatten(1,2)
                extra_scores,extra_features=forecast(extra)
                library=torch.cat((library,extra),1)
                scores=torch.cat((scores,extra_scores),1)
                if learner is not None:features=torch.cat((features,extra_features),1)
            if learner is None:
                choice=select_feedback_fleet_plan(scores,library,reference_start,selection)
            else:
                choice=learner.choose(features,scores,library,reference_start,selection)
            if (extra_search and selection=='single_reference'
                    and not (learner is not None and learner.sample)):
                # Retain the exact local choices that proposed this plan;
                # a common logit offset from the extra forecast must not
                # perturb floating-point tie breaking in the control arm.
                choice=proposal
            gates=candidates.gather(2,choice[...,None,None].expand(-1,-1,1,2)).squeeze(2)
            gates=gates.gather(1,order.argsort(-1)[...,None].expand_as(gates))
            physical=torch.cat((gates*source+(1.-gates)*rule,torch.ones_like(source_action[...,2:3])),-1)
            return (physical,gates) if return_coefficients else physical
        paths = actor.candidate_risk(dict(packet), source, candidate_grid=candidates, return_paths=True)
        target, velocity = actor.public_target_motion(packet)
        times = torch.arange(1, paths.shape[-2]+1, device=source.device)*actor.prediction_step
        target_path = (torch.stack([actor.target_motion.predict(packet, float(time))[0] for time in times], 1)
            if actor.target_motion is not None else target[:, None]+times[None, :, None]*velocity[:, None])
        _, order = public_return_context(packet, True, 2)
        candidates = candidates.gather(1, order[..., None, None].expand_as(candidates))
        preferred = preferred.gather(1, order[..., None].expand_as(preferred))
        paths = paths.gather(1, order[..., None, None, None].expand_as(paths))
        unary = ((candidates-preferred[:, :, None])/preference_width).square().mean(-1)
        k = candidates.shape[2]
        identical = (candidates[:, :, :, None]-candidates[:, :, None, :]).abs().amax(-1) < 1e-6
        duplicate = (identical & torch.tril(torch.ones((k, k), dtype=torch.bool,
            device=source.device), -1)).any(-1)
        unary = unary.masked_fill(duplicate, float('inf'))
        if target_envelope_weight:
            target_paths, weights = target_turning_envelope(target, target_path, target_envelope_degrees)
            factors = task_envelope_fusion_factors(paths, target_paths, unary, actor.distance_risk,
                weights=weights, interception_weight=target_envelope_weight,
                control_stride=max(1, round(.2/actor.prediction_step)))
        else:
            factors = task_stopped_fusion_factors(paths, target_path, unary, actor.distance_risk,
                control_stride=max(1, round(.2/actor.prediction_step)))
        choices = select_task_stopped_fusion(factors, selection=selection, beam_width=beam_width,
            risk_weight=risk_weight, escape_weight=escape_weight)
        gates = candidates.gather(2, choices[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        gates = gates.gather(1, order.argsort(-1)[..., None].expand_as(gates))
        return torch.cat((gates*source+(1.-gates)*rule, torch.ones_like(source_action[..., 2:3])), -1)
    if coupled_prediction:
        _,order=public_return_context(packet,True,2);b,n=order.shape
        candidates=candidates.gather(1,order[...,None,None].expand_as(candidates))
        preferred=preferred.gather(1,order[...,None].expand_as(preferred))
        design=actuator_joint_design(n,source.device,interaction_order);count=len(design)
        plans=candidates[:,None].expand(-1,count,-1,-1,-1).gather(3,
            design[None,:,:,None,None].expand(b,-1,-1,1,2)).squeeze(3)
        physical_plans=plans.gather(2,order.argsort(-1)[:,None,:,None].expand(-1,count,-1,2))
        risk,clearance=coupled_actuator_risk(actor,packet,source,physical_plans,'both',synchronized_prediction,
            coupled_horizon,tail_horizon=coupled_tail)
        score=((plans-preferred[:,None])/preference_width).square().mean(-1).sum(-1)+risk_weight*risk
        safe=clearance>=contact_distance
        # Feasible joint forecasts take priority over the soft clearance sum.
        # When no plan is feasible, maximize the worst predicted separation.
        score=score+torch.where(safe.any(-1,keepdim=True),1e6*(~safe).to(score.dtype),
            1e6*(clearance.amax(-1,keepdim=True)-clearance))
        if interaction_order==2:
            identical=(plans.flatten(2)[:,:,None]-plans.flatten(2)[:,None]).abs().amax(-1)<1e-6
            duplicate=(identical & torch.tril(torch.ones((count,count),dtype=torch.bool,device=source.device),-1)).any(-1)
        else:
            # Coalescing equal actuator candidates before sorting complete
            # plans avoids a quadratic fleet-by-plan distance tensor.
            equivalent=(candidates[:,:,:,None]-candidates[:,:,None]).abs().amax(-1)<1e-6
            labels=torch.arange(9,device=source.device)[None,None,None].expand_as(equivalent)
            canonical=labels.masked_fill(~equivalent,9).amin(-1)
            assignment=canonical[:,None].expand(-1,count,-1,-1).gather(3,
                design[None,:,:,None].expand(b,-1,-1,1)).squeeze(-1)
            codes=(assignment*(9**torch.arange(n,device=source.device))).sum(-1)
            ordered,permutation=codes.sort(dim=-1,stable=True)
            repeated=torch.cat((torch.zeros_like(ordered[:,:1],dtype=torch.bool),ordered[:,1:]==ordered[:,:-1]),-1)
            duplicate=torch.zeros_like(repeated).scatter(1,permutation,repeated)
        refining=~safe.any(-1)
        if coupled_refinements and bool(refining.any()):
            # A balanced library covers low-order effects, but its best fleet
            # plan need not be locally optimal under the actual coupled
            # feedback forecast. Refine several coordination basins, retaining
            # every evaluated complete plan for BOTH selection controls.
            seeds=score.masked_fill(duplicate,float('inf')).topk(min(4,count),largest=False).indices
            current=plans.gather(1,seeds[:,:,None,None].expand(-1,-1,n,2))
            plan_bank=[plans];risk_bank=[risk];clearance_bank=[clearance]
            for _ in range(int(coupled_refinements)):
                for vessel in range(n):
                    alternatives=current[:,:,None].expand(-1,-1,9,-1,-1).clone()
                    alternatives[:,:,:,vessel]=candidates[:,None,vessel]
                    alternatives=torch.where(refining[:,None,None,None,None],alternatives,current[:,:,None])
                    alternatives=alternatives.flatten(1,2)
                    physical_alternatives=alternatives.gather(2,
                        order.argsort(-1)[:,None,:,None].expand(-1,alternatives.shape[1],-1,2))
                    new_risk,new_clearance=coupled_actuator_risk(actor,packet,source,physical_alternatives,
                        'both',synchronized_prediction,coupled_horizon,tail_horizon=coupled_tail)
                    local=((alternatives-preferred[:,None])/preference_width).square().mean(-1).sum(-1)+risk_weight*new_risk
                    local=local.reshape(b,-1,9);separation=new_clearance.reshape(b,-1,9)
                    feasible=separation>=contact_distance
                    local+=torch.where(feasible.any(-1,keepdim=True),1e6*(~feasible).to(local.dtype),
                        1e6*(separation.amax(-1,keepdim=True)-separation))
                    selected=local.argmin(-1)
                    current=alternatives.reshape(b,-1,9,n,2).gather(2,
                        selected[:,:,None,None,None].expand(-1,-1,1,n,2)).squeeze(2)
                    plan_bank.append(alternatives);risk_bank.append(new_risk);clearance_bank.append(new_clearance)
            plans=torch.cat(plan_bank,1);risk=torch.cat(risk_bank,1);clearance=torch.cat(clearance_bank,1)
            count=plans.shape[1];safe=clearance>=contact_distance
            score=((plans-preferred[:,None])/preference_width).square().mean(-1).sum(-1)+risk_weight*risk
            score+=torch.where(safe.any(-1,keepdim=True),1e6*(~safe).to(score.dtype),
                1e6*(clearance.amax(-1,keepdim=True)-clearance))
            matches=(plans[:,:,:,None]-candidates[:,None]).abs().amax(-1)<1e-6
            assignment=matches.to(torch.long).argmax(-1)
            codes=(assignment*(9**torch.arange(n,device=source.device))).sum(-1)
            ordered,permutation=codes.sort(dim=-1,stable=True)
            repeated=torch.cat((torch.zeros_like(ordered[:,:1],dtype=torch.bool),ordered[:,1:]==ordered[:,:-1]),-1)
            duplicate=torch.zeros_like(repeated).scatter(1,permutation,repeated)
        if selection=='joint':
            choice=score.masked_fill(duplicate,float('inf')).argmin(-1)
            gates=plans[torch.arange(b,device=source.device),choice]
        elif selection=='independent':
            # Both selectors see the same jointly predicted fleet library.
            # This control chooses each vessel from its conditional marginal.
            matches=(plans[:,:,:,None]-candidates[:,None]).abs().amax(-1)<1e-6
            weights=matches & ~duplicate[:,:,None,None]
            expected=(score[:,:,None,None]*weights).sum(1)/weights.sum(1).clamp_min(1)
            choices=expected.argmin(-1)
            gates=candidates.gather(2,choices[...,None,None].expand(-1,-1,1,2)).squeeze(2)
        else:raise ValueError('Actuator candidate selection must be joint or independent.')
        gates=gates.gather(1,order.argsort(-1)[...,None].expand_as(gates))
        command=gates*source+(1.-gates)*rule
        physical=torch.cat((command,torch.ones_like(source_action[...,2:3])),-1)
        if expand_interactions and interaction_order==2:
            # Expand the complete fleet search only when the pair-balanced
            # library supplies no feasible feedback trajectory. Both joint
            # and independent selectors receive the same expanded library.
            expand=~safe.any(-1)
            if bool(expand.any()):
                subset={key:(value[expand] if torch.is_tensor(value) and value.ndim
                    and value.shape[0]==b else value) for key,value in packet.items()}
                physical[expand]=refine_actuator_interactions(actor,subset,source_action[expand],
                    preferred_action[expand],beam_width=beam_width,risk_weight=risk_weight,
                    preference_width=preference_width,selection=selection,source_preference=source_preference,
                    prior_map=prior_map,coupled_prediction=True,contact_distance=contact_distance,
                    synchronized_prediction=synchronized_prediction,commit_interventions=commit_interventions,
                    coupled_horizon=coupled_horizon,interaction_order=3,coupled_tail=coupled_tail,
                    coupled_refinements=coupled_refinements,interior_fusion=interior_fusion)
        return physical
    def interception_cost(paths):
        target, velocity = actor.public_target_motion(packet)
        times = torch.arange(1, paths.shape[-2]+1, device=source.device)*actor.prediction_step
        target_path = (torch.stack([actor.target_motion.predict(packet, float(time))[0] for time in times], 1)
            if actor.target_motion is not None else target[:, None]+times[None, :, None]*velocity[:, None])
        return ((paths-target_path[:, None, None]).norm(dim=-1).amin(-1)-5.).clamp_min(0.)/5.

    if selection=='single_reference':
        # Only the subject's candidate varies. Each neighbor follows its
        # single source-preferred trajectory; no neighbor candidate pair or
        # joint optimizer enters this comparison.
        paths=actor.candidate_risk(dict(packet),source,candidate_grid=candidates,return_paths=True)
        reference_index=candidate_subdivisions*(2*candidate_subdivisions+1+include_uniform_midpoint)+candidate_subdivisions
        reference=paths[:,:,reference_index]
        separation=(paths[:,:,None]-reference[:,None,:,None]).norm(dim=-1)
        risk=actor.distance_risk(separation.amin(-1))
        mask=~torch.eye(source.shape[1],dtype=torch.bool,device=source.device)
        risk=(risk*mask[None,:,:,None]).sum(2)
        unary=((candidates-preferred[:,:,None])/preference_width).square().mean(-1)
        if capture_weight:
            reach = interception_cost(paths)
            index = reference_index
            others = reach[:, :, index][:, None].expand(-1, source.shape[1], -1)
            others = others.masked_fill(~mask[None], float('inf')).amin(-1)
            unary = unary+capture_weight*torch.minimum(reach, others[..., None])
        choice=(unary+risk_weight*risk).argmin(-1)
        gates=candidates.gather(2,choice[...,None,None].expand(-1,-1,1,2)).squeeze(2)
        return torch.cat((gates*source+(1.-gates)*rule,torch.ones_like(source_action[...,2:3])),-1)
    forecast_packet = dict(packet)
    if capture_weight or candidate_output is not None:
        forecast_packet['_retain_candidate_paths'] = True
    risk=actor.candidate_risk(forecast_packet,source,pairwise=True,candidate_grid=candidates)
    reach = interception_cost(forecast_packet['_retained_candidate_paths']) if capture_weight else None
    _,order=public_return_context(packet,True,2)
    batch=torch.arange(len(source),device=source.device)
    candidates=candidates.gather(1,order[...,None,None].expand_as(candidates))
    preferred=preferred.gather(1,order[...,None].expand_as(preferred))
    if reach is not None:
        reach = reach.gather(1, order[..., None].expand_as(reach))
    risk=risk[batch[:,None,None],order[:,:,None],order[:,None,:]]
    unary=((candidates-preferred[:,:,None])/preference_width).square().mean(-1)
    b,n,k=unary.shape
    identical=(candidates[:,:,:,None]-candidates[:,:,None,:]).abs().amax(-1)<1e-6
    duplicate=(identical & torch.tril(torch.ones((k,k),dtype=torch.bool,device=source.device),-1)).any(-1)
    unary=unary.masked_fill(duplicate,float('inf'))
    if selection=='independent':
        probability=(-unary).softmax(-1)
        mask=~torch.eye(n,dtype=torch.bool,device=source.device)
        expected=torch.einsum('bijgh,bjh,ij->big',risk,probability,mask.to(source.dtype))
        cost = unary+risk_weight*expected
        if capture_weight:
            cost = cost+capture_weight*independent_fleet_minimum_cost(reach, probability)
        choices=cost.argmin(-1)
    elif selection=='joint':
        scores=source.new_zeros((b,1));plans=torch.empty((b,1,0),dtype=torch.long,device=source.device)
        pursuit = source.new_zeros((b, 1))
        for vessel in range(n):
            cost=scores[:,:,None]+unary[:,vessel,None]
            for other in range(vessel):
                pair=risk[:,other,vessel]
                cost=cost+risk_weight*pair.gather(1,plans[:,:,other,None].expand(-1,-1,k))
            if capture_weight:
                next_pursuit = reach[:, vessel, None].expand(-1, plans.shape[1], -1)
                if vessel:
                    next_pursuit = torch.minimum(pursuit[..., None], next_pursuit)
                cost = cost+capture_weight*(next_pursuit-(pursuit[..., None] if vessel else 0.))
            flat=cost.flatten(1);length=min(beam_width,flat.shape[-1])
            scores,index=flat.topk(length,largest=False)
            if capture_weight:
                pursuit = next_pursuit.flatten(1).gather(1, index)
            ancestor=index//k
            plans=torch.cat((plans.gather(1,ancestor[...,None].expand(-1,-1,vessel)),(index%k)[...,None]),-1)
        if candidate_output is not None:
            candidate_output['beam_plans']=plans.clone()
        if refine_all_plans:
            # Joint fusion is nonconvex. Refining only the cheapest beam
            # endpoint discards the other coordination basins already found.
            # Retain that endpoint and refine every distinct beam plan under
            # the SAME candidate-interaction objective before choosing one.
            proposal_plans = plans.clone() if joint_feedback_topk else None
            if pareto_refinement:
                if joint_feedback_topk:
                    raise ValueError('Pareto refinement compares the same factorized risk objective.')
                reference = plans[:, 0].clone()
                for _ in range(3):
                    for vessel in range(n):
                        cost = unary[:, vessel].clone()
                        for other in range(n):
                            if other == vessel:
                                continue
                            cost += risk_weight*risk[:, vessel, other].gather(
                                2, reference[:, other, None, None].expand(-1, k, 1)).squeeze(-1)
                        reference[:, vessel] = cost.argmin(-1)
            for _ in range(3):
                for vessel in range(n):
                    cost=unary[:,vessel,None].expand(-1,plans.shape[1],-1).clone()
                    for other in range(n):
                        if other==vessel:continue
                        cost+=risk_weight*risk[:,vessel,other].gather(2,
                            plans[:,None,:,other].expand(-1,k,-1)).transpose(1,2)
                    if capture_weight:
                        peer_reach = torch.stack([reach[:, other].gather(1, plans[:, :, other])
                            for other in range(n) if other != vessel], -1).amin(-1)
                        cost += capture_weight*torch.minimum(reach[:, vessel, None], peer_reach[..., None])
                    plans[:,:,vessel]=cost.argmin(-1)
            if pair_refinements:
                plans = refine_pair_blocks(unary, risk, plans, risk_weight, pair_refinements)
            energy=source.new_zeros(plans.shape[:2])
            for vessel in range(n):
                energy+=unary[:,vessel].gather(1,plans[:,:,vessel])
                for other in range(vessel):
                    code=plans[:,:,other]*k+plans[:,:,vessel]
                    energy+=risk_weight*risk[:,other,vessel].flatten(1).gather(1,code)
            if capture_weight:
                energy += capture_weight*torch.stack([reach[:, vessel].gather(1, plans[:, :, vessel])
                    for vessel in range(n)], -1).amin(-1)
            if joint_feedback_topk:
                # Factorized paths are inexpensive proposals. Re-evaluate the
                # most promising distinct maneuvers with all boats responding
                # to the same evolving fleet state and source-policy feedback.
                shortlist = distinct_fleet_shortlist(torch.cat((proposal_plans, plans), 1),
                    torch.cat((scores, energy), 1), joint_feedback_topk)
                proposed = candidates[:, None].expand(-1, shortlist.shape[1], -1, -1, -1).gather(
                    3, shortlist[..., None, None].expand(-1, -1, -1, 1, 2)).squeeze(3)
                physical_proposals = proposed.gather(2, order.argsort(-1)[:, None, :, None].expand(
                    -1, shortlist.shape[1], -1, 2))
                feedback_risk = coupled_actuator_risk(actor, packet, source, physical_proposals,
                    synchronized=True, horizon=joint_feedback_horizon)
                feedback_score = (((proposed-preferred[:, None])/preference_width).square().mean(-1).sum(-1)
                                  +risk_weight*feedback_risk)
                choices = shortlist[torch.arange(b, device=source.device), feedback_score.argmin(-1)]
            elif pareto_refinement:
                choices = pareto_fleet_refinement(
                    unary, risk, plans, reference, risk_weight, componentwise=pareto_componentwise)
                if pareto_feedback_horizons:
                    choices = feedback_guarded_refinement(
                        actor, packet, source, candidates, order, reference, choices, pareto_feedback_horizons,
                        tail_horizon=pareto_feedback_tail, task_guard=pareto_task_guard)
            else:
                choices=plans[torch.arange(b,device=source.device),energy.argmin(-1)]
        else:
            choices=plans[:,0].clone()
            for _ in range(3):
                for vessel in range(n):
                    cost=unary[:,vessel].clone()
                    for other in range(n):
                        if other==vessel:continue
                        cost+=risk_weight*risk[:,vessel,other].gather(2,
                            choices[:,other,None,None].expand(-1,k,1)).squeeze(-1)
                    if capture_weight:
                        peer_reach = torch.stack([reach[:, other].gather(1, choices[:, other, None]).squeeze(1)
                            for other in range(n) if other != vessel], -1).amin(-1)
                        cost += capture_weight*torch.minimum(reach[:, vessel], peer_reach[:, None])
                    choices[:,vessel]=cost.argmin(-1)
            if pair_refinements:
                choices = refine_pair_blocks(
                    unary, risk, choices[:, None], risk_weight, pair_refinements)[:, 0]
    else:raise ValueError('Actuator candidate selection must be joint or independent.')
    if candidate_output is not None:
        candidate_output.update(candidates=candidates,unary=unary,pair_risk=risk,
            order=order,reference=choices,
            refined_plans=(plans if selection=='joint' else choices[:,None]),
            prediction_paths=forecast_packet['_retained_candidate_paths'],
            source=source,rule=rule)
    gates=candidates.gather(2,choices[...,None,None].expand(-1,-1,1,2)).squeeze(2)
    gates=gates.gather(1,order.argsort(-1)[...,None].expand_as(gates))
    command=gates*source+(1.-gates)*rule
    physical=torch.cat((command,torch.ones_like(source_action[...,2:3])),-1)
    if coupled_recovery:
        # A long factorized forecast that still predicts contact after joint
        # optimization supplies no feasible continuation. Replan that state
        # with a shorter, mutually consistent feedback forecast.
        selected_risk=torch.stack([risk[batch,i,j,choices[:,i],choices[:,j]]
            for i in range(n) for j in range(i+1,n)],-1).amax(-1)
        recover=selected_risk>actor.distance_risk(source.new_tensor(5.))
        if recovery_distance is not None:
            # A short feedback forecast resolves an imminent encounter. It
            # cannot replace the long pursuit forecast merely because a
            # distant, uncertain trajectory pair crosses later in time.
            position=packet['motion'][...,:2]
            separation=(position[:,:,None]-position[:,None,:]).norm(dim=-1)
            separation=separation.masked_fill(torch.eye(n,dtype=torch.bool,device=source.device)[None],float('inf'))
            recover&=separation.flatten(1).amin(-1)<recovery_distance
        if bool(recover.any()):
            subset={key:(value[recover] if torch.is_tensor(value) and value.ndim and value.shape[0]==b else value)
                for key,value in packet.items()}
            physical[recover]=refine_actuator_interactions(actor,subset,source_action[recover],
                preferred_action[recover],beam_width=beam_width,risk_weight=risk_weight,
                preference_width=preference_width,selection=selection,source_preference=source_preference,
                prior_map=prior_map,coupled_prediction=True,contact_distance=contact_distance,
                synchronized_prediction=True,commit_interventions=commit_interventions,
                coupled_horizon=recovery_horizon,interaction_order=interaction_order,
                expand_interactions=expand_interactions,coupled_tail=coupled_tail,
                coupled_refinements=coupled_refinements,interior_fusion=interior_fusion)
    return physical


class ResponseFeedbackValue(nn.Module):
    """Receding joint actuator fusion conditioned on observed motion responses.

    Stored controls are convex coefficients, never alternative source-policy
    commands. Each tick refreshes both original commands before decoding the
    selected joint response. Unsupported encounters retain their predecessor.
    """
    def __init__(self, contexts, gates, max_distance=.15, response_gain=None,
                 response_horizon=None, feedback_strength=1., maximum_correction=.15,
                 water_relative=False, target_relative=False, response_operator=None,
                 response_endpoint=None, independent_response_gain=None, joint_response=True,
                 state_response=False, response_velocity=None, viability_lookahead=0.,
                 feedback_objective='tracking', learning_objective=None, prior=None,
                 minimum_clearance=0.):
        super().__init__()
        self.register_buffer('contexts', torch.as_tensor(contexts, dtype=torch.float32))
        self.register_buffer('gates', torch.as_tensor(gates, dtype=torch.float32))
        self.register_buffer('response_gain', None if response_gain is None else
                             torch.as_tensor(response_gain, dtype=torch.float32))
        self.register_buffer('response_horizon', None if response_horizon is None else
                             torch.as_tensor(response_horizon, dtype=torch.float32))
        self.register_buffer('response_operator', None if response_operator is None else
                             torch.as_tensor(response_operator, dtype=torch.float32))
        self.register_buffer('response_endpoint', None if response_endpoint is None else
                             torch.as_tensor(response_endpoint, dtype=torch.float32))
        self.register_buffer('response_velocity', None if response_velocity is None else
                             torch.as_tensor(response_velocity, dtype=torch.float32))
        self.register_buffer('independent_response_gain', None if independent_response_gain is None else
                             torch.as_tensor(independent_response_gain, dtype=torch.float32))
        self.max_distance = max_distance
        self.feedback_strength = feedback_strength
        self.maximum_correction = maximum_correction
        self.water_relative = water_relative
        self.target_relative = target_relative
        self.feedback_objective = feedback_objective
        self.joint_response = joint_response
        self.state_response = state_response
        self.viability_lookahead = viability_lookahead
        self.minimum_clearance = minimum_clearance
        self.prior = None if prior is None else ResponseFeedbackValue(**prior)

    def forward(self, packet, action):
        fallback = action if self.prior is None else self.prior(packet, action)
        context, order = response_feedback_context(packet, self.water_relative)
        if context.shape[-1] != self.contexts.shape[-1]:
            return fallback
        distance = (context[:, None]-self.contexts[None]).square().sum(-1)
        # A learned finite response window has a physical clock. A nearby
        # later pose does not authorize extending that intervention forever.
        distance = distance.masked_fill(
            (context[:, None, -1]-self.contexts[None, :, -1]).abs() > .001, float('inf'))
        closest, index = distance.min(-1)
        supported = closest <= self.max_distance**2
        if not bool(supported.any()):
            return fallback
        gates = self.gates[index]
        if self.response_gain is not None:
            n = action.shape[1]
            residual = (context-self.contexts[index])[:, :8*n].reshape(-1, n, 8)
            velocity = 3.*residual[..., 2:4]
            if self.target_relative:
                velocity = velocity-5.*(context-self.contexts[index])[:, None, -3:-1]
            horizon = self.response_horizon[index, None, None]
            innovation = 30.*residual[..., :2]+horizon*velocity
            if self.state_response:
                innovation = torch.cat((innovation, horizon*velocity,
                    2.*residual[..., 4:6], 2.*horizon*residual[..., 6:7]), -1)
            if self.feedback_objective == 'response_value':
                if self.response_operator is None or self.independent_response_gain is None:
                    raise ValueError('Conditional response values require joint and independent response operators.')
                # M2 supplies the learned response-value surface. M1 solves
                # its coupled decision problem; its removal makes each boat
                # optimize once with its teammates fixed at their current
                # proposals. Neither arm directly executes the stored plan.
                decision = action[..., 2:3].expand_as(gates).gather(
                    1, order[..., None].expand(-1, -1, 2)).clone()
                inverse = (self.response_gain if self.joint_response else self.independent_response_gain)[index]
                for _ in range(3 if self.joint_response else 1):
                    error = (self.response_operator[index]@(decision-gates).flatten(1)[..., None]
                             +innovation.flatten(1)[..., None])
                    adjustment = -(inverse@error).reshape(-1, n, 2)
                    decision = (decision+self.feedback_strength*adjustment).clamp(0., 1.)
                correction = decision-gates
            elif self.feedback_objective == 'viability':
                if self.response_operator is None or self.response_endpoint is None:
                    raise ValueError('Response viability requires fitted conditional effects and endpoints.')
                operator = self.response_operator[index]
                reference = self.response_endpoint[index]
                reference_velocity = (self.response_velocity[index] if self.response_velocity is not None
                                      else torch.zeros_like(reference))
                correction = torch.zeros_like(gates)
                pairs = torch.triu_indices(n, n, 1, device=action.device)
                left, right = pairs
                def approach(position, velocity):
                    time = (-(position*velocity).sum(-1)/velocity.square().sum(-1).clamp_min(1e-6))
                    time = time.clamp(0., self.viability_lookahead)
                    return position+time[..., None]*velocity, time
                baseline_relative, _ = approach(reference[:, left]-reference[:, right],
                    reference_velocity[:, left]-reference_velocity[:, right])
                baseline = baseline_relative.norm(dim=-1)
                clearance = baseline.clamp_max(5.2).clamp_min(self.minimum_clearance)
                interceptor = reference.norm(dim=-1).argmin(-1)
                batch = torch.arange(len(action), device=action.device)
                capture_distance = reference[batch, interceptor].norm(dim=-1)
                width = 7 if self.state_response else 2
                response_jacobian = operator.reshape(-1, n, width, 2*n)
                jacobian = response_jacobian[..., :2, :]
                velocity_jacobian = (response_jacobian[..., 2:4, :]/horizon[..., None]
                    if self.state_response else torch.zeros_like(jacobian))
                identity = torch.eye(2*n, device=action.device, dtype=action.dtype)
                for _ in range(3):
                    effect = (operator@correction.flatten(1)[..., None]).reshape(-1, n, width)
                    forecast = reference+innovation[..., :2]+effect[..., :2]
                    forecast_velocity = reference_velocity+(innovation[..., 2:4]+effect[..., 2:4])/horizon if self.state_response else reference_velocity
                    relative, encounter_time = approach(forecast[:, left]-forecast[:, right],
                        forecast_velocity[:, left]-forecast_velocity[:, right])
                    distance = relative.norm(dim=-1).clamp_min(1e-6)
                    normal = relative/distance[..., None]
                    encounter_jacobian = (jacobian[:, left]-jacobian[:, right]
                        +encounter_time[..., None, None]*(velocity_jacobian[:, left]-velocity_jacobian[:, right]))
                    derivative = (normal[..., None]*encounter_jacobian).sum(-2)
                    deficit = (clearance-distance).clamp_min(0.)
                    active = deficit > 0.
                    target = forecast[batch, interceptor]
                    target_distance = target.norm(dim=-1).clamp_min(1e-6)
                    target_derivative = -(target[..., None]*jacobian[batch, interceptor]).sum(-2)/target_distance[:, None]
                    target_deficit = (target_distance-capture_distance-.05).clamp_min(0.)
                    target_active = (capture_distance < 7.) & (target_deficit > 0.)
                    derivative = torch.cat((derivative, target_derivative[:, None]), 1)
                    deficit = torch.cat((deficit, target_deficit[:, None]), 1)
                    active = torch.cat((active, target_active[:, None]), 1)
                    derivative = derivative*active[..., None]
                    normal_matrix = derivative.transpose(1, 2)@derivative+.01*identity
                    rhs = (derivative.transpose(1, 2)@(deficit*active)[..., None]).squeeze(-1)
                    step = torch.linalg.solve(normal_matrix, rhs).reshape(-1, n, 2)
                    correction = (correction+self.feedback_strength*step).clamp(
                        -self.maximum_correction, self.maximum_correction)
                    correction = (gates+correction).clamp(0., 1.)-gates
            else:
                correction = -(self.response_gain[index]@innovation.flatten(1)[..., None])
                correction = (self.feedback_strength*correction.reshape(-1, n, 2)).clamp(
                    -self.maximum_correction, self.maximum_correction)
            gates = (gates+correction).clamp(0., 1.)
        gates = gates.gather(1, order.argsort(-1)[..., None].expand(-1, -1, 2))
        rule = packet['node'][..., 11:13]
        command = gates*action[..., :2] + (1.-gates)*rule
        decoded = torch.cat((command, torch.ones_like(action[..., 2:3])), -1)
        return torch.where(supported[:, None, None], decoded, fallback)


@torch.no_grad()
def collect_response_feedback(actor, trajectories, horizon=16, perturbation=.2, device='cpu',root_stride=1):
    """Measure single-vessel response interventions on paired real futures.

    A trajectory contains saved simulator states, observations, RNG states and
    its measured fusion coefficients. All branches refresh the
    frozen source commands. Sampling stops the whole paired group at its first
    terminal event, so every response has the same valid physical horizon.
    """
    import copy
    import numpy as np
    original_device = next(actor.parameters()).device
    actor.to(device); records = []; random_state = np.random.get_state()
    try:
        for trajectory in trajectories:
            n = len(trajectory[0]['gates'])
            offsets = np.concatenate((np.zeros((1, 2*n)), perturbation*np.eye(2*n),
                                      -perturbation*np.eye(2*n))).reshape(-1, n, 2)
            for tick in range(0,len(trajectory),root_stride):
                root=trajectory[tick]
                environments = [copy.deepcopy(root['environment']) for _ in offsets]
                observations = [root['observation'].copy() for _ in offsets]
                states = [root['random_state'] for _ in offsets]
                sequence = []; terminals = None
                for step in range(min(horizon, len(trajectory)-tick)):
                    source = actor.proposal_distribution(torch.as_tensor(np.stack(observations),
                        dtype=torch.float32, device=device))[0].tanh().cpu().numpy()
                    reference = trajectory[tick+step]['gates']
                    gates = np.clip(reference[None]+offsets, 0., 1.)
                    terminals = []
                    for i, env in enumerate(environments):
                        np.random.set_state(states[i])
                        rule = (env.boids_actions-250.)/750.
                        command = gates[i]*source[i]+(1.-gates[i])*rule
                        action = np.column_stack((command, np.ones(n))).astype(np.float32)
                        observations[i], _, terminal, _ = env.step(action, 'AdaRes')
                        states[i] = np.random.get_state(); terminals.append(terminal)
                    sequence.append(reference)
                    if any(terminals):
                        break
                records.append(dict(packet=root['packet'], gates=root['gates'],
                    horizon=len(sequence), offsets=offsets,
                    positions=np.asarray([[[boat.x, boat.y] for boat in env.defender_list]
                                          for env in environments]),
                    attacker_positions=np.asarray([env.attacker.pos.copy() for env in environments]),
                    defender_motion=np.asarray([[[boat.x, boat.y, boat.theta, *boat.velocity_r]
                                                 for boat in env.defender_list] for env in environments]),
                    attacker_velocity=np.asarray([env.attacker.vel.copy() for env in environments]),
                    terminal_codes=terminals, sequence=np.asarray(sequence),
                    scene=root.get('scene'), future=root.get('future')))
                if len(records) % 16 == 0:
                    print(dict(phase='paired_response_sampling', roots=len(records), horizon=horizon), flush=True)
    finally:
        np.random.set_state(random_state); actor.to(original_device)
    return records


def fit_response_feedback(records, ridge=.02, response_weight=1., max_distance=.15,
                          water_relative=True, target_relative=False, context_bandwidth=.3,
                          state_response=False, pooled_reference=False):
    """Learn a local conditional response value from paired physical branches.

    Each branch changes one vessel's actuator fusion, sharing the initial
    state, continuation clock and future draws with its reference. The labels
    are joint position responses, not duplicated terminal-return differences.
    A quadratic value around successful responses gives a receding correction
    through the fitted, generally nonseparable response operator.
    """
    import numpy as np
    contexts, controls, horizons, operators, metrics, examples, endpoints, velocities = [], [], [], [], [], [], [], []
    for record in records:
        packet = {key: torch.as_tensor(value[None]) for key, value in record['packet'].items()}
        context, order = response_feedback_context(packet, water_relative)
        order = order[0].numpy(); n = len(order)
        h = record['packet']['motion'][0, 2]
        target = record['packet']['motion'][0, :2] + np.array(
            [[np.cos(h), -np.sin(h)], [np.sin(h), np.cos(h)]]) @ (30.*record['packet']['node'][0, 2:4])
        axis = -target/max(np.linalg.norm(target), 1.)
        rotation = np.array([[axis[0], axis[1]], [-axis[1], axis[0]]])
        positions = np.asarray(record['positions'])
        if target_relative:
            positions = positions-np.asarray(record['attacker_positions'])[:, None]
        outputs = positions@rotation.T
        if state_response:
            motion = np.asarray(record['defender_motion'])
            velocity = motion[..., 3:5]-np.asarray(record['attacker_velocity'])[:, None]
            heading = np.stack((np.cos(motion[..., 2]), np.sin(motion[..., 2])), -1)
            time = .2*record['horizon']
            outputs = np.concatenate((outputs, time*(velocity@rotation.T),
                                      2.*(heading@rotation.T), 2.*time*motion[..., 5:6]), -1)
        displacement = outputs[1:]-outputs[0]
        response = displacement[:, order].reshape(len(displacement), -1)
        offsets = np.asarray(record['offsets'])
        sequence = np.asarray(record['sequence'])
        exposure = np.clip(sequence[None]+offsets[:, None], 0., 1.)-sequence[None]
        weight = np.arange(record['horizon'], 0, -1, dtype=float)
        exposure = (exposure*weight[None, :, None, None]).sum(1)/weight.sum()
        change = exposure[1:, order].reshape(len(displacement), -1)
        # The zero-intervention response is exactly zero. No fitted factual
        # intercept can absorb the conditional action effect.
        operator = np.linalg.solve(change.T@change+1e-5*np.eye(2*n), change.T@response).T
        endpoint = np.asarray(record['positions'])[0, order]
        width = 7 if state_response else 2
        metric = .1*np.eye(width*n)
        for i in range(n):
            for j in range(i+1, n):
                relative = endpoint[i]-endpoint[j]
                distance = np.linalg.norm(relative)
                normal = rotation @ (relative/max(distance, 1e-8))
                direction = np.zeros(width*n); direction[width*i:width*i+2] = normal
                direction[width*j:width*j+2] = -normal
                metric += response_weight*np.exp(-max(distance-5., 0.))*np.outer(direction, direction)
        contexts.append(context[0].numpy()); controls.append(record['gates'][order])
        operators.append(operator); metrics.append(metric); examples.append((change, response))
        endpoints.append(positions[0, order]@rotation.T)
        if state_response:
            velocities.append(velocity[0, order]@rotation.T)
        horizons.append(.2*record['horizon'])
    contexts = np.asarray(contexts); controls = np.asarray(controls)
    operators = np.asarray(operators); horizons = np.asarray(horizons)
    distance = np.square(contexts[:, None]-contexts[None]).sum(-1)
    # Share conditional effects across nearby observed responses. Identical
    # initial contexts from several futures are repeated noise measurements,
    # not four competing lookup entries whose first tie wins.
    weights = np.exp(-distance/(2.*context_bandwidth**2))
    weights *= np.abs(contexts[:, None, -1]-contexts[None, :, -1]) < .001
    weights *= np.abs(horizons[:, None]-horizons[None]) < 1e-6
    weights *= np.exp(-np.square(controls[:, None]-controls[None]).sum((-1, -2))/.25)
    weights /= weights.sum(1, keepdims=True)
    pooled = np.einsum('ij,jkl->ikl', weights, operators)
    if pooled_reference:
        if not target_relative or not state_response:
            raise ValueError('Pooling factual motion requires relative positions and velocities.')
        # Average conditional motion increments, not absolute endpoint poses.
        # Identical public roots can have several future disturbances. The
        # previous first-neighbor endpoint selected one of those disturbances
        # even though the response operator was fitted over all of them.
        rows=contexts[:,:8*controls.shape[1]].reshape(len(contexts),-1,8)
        position=30.*rows[...,:2]
        velocity=3.*rows[...,2:4]-5.*contexts[:,None,-3:-1]
        drift=np.asarray(endpoints)-position-horizons[:,None,None]*velocity
        acceleration=np.asarray(velocities)-velocity
        endpoints=position+horizons[:,None,None]*velocity+np.einsum('ij,jkl->ikl',weights,drift)
        velocities=velocity+np.einsum('ij,jkl->ikl',weights,acceleration)
    gains, independent_gains, errors = [], [], []
    for operator, metric, (change, response) in zip(pooled, metrics, examples):
        normal = operator.T@metric@operator+ridge*np.eye(operator.shape[1])
        right = operator.T@metric
        gain = np.linalg.solve(normal, right)
        independent = np.zeros_like(gain)
        for vessel in range(operator.shape[1]//2):
            indices = slice(2*vessel, 2*vessel+2)
            independent[indices] = np.linalg.solve(normal[indices, indices], right[indices])
        independent_gains.append(independent)
        gains.append(gain); errors.append(float(np.square(change@operator.T-response).mean()))
    model = dict(contexts=contexts, gates=controls,
        response_gain=np.asarray(gains), response_horizon=np.asarray(horizons),
        response_operator=pooled, response_endpoint=np.asarray(endpoints),
        response_velocity=np.asarray(velocities) if len(velocities) else None,
        independent_response_gain=np.asarray(independent_gains),
        max_distance=max_distance, water_relative=water_relative, target_relative=target_relative,
        state_response=state_response,
        learning_objective='paired physical-response conditional value')
    return model, dict(roots=len(records), paired_branches=sum(len(r['offsets'])-1 for r in records),
        response_fit_mse=float(np.mean(errors)), terminal_difference_labels=0,
        context_bandwidth=context_bandwidth,
        pooled_reference=pooled_reference,
        mean_effective_response_roots=float(np.mean(1./np.square(weights).sum(1))))


class ContinuousProgramValue(nn.Module):
    """Select supported timed programs without quantizing their coefficients."""
    def __init__(self,contexts,programs,roots,values,variances,bandwidth=.03,margin=.05,
                 confidence=0.,learning_objective=None):
        super().__init__()
        for name,value,dtype in (('contexts',contexts,torch.float32),('programs',programs,torch.float32),
                ('roots',roots,torch.long),('values',values,torch.float32),('variances',variances,torch.float32)):
            self.register_buffer(name,torch.as_tensor(value,dtype=dtype))
        self.bandwidth=bandwidth;self.margin=margin;self.confidence=confidence

    def forward(self,packet,action):
        context,order=public_return_context(packet,True,5,action)
        if context.shape[-1]!=self.contexts.shape[-1]:return action
        distance=(context[:,None]-self.contexts[None]).square().sum(-1)
        support=torch.exp(-distance/(2.*self.bandwidth**2))[:,self.roots]
        gain=support*(self.values-self.confidence*self.variances.sqrt())[None]
        best,selected=gain.max(-1);accept=best>self.margin
        if not bool(accept.any()):return action
        program=self.programs[selected].gather(1,order.argsort(-1)[...,None].expand(-1,-1,4))
        replacement=torch.cat((action[...,:3],program[...,:1],program[...,2:3],program[...,1:2],
            torch.zeros_like(action[...,:2]),program[...,3:4]),-1)
        if action.shape[-1]==3:
            free=torch.full_like(action[...,:1],-1.)
            action=torch.cat((action,free,torch.ones_like(free),free),-1)
        if action.shape[-1]<9:
            # A legacy all-feedback instruction redecides on the next tick.
            # Padding another batch row to an explicit program must not turn
            # this rejected intervention into a long, all-feedback commitment.
            holds=(action[...,[3,5]]>=0.).any(-1).any(-1)|(action[...,5]==-2.).all(-1)
            duration=torch.where(holds[:,None,None],action[...,4:5],torch.ones_like(action[...,4:5]))
            action=torch.cat((action[...,:4],duration,action[...,5:]),-1)
            boundary=torch.floor(action[...,4:5]/2.)
            offsets=action[...,6:8] if action.shape[-1]>=8 else torch.zeros_like(action[...,:2])
            action=torch.cat((action[...,:6],offsets,boundary),-1)
        return torch.where(accept[:,None,None],replacement,action)


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
        if context_version>=3 and effect_locality:
            raise ValueError('Control-conditioned values require matching all predecessor-control coordinates.')
        self.program_model=None if program_model is None else load_program_potential(program_model)
        self.register_buffer('program_options',None if program_options is None else
            torch.as_tensor(program_options,dtype=torch.float32))
        self.program_radix=4 if program_options is None else len(program_options)
        self.prefix_refinement=prefix_refinement
        if prefix_refinement and (program_options is None or decision_tolerance or blending_penalty):
            raise ValueError('Prefix refinements use relative whole-plan options and their paired task values.')
        if program_options is not None and effect_locality:
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
        nearest,index,mapping=match_fleet_context(context,self.contexts,context_version=self.context_version)
        weights=torch.exp(-nearest/(2.*self.bandwidth**2))
        if action.shape[-1]==3:
            free=torch.full_like(action[...,:1],-1.)
            action=torch.cat((action,free,torch.full_like(free,self.span),free),-1)
        result=action.clone();best_gain=context.new_full((b,),-1.)
        if (weights.sum(-1)<=self.margin).all():
            return (result,best_gain) if return_gain else result
        m=mapping.shape[-1]
        ids=self.supported_program_codes
        programs=((ids[:,None]//(self.program_radix**torch.arange(m,device=action.device)))%self.program_radix).long()
        factual=(context.new_zeros(b) if self.paired_advantage else
            self.anchor(context,set_context=True,context_version=self.context_version)[0])
        powers=self.program_radix**torch.arange(n,device=action.device)
        for row in range(b):
            candidate_codes=[];candidate_mass=[];candidate_value=[];candidate_wait=[];candidate_second=[]
            for j,source in enumerate(index[row]):
                if weights[row,j]<1e-8:continue
                roles=mapping[row,j];missing=roles<0
                available=self.observed[source,ids].clone()
                if missing.any():available&=(programs[:,missing]==0).all(-1)
                chosen=programs[available]
                if not len(chosen):continue
                assigned=torch.zeros((len(chosen),n),dtype=torch.long,device=action.device)
                assigned[:,roles[~missing]]=chosen[:,~missing]
                value=self.values[source,ids[available]]
                if self.conditional_strata:
                    value=(1.-factual[row])*value[...,0]+factual[row]*value[...,1]
                mass=weights[row,j].expand(len(chosen))
                candidate_codes.append((assigned*powers).sum(-1))
                candidate_mass.append(mass);candidate_value.append(mass*value)
                candidate_wait.append(mass*self.wait_values[source])
                if self.confidence:candidate_second.append(mass*(self.gain_variance[source,ids[available]]+value.square()))
            if not candidate_codes:continue
            codes,inverse=torch.cat(candidate_codes).unique(return_inverse=True)
            mass=context.new_zeros(len(codes)).scatter_add_(0,inverse,torch.cat(candidate_mass))
            values=context.new_zeros(len(codes)).scatter_add_(0,inverse,torch.cat(candidate_value))
            waits=context.new_zeros(len(codes)).scatter_add_(0,inverse,torch.cat(candidate_wait))
            transferred=(codes[:,None]//powers)%self.program_radix
            if self.prefix_refinement:
                baseline=torch.zeros(n,dtype=torch.long,device=action.device)
            elif self.program_options is not None:
                pair=action[row][:,[3,5]]
                pair=torch.where((action[row,:,3]<0.)[:,None],torch.full_like(pair,-1.),pair)
                baseline=(pair[:,None]-self.program_options[None]).square().sum(-1).argmin(-1)[order[row]]
            else:baseline=torch.where(action[row,:,3]>=0.,1.+2.*action[row,:,3],0.).long()[order[row]]
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
                if self.prefix_refinement:
                    # Role matching transports the edited vessel, not a new
                    # clock or a different teammate's committed suffix.
                    physical=torch.where(selected==0,action[row,:,3],self.program_options[selected,0])
                    second=self.program_options[selected,1]
                    result[row,:,3]=physical
                    result[row,:,5]=torch.where(second==-2.,action[row,:,5],second)
                    continue
                if self.program_options is None:
                    physical=torch.where(selected>0,(selected.to(action.dtype)-1.)*.5,-1.)
                    second=torch.where((selected>0)&self.reversal,1.-physical,physical)
                else:physical,second=self.program_options[selected].unbind(-1)
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
