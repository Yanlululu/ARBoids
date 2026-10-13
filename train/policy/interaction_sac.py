"""Interaction-conditioned residual SAC and the matched legacy continuation.

Policy inputs are public measurements. The centralized dynamics state is used
only by the critics. One replay row is one complete team decision.
"""
import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from policy.networks import ActorAdap


PACKET_KEYS = ('obs', 'motion', 'central')
CONTROL_KEYS = ('prior', 'composition', 'memory_nodes', 'memory_global')
MEMORY_GLOBAL_DIM = 75


def bootstrap_value(q1, q2, estimator='min'):
    if estimator == 'mean':
        return .5 * (q1 + q2)
    if estimator == 'min':
        return torch.minimum(q1, q2)
    raise ValueError('Unknown bootstrap estimator.')


def tensor_packet(packet, device='cpu'):
    return {k: torch.as_tensor(packet[k], dtype=torch.float32, device=device)
            for k in (*PACKET_KEYS, *CONTROL_KEYS) if k in packet}


def actor_kwargs(config):
    interaction = config['interaction']
    return dict(peer_candidates=interaction.get('peer_candidates', True),
                entropy_objective=interaction.get('entropy_objective', 'joint-sum-v1'),
                adapter=interaction.get('adapter', 'legacy-joint'),
                initial_gate=interaction.get('initial_gate'),
                deterministic_gate_proposals=interaction.get('deterministic_gate_proposals', False))


def policy_action(actor, packet, **kwargs):
    return actor(packet['obs'], packet['motion'], prior=packet.get('prior'),
                 composition=packet.get('composition'), **kwargs)


def transformed_normal(mean, log_std, noise=None, deterministic=False, gate=False):
    std = log_std.exp()
    raw = mean if deterministic else mean + std * (torch.randn_like(mean) if noise is None else noise)
    logp = Normal(mean, std).log_prob(raw)
    logp = logp - 2 * (math.log(2.) - raw - F.softplus(-2 * raw))
    action = raw.tanh()
    if gate:
        action = (action + 1.) * .5
        logp = logp + math.log(2.)
    return action, logp.sum(-1, keepdim=True)


def masked_mean(x, mask, dim):
    weight = mask.to(x.dtype).unsqueeze(-1)
    return (x * weight).sum(dim) / weight.sum(dim).clamp_min(1.)


class InteractionActor(nn.Module):
    def __init__(self, hidden=512, relation=128, peer_candidates=True, entropy_objective='joint-sum-v1',
                 adapter='legacy-joint', initial_gate=None, deterministic_gate_proposals=False):
        super().__init__()
        self.hidden, self.relation = hidden, relation
        self.peer_candidates = bool(peer_candidates)
        self.deterministic_gate_proposals = bool(deterministic_gate_proposals)
        if adapter not in ('legacy-joint', 'joint', 'separable'):
            raise ValueError('Unknown candidate adapter.')
        self.adapter, self.initial_gate = adapter, initial_gate
        if entropy_objective not in ('joint-sum-v1', 'stage-mean-v2', 'proposal-mean-v3', 'task-return-v4'):
            raise ValueError('Unknown entropy objective.')
        self.entropy_objective, self.stage = entropy_objective, 'gate'
        self.base = ActorAdap(6, 8, 3, hidden)
        relation_inputs = 15 if adapter in ('legacy-joint', 'separable') else 19
        self.relations = nn.Sequential(nn.Linear(relation_inputs, relation), nn.LeakyReLU(),
                                       nn.Linear(relation, relation), nn.LeakyReLU())
        self.relation_gate = nn.Linear(relation, 1, bias=False)
        self.gate_log_std = nn.Linear(hidden + relation, 1)
        nn.init.zeros_(self.relation_gate.weight)
        nn.init.zeros_(self.gate_log_std.weight)
        nn.init.constant_(self.gate_log_std.bias, math.log(.1))
        if adapter != 'legacy-joint':
            self.composition_gate = nn.Linear(2, 1, bias=False)
            nn.init.zeros_(self.composition_gate.weight)
        self.initialize_gate()

    def initialize_gate(self):
        if self.initial_gate is not None:
            if not 0. < self.initial_gate < 1.:
                raise ValueError('Initial gate must be strictly interior.')
            nn.init.zeros_(self.base.adap_layer.weight)
            nn.init.constant_(self.base.adap_layer.bias, math.atanh(2.*self.initial_gate-1.))

    def initialize_source(self, weights):
        self.base.load_state_dict(weights, strict=True)
        self.initialize_gate()

    def freeze_proposals(self, freeze=True):
        self.stage = 'gate' if freeze else 'joint'
        for name, parameter in self.base.named_parameters():
            if name.split('.')[0] in ('f1', 'f2', 't1', 'l1', 'l2', 'mean_layer', 'log_std_layer'):
                parameter.requires_grad_(not freeze)

    def encode(self, obs, mask):
        b, n, d = obs.shape
        if d != 14 + 2 * (n - 1) or n < 2:
            raise ValueError('Expected the released observation [B,N,14+2*(N-1)], N>=2.')
        net, act = self.base, self.base.activation
        neighbor_mask = mask[:, None, :].expand(b, n, n)
        off_diagonal = ~torch.eye(n, dtype=torch.bool, device=obs.device)
        neighbor_mask = neighbor_mask[:, off_diagonal].reshape(b, n, n - 1)
        neighbors = act(net.t1(obs[..., 14:].reshape(b, n, n - 1, 2)))
        h = torch.cat((act(net.f1(obs[..., :6])), act(net.f2(obs[..., 6:14])),
                       masked_mean(neighbors, neighbor_mask, -2)), -1)
        return act(net.l2(act(net.l1(h))))

    def relation_features(self, motion, proposals, boids):
        n = motion.shape[-2]
        delta = motion.unsqueeze(-3) - motion.unsqueeze(-2)
        heading = motion[..., 2].unsqueeze(-1)
        c, s = heading.cos(), heading.sin()
        p = torch.stack((c * delta[..., 0] + s * delta[..., 1],
                         -s * delta[..., 0] + c * delta[..., 1]), -1) / 60.
        v = torch.stack((c * delta[..., 3] + s * delta[..., 4],
                         -s * delta[..., 3] + c * delta[..., 4]), -1) / 5.
        relation = torch.cat((p, delta[..., 2:3].sin(), delta[..., 2:3].cos(),
                              v, delta[..., 5:6]), -1)
        candidates = torch.cat((proposals, boids), -1)
        peers = candidates if self.peer_candidates else torch.zeros_like(candidates)
        return torch.cat((relation, candidates.unsqueeze(-2).expand(-1, -1, n, -1),
                          peers.unsqueeze(-3).expand(-1, n, -1, -1)), -1)

    def forward(self, obs, motion, mask=None, deterministic=False, noise=None, prior=None, composition=None):
        motion = motion.to(dtype=obs.dtype)
        b, n = obs.shape[:2]
        if mask is None:
            mask = torch.ones((b, n), dtype=torch.bool, device=obs.device)
        h = self.encode(obs, mask)
        proposals, log_l = transformed_normal(self.base.mean_layer(h),
            self.base.log_std_layer(h).clamp(-20, 2),
            None if noise is None else noise[..., :2],
            deterministic or (self.stage == 'gate' and self.deterministic_gate_proposals))
        boids = obs[..., 12:14] if prior is None else prior
        gates, log_g = self.gates(h, motion, proposals, boids, mask, deterministic,
                                None if noise is None else noise[..., 2:3], composition)
        action = torch.cat((proposals, gates), -1)
        if self.entropy_objective == 'task-return-v4':
            logp = torch.zeros((b, 1), dtype=obs.dtype, device=obs.device)
        elif self.entropy_objective == 'proposal-mean-v3':
            # Blending noise receives no intrinsic reward. Gate-only learning
            # optimizes the physical task; joint learning regularizes only
            # the trainable proposal distribution, as in the source policy.
            logp = (masked_mean(log_l, mask, -2) if self.stage == 'joint'
                    else torch.zeros((b, 1), dtype=obs.dtype, device=obs.device))
        elif self.entropy_objective == 'stage-mean-v2':
            # Team reward is a mean. Match its fleet-size scale, and in the
            # gate-only MDP treat frozen proposals as exogenous candidates.
            # Their state-dependent density must not become a gate reward.
            logp = masked_mean(log_g if self.stage == 'gate' else log_l + log_g, mask, -2)
        else:
            logp = ((log_l + log_g) * mask.unsqueeze(-1)).sum(-2)
        return action, logp

    def gates(self, h, motion, proposals, boids, mask, deterministic=False, noise=None, composition=None):
        """Gate evaluation with explicit candidates, also used by controlled diagnostics."""
        n = proposals.shape[-2]
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        pair_mask &= ~torch.eye(n, dtype=torch.bool, device=proposals.device)
        features = self.relation_features(motion, proposals, boids)
        if self.adapter != 'legacy-joint':
            if composition is None:
                raise ValueError('Composition-aware adapters require an explicit composition.')
            z = torch.cat((composition.unsqueeze(-2).expand(-1, -1, n, -1),
                           composition.unsqueeze(-3).expand(-1, n, -1, -1)), -1)
            # Separable peer logits see kinematics, peer candidates and z only.
            # Own candidates never enter this encoder; there is no learned
            # nonlinear mixer after summation, including the noise pathway.
            features = torch.cat((features[..., :7], features[..., 11:], z), -1) if self.adapter == 'separable' else torch.cat((features, z), -1)
        edges = self.relations(features)
        message = ((edges * pair_mask.unsqueeze(-1)).sum(-2) if self.adapter == 'separable'
                   else masked_mean(edges, pair_mask, -2))
        own = self.base.activation(self.base.a1(torch.cat((proposals, boids), -1)))
        mean = self.base.adap_layer(torch.cat((own, h), -1)) + self.relation_gate(message)
        if self.adapter != 'legacy-joint':
            mean = mean + self.composition_gate(composition)
        if self.adapter == 'separable' and not deterministic:
            raise ValueError('Separable gates require candidate-independent external exploration.')
        log_std = self.gate_log_std(torch.cat((h, message), -1)).clamp(-5, 2)
        return transformed_normal(mean, log_std, noise, deterministic, gate=True)


class CandidateCoordinationGate(nn.Module):
    """A joint, public-information gate distribution with identifiable factors.

    The two basis functions per receiver are its signed intervention and its
    square. Four mixed factors represent the complete interaction of a pair
    of ternary gates. Unlike independent mean gates, joint inference retains
    combinations which are useful only when both receivers change together.
    The state baseline is separate: conditional losses cannot train it.
    """
    def __init__(self, width=64, peer_candidates=True, target_pose=False, complete_state=False, response_encoding=False, cardinal_gates=False, gate_bias=-.05, reaction_encoding=False, potential_value=False, curved_potential=False, constrained_potential=False, coupled_potential=False):
        super().__init__()
        self.width, self.peer_candidates = width, bool(peer_candidates)
        self.target_pose = bool(target_pose)
        self.complete_state = bool(complete_state)
        self.response_encoding = bool(response_encoding)
        self.reaction_encoding = bool(reaction_encoding)
        self.potential_value = bool(potential_value)
        self.curved_potential = bool(curved_potential)
        self.constrained_potential = bool(constrained_potential)
        self.coupled_potential = bool(coupled_potential)
        if coupled_potential and not potential_value:raise ValueError('Coupled evasion needs a response potential.')
        if constrained_potential and not potential_value:raise ValueError('Applied-control features require a response potential.')
        if curved_potential and not potential_value:raise ValueError('Curved fusion needs a response potential.')
        if potential_value and not (response_encoding and cardinal_gates):raise ValueError('Response potentials require motion responses and cardinal gates.')
        if reaction_encoding and not response_encoding:raise ValueError('Joint evasion features require candidate motion responses.')
        self.cardinal_gates, self.gate_bias = bool(cardinal_gates), float(gate_bias)
        def network(inputs, outputs, zero=False):
            net = nn.Sequential(nn.Linear(inputs, width), nn.SiLU(),
                                nn.Linear(width, width), nn.SiLU(), nn.Linear(width, outputs))
            if zero:
                nn.init.zeros_(net[-1].weight)
                nn.init.zeros_(net[-1].bias)
            return net
        self.node = network((25 if target_pose else 22) + (28 if response_encoding else 0), width)
        edge_inputs = (36 if target_pose else 31) + (2 * (13 if target_pose else 10) if complete_state else 0)
        if response_encoding:edge_inputs += 72
        if reaction_encoding:edge_inputs += 24
        self.message = network(edge_inputs, width)
        self.unary = network(2 * width, 2, True)
        self.mixed = network(edge_inputs + (2*width if response_encoding else 0), 4, True)
        self.baseline = network(19 if target_pose else 16, width if complete_state else 1)
        if complete_state:self.global_value = network(2 * width, 1)
        if response_encoding:self.coordination_context = network(19 if target_pose else 16, width)
        if potential_value:
            # Shared coefficients act on physical counterfactual differences.
            # Identical candidate motions therefore have exactly zero value
            # increment, irrespective of the learned coefficient values.
            self.motion_potential = nn.Linear(48, 1, bias=False)
            self.pair_potential = nn.Linear(48, 1, bias=False)
            nn.init.zeros_(self.motion_potential.weight)
            nn.init.zeros_(self.pair_potential.weight)

    @classmethod
    @torch.no_grad()
    def coupled_evasion(cls,packet,own,incoming,enabled):
        """A public nominal evader response to each changing candidate pair.

        Receiver paths are nominal; the evader continually reacts to their
        changing positions. Initial yaw is estimated from observed heading
        history, never read from the simulator's private angular velocity.
        The fixed nominal thrust scale is a representation prior; supervision
        and policy checks still use the original noisy complete environment.
        """
        motion,obs=packet['motion'],packet['obs'];b,n=motion.shape[:2];k=own.shape[3]
        bearing=motion[:,0,2]+obs[:,0,3]
        target=motion[:,0,:2]+obs[:,0,2,None]*torch.stack((bearing.cos(),bearing.sin()),-1)
        angle=motion[:,0,2]+obs[:,0,5]
        velocity=obs[:,0,4,None]*torch.stack((angle.cos(),angle.sin()),-1)
        shape=(b,n,n,k,k)
        position=target[:,None,None,None,None].expand(*shape,2).clone()
        speed=velocity[:,None,None,None,None].expand(*shape,2).clone()
        heading=packet.get('target_heading',angle)[:,None,None,None,None].expand(shape).clone()
        yaw=packet.get('target_turn',torch.zeros_like(angle)).clamp(-3.,3.)[:,None,None,None,None].expand(shape).clone()
        identity=torch.arange(n,device=motion.device)
        def vessel_paths(time):
            first=target[:,None,None]+velocity[:,None,None]-own[:,:,0]
            second=target[:,None,None]+2.*velocity[:,None,None]-own[:,:,1]
            left=(motion[:,:,:2,None].transpose(-1,-2)*(1.-time)+time*first if time<=1. else
                first*(2.-time)+(time-1.)*second)
            if enabled:
                first=target[:,None,None]+velocity[:,None,None]-incoming[:,:,0]
                second=target[:,None,None]+2.*velocity[:,None,None]-incoming[:,:,1]
                right=(motion[:,:,:2,None].transpose(-1,-2)*(1.-time)+time*first if time<=1. else
                    first*(2.-time)+(time-1.)*second)
            else:right=(motion[...,:2]+time*motion[...,3:5])[:,:,None].expand(-1,-1,k,-1)
            return left,right
        result=[]
        for step in range(10):
            left,right=vessel_paths(.2*step)
            force=-.1*position;limit=position.new_full(shape,50.)
            for receiver in range(n):
                other=(right[:,receiver,0] if enabled else motion[:,receiver,:2]+(.2*step)*motion[:,receiver,3:5])
                boat=other[:,None,None,None,None].expand(*shape,2)
                boat=torch.where((identity==receiver)[None,:,None,None,None,None],left[:,:,None,:,None],boat)
                boat=torch.where((identity==receiver)[None,None,:,None,None,None],right[:,None,:,None,:],boat)
                delta=position-boat;distance=delta.norm(dim=-1).clamp_min(1e-5);d=(distance-8.).clamp_min(.1)
                active=d<limit;amplitude=3000.*(1./d-1./limit)/d.square()
                force=force+torch.where(active[...,None],amplitude[...,None]*delta/distance[...,None],torch.zeros_like(delta))
                limit=torch.where(active,d,limit)
            direction=force/force.norm(dim=-1,keepdim=True).clamp_min(1e-5)
            c,s=heading.cos(),heading.sin()
            forward=c*direction[...,0]+s*direction[...,1]
            side=-s*direction[...,0]+c*direction[...,1]
            turn=torch.atan2(side,forward)
            thrust=torch.stack((forward+.75*turn,forward-.75*turn),-1).clamp(-.5,1.)*2250.
            position=position+.2*speed;heading=heading+.2*yaw
            c,s=heading.cos(),heading.sin()
            u=c*speed[...,0]+s*speed[...,1];v=-s*speed[...,0]+c*speed[...,1]
            du=(400.*yaw*v-(100.+150.*u.abs())*u+thrust.sum(-1))/380.
            dv=(-380.*yaw*u-(100.+150.*v.abs())*v)/400.
            dr=(-20.*u*v-(980.+950.*yaw.abs())*yaw+1.25*(thrust[...,0]-thrust[...,1]))/1430.
            u,v,yaw=u+.2*du,v+.2*dv,yaw+.2*dr
            speed=torch.stack((c*u-s*v,s*u+c*v),-1)
            if step in (4,9):
                left,right=vessel_paths(.2*(step+1))
                result.append((position-left[:,:,None,:,None],position-right[:,None,:,None,:],speed.clone()))
        return result

    @classmethod
    def candidate_potentials(cls, packet, baseline, bias=-.05, peer_candidates=True, probe='aligned', curved=False, constrained=False, coupled=False):
        """ANOVA differences of shared nominal-motion potentials.

        The pair potential measures interception distance, angular coverage,
        and the escape direction induced by both candidates. It is a public
        response representation, not a simulated return or a privileged
        target-state input. Single and mixed effects share the same scorer.
        """
        response,paths=((packet['candidate_response'],packet['candidate_paths'])
            if 'candidate_response' in packet else cls.candidate_responses(packet,baseline))
        motion,obs=packet['motion'],packet['obs'];b,n=motion.shape[:2]
        reference=(baseline[...,2]+bias).clamp(0.,1.)
        def branches(candidate_paths,gate):
            center=candidate_paths[:,:,:,1]+gate[:,:,None,None]*(candidate_paths[:,:,:,0]-candidate_paths[:,:,:,1])
            boids,learned=candidate_paths[:,:,:,1],candidate_paths[:,:,:,0]
            return (torch.stack((center,.5*(center+boids),boids,.5*(center+learned),learned),3) if curved else
                torch.stack((center,boids,learned),3))
        own=branches(paths,reference)
        size=own.shape[3]
        incoming=own
        enabled=peer_candidates and probe!='state-only'
        if enabled and probe=='permuted':
            _,other=cls.candidate_responses(packet,baseline,True)
            incoming=branches(other,reference.roll(1,1))
        if constrained:
            own=cls.applied_candidate_paths(packet,baseline,bias,peer_candidates,probe,curved)
            incoming=(cls.applied_candidate_paths(packet,baseline,bias,peer_candidates,probe,curved,True)
                if enabled and probe=='permuted' else own)
        angle=motion[:,0,2]+obs[:,0,3]
        target=motion[:,0,:2]+obs[:,0,2,None]*torch.stack((angle.cos(),angle.sin()),-1)
        angle=motion[:,0,2]+obs[:,0,5]
        velocity=obs[:,0,4,None]*torch.stack((angle.cos(),angle.sin()),-1)
        reactions=cls.coupled_evasion(packet,own,incoming,enabled) if coupled else None
        node_features=[];pair_features=[];identity=torch.arange(n,device=motion.device)
        for t in range(2):
            destination=target+(t+1)*velocity
            kinematic=destination[:,None]-motion[...,:2]-(t+1)*motion[...,3:5]
            left=own[:,:,t]
            right=incoming[:,:,t] if enabled else kinematic[:,:,None].expand(-1,-1,size,-1)
            if coupled:
                joint_left,joint_right,target_velocity=reactions[t]
                mask=1.-torch.eye(n,device=motion.device,dtype=motion.dtype)
                left=(joint_left[:,:,:,:,0]*mask[None,:,:,None,None]).sum(2)/max(n-1,1)
            distance=left.norm(dim=-1).clamp_min(1e-5)
            heading=motion[...,2,None];c,s=heading.cos(),heading.sin()
            forward=(c*left[...,0]+s*left[...,1])/distance
            lateral=(-s*left[...,0]+c*left[...,1])/distance
            node_features.append(torch.stack((distance/10.,distance.square()/100.,
                torch.exp(-distance/5.),torch.exp(-distance/15.),forward,lateral,
                forward*torch.exp(-distance/15.),lateral*torch.exp(-distance/15.)),-1))
            l=left[:,:,None,:,None];r=right[:,None,:,None,:]
            if coupled:l,r=joint_left,joint_right
            di=l.norm(dim=-1).expand(-1,n,n,size,size).clamp_min(1e-5)
            dj=r.norm(dim=-1).expand(-1,n,n,size,size).clamp_min(1e-5)
            separation=(l-r).norm(dim=-1)
            coverage=F.cosine_similarity(l,r,dim=-1,eps=1e-6)
            if coupled:shifted=.5*target_velocity
            else:
                force=(-.1*destination)[:,None,None,None,None].expand(-1,n,n,size,size,-1).clone()
                limit=motion.new_full((b,n,n,size,size),50.)
                for k in range(n):
                    delta=kinematic[:,k,None,None,None,None].expand(-1,n,n,size,size,-1)
                    delta=torch.where((identity==k)[None,:,None,None,None,None],l,delta)
                    delta=torch.where((identity==k)[None,None,:,None,None,None],r,delta)
                    norm=delta.norm(dim=-1).clamp_min(1e-5);d=(norm-8.).clamp_min(.1)
                    active=d<limit;amplitude=3000.*(1./d-1./limit)/d.square()
                    force=force+torch.where(active[...,None],amplitude[...,None]*delta/norm[...,None],torch.zeros_like(delta))
                    limit=torch.where(active,d,limit)
                escape=force/force.norm(dim=-1,keepdim=True).clamp_min(1e-5)
                shifted=.5*velocity.norm(dim=-1)[:,None,None,None,None,None]*escape
            after_i=(l+shifted).norm(dim=-1);after_j=(r+shifted).norm(dim=-1)
            def soft_nearest(a,b,temperature):
                return -temperature*torch.logsumexp(torch.stack((-a/temperature,-b/temperature),-1),-1)/10.
            pair_features.append(torch.stack((soft_nearest(di,dj,2.),soft_nearest(di,dj,8.),
                torch.exp(-(di+dj)/15.),coverage*torch.exp(-(di+dj)/30.),
                torch.log1p(separation/5.),soft_nearest(after_i,after_j,2.),
                soft_nearest(after_i,after_j,8.),(1.-coverage)*torch.exp(-(di+dj)/15.)),-1))
        nodes=torch.cat(node_features,-1);pairs=torch.cat(pair_features,-1)
        # A small, state-conditioned linear potential learns across trajectories
        # without fitting independent unrestricted factor surfaces at each state.
        near=torch.exp(-obs[...,2]/10.)
        node_context=torch.stack((torch.ones_like(near),near,near.square()),-1)
        pair_near=torch.exp(-torch.minimum(obs[...,2,None],obs[:,None,:,2])/10.)
        pair_context=torch.stack((torch.ones_like(pair_near),pair_near,pair_near.square()),-1)
        nodes=(nodes[...,None]*node_context[:,:,None,None,:]).flatten(-2)
        pairs=(pairs[...,None]*pair_context[:,:,:,None,None,None,:]).flatten(-2)
        own_delta=nodes[:,:,1:]-nodes[:,:,:1]
        peer_delta=pairs[:,:,:,1:,0]-pairs[:,:,:,:1,0]
        mask=1.-torch.eye(n,device=motion.device,dtype=motion.dtype)
        context_delta=(peer_delta*mask[None,:,:,None,None]).sum(2)/max(n-1,1)
        mixed=pairs[:,:,:,1:,1:]-pairs[:,:,:,1:,:1]-pairs[:,:,:,:1,1:]+pairs[:,:,:,:1,:1]
        mixed=mixed*mask[None,:,:,None,None,None]*float(enabled)/max(n-1,1)
        return own_delta,context_delta,mixed

    @classmethod
    def cache_potentials(cls,packet,baseline,bias,peer_candidates,probe='aligned',curved=False,constrained=False,coupled=False):
        key='candidate_potential' if peer_candidates and probe!='state-only' else 'independent_potential'
        if curved:key+='2'
        if constrained:key+='s'
        if coupled:key+='e'
        if probe=='permuted' or key+'_own' in packet:return packet
        values=cls.candidate_potentials(packet,baseline,bias,peer_candidates,probe,curved,constrained,coupled)
        return dict(packet,**{key+'_'+part:value for part,value in zip(('own','context','pair'),values)})

    def potential_factors(self,packet,baseline,probe):
        enabled=self.peer_candidates and probe!='state-only'
        key='candidate_potential' if enabled else 'independent_potential'
        if self.curved_potential:key+='2'
        if self.constrained_potential:key+='s'
        if self.coupled_potential:key+='e'
        cache=tuple(packet[key+'_'+part] for part in ('own','context','pair')) if probe!='permuted' and key+'_own' in packet else None
        own,context,pair=(cache if cache is not None else
            self.candidate_potentials(packet,baseline,self.gate_bias,self.peer_candidates,probe,self.curved_potential,self.constrained_potential,self.coupled_potential))
        unary=self.motion_potential(own).squeeze(-1)+self.pair_potential(context).squeeze(-1)
        mixed=self.pair_potential(pair).squeeze(-1)
        mixed=.5*(mixed+mixed.transpose(1,2).transpose(-1,-2))
        state=self.public_inputs(packet,baseline,self.target_pose)[-1]
        value=self.baseline(state)
        value=(self.global_value(torch.cat((value.mean(1),value.amax(1)),-1)).squeeze(-1)
            if self.complete_state else value.mean(1).squeeze(-1))
        return value,unary,mixed,(baseline[...,2]+self.gate_bias).clamp(0.,1.)

    @staticmethod
    def public_inputs(packet, baseline, target_pose=False):
        obs, motion = packet['obs'], packet['motion']
        b, n = obs.shape[:2]
        heading = motion[..., 2]
        c, s = heading.cos(), heading.sin()
        velocity = motion[..., 3:5]
        own_velocity = torch.stack((c * velocity[..., 0] + s * velocity[..., 1],
                                    -s * velocity[..., 0] + c * velocity[..., 1]), -1) / 5.
        target = torch.stack((obs[..., 2] * obs[..., 3].cos(),
                              obs[..., 2] * obs[..., 3].sin()), -1) / 30.
        target_velocity = torch.stack((obs[..., 4] * obs[..., 5].cos(),
                                       obs[..., 4] * obs[..., 5].sin()), -1) / 5.
        origin = torch.stack((obs[..., 0] * obs[..., 1].cos(),
                              obs[..., 0] * obs[..., 1].sin()), -1) / 60.
        order = (torch.arange(n, device=obs.device, dtype=obs.dtype) + 1.) / n
        order = order.expand(b, n)[..., None]
        state = torch.cat((origin, target, own_velocity, target_velocity,
                           motion[..., 5:6], order), -1)
        if target_pose:
            angle = packet['target_heading'][:, None] - heading
            target_orientation = torch.stack((angle.cos(), angle.sin()), -1)
            remaining = packet['time_remaining'][:, None, None].expand(-1, n, -1)
            state = torch.cat((state, target_orientation, remaining), -1)
        boids = obs[..., 12:14]
        gap = baseline[..., :2] - boids
        candidate = torch.cat((baseline, boids, gap,
                               gap.sum(-1, keepdim=True), gap.diff(dim=-1)), -1)
        # Candidate-dependent acceleration is observable from nominal thrust.
        blended = baseline[..., 2:3] * baseline[..., :2] + (1. - baseline[..., 2:3]) * boids
        node = torch.cat((state, candidate, blended, gap.square().sum(-1, keepdim=True)), -1)
        delta = motion[:, None, :, :2] - motion[:, :, None, :2]
        distance = delta.norm(dim=-1)
        relative = torch.stack((c[..., None] * delta[..., 0] + s[..., None] * delta[..., 1],
                                -s[..., None] * delta[..., 0] + c[..., None] * delta[..., 1]), -1) / 30.
        dv = velocity[:, None] - velocity[:, :, None]
        relative_velocity = torch.stack((c[..., None] * dv[..., 0] + s[..., None] * dv[..., 1],
                                         -s[..., None] * dv[..., 0] + c[..., None] * dv[..., 1]), -1) / 5.
        angle = heading[:, None] - heading[:, :, None]
        geometry = torch.cat((relative, relative_velocity, angle.sin()[..., None],
                              angle.cos()[..., None], distance[..., None] / 30.,
                              order[:, :, None].expand(-1, -1, n, -1),
                              order[:, None].expand(-1, n, -1, -1),
                              target[:, :, None].expand(-1, -1, n, -1),
                              target[:, None].expand(-1, n, -1, -1)), -1)
        if target_pose:
            geometry = torch.cat((geometry, target_orientation[:, :, None].expand(-1, -1, n, -1),
                                   target_orientation[:, None].expand(-1, n, -1, -1),
                                   remaining[:, :, None].expand(-1, -1, n, -1)), -1)
        state_value = torch.cat((state, obs[..., 6:12] / obs.new_tensor([15., 3.15, 5., 3.15, 30., 3.15])), -1)
        return node, candidate, geometry, state_value

    @staticmethod
    def candidate_responses(packet, baseline, permuted=False, applied_thrust=None):
        """Public, nominal motion features for the two current candidates.

        Eight quarter-second integrations supply an inductive bias, not a
        task value or an online action search. Water velocity is approximated
        by measured velocity; training targets retain the actual CBF, water
        disturbance, evader response and full subsequent control policy.
        """
        motion, obs = packet['motion'], packet['obs']
        b, n = baseline.shape[:2]
        candidates = torch.stack((baseline[..., :2], obs[..., 12:14]), 2)
        if permuted:candidates = candidates.roll(1, 1)
        thrust = (750. * candidates + 250.).clamp(-500., 1000.) if applied_thrust is None else applied_thrust
        choices=thrust.shape[2]
        position = motion[..., :2, None].transpose(-1, -2).expand(-1, -1, choices, -1).clone()
        heading = motion[..., 2, None].expand(-1, -1, choices).clone()
        velocity = motion[..., None, 3:6].expand(-1, -1, choices, -1).clone()
        angle = motion[..., 2] + obs[..., 3]
        target = motion[..., :2] + obs[..., 2, None] * torch.stack((angle.cos(), angle.sin()), -1)
        angle = motion[..., 2] + obs[..., 5]
        target_velocity = obs[..., 4, None] * torch.stack((angle.cos(), angle.sin()), -1)
        snapshots=[];encoded=[]
        c0, s0 = motion[..., 2, None].cos(), motion[..., 2, None].sin()
        for step in range(8):
            position = position + .25 * velocity[..., :2]
            heading = heading + .25 * velocity[..., 2]
            c, s = heading.cos(), heading.sin()
            u = c * velocity[..., 0] + s * velocity[..., 1]
            v = -s * velocity[..., 0] + c * velocity[..., 1]
            r = velocity[..., 2]
            du = (400.*r*v - (100.+150.*u.abs())*u + thrust.sum(-1))/380.
            dv = (-380.*r*u - (100.+150.*v.abs())*v)/400.
            dr = (-20.*u*v - (980.+950.*r.abs())*r + 1.25*(thrust[..., 0]-thrust[..., 1]))/1430.
            u, v, r = u+.25*du, v+.25*dv, r+.25*dr
            velocity = torch.stack((c*u-s*v, s*u+c*v, r), -1)
            if step in (3,7):
                relative = target[:, :, None] + (.25*(step+1))*target_velocity[:, :, None] - position
                dv = target_velocity[:, :, None] - velocity[..., :2]
                direction = heading - motion[..., 2, None]
                encoded.append(torch.stack((
                    (c0*relative[..., 0]+s0*relative[..., 1])/30.,
                    (-s0*relative[..., 0]+c0*relative[..., 1])/30.,
                    (c0*dv[..., 0]+s0*dv[..., 1])/5.,
                    (-s0*dv[..., 0]+c0*dv[..., 1])/5.,
                    direction.cos(), direction.sin(), relative.norm(dim=-1)/30.), -1))
                snapshots.append(relative)
        return torch.cat(encoded, -1).flatten(2), torch.stack(snapshots, 2)

    @classmethod
    @torch.no_grad()
    def applied_candidate_paths(cls,packet,baseline,bias,peer_candidates,probe,curved,permuted_own=False):
        """Predict motions from the unchanged public CBF's actual outputs.

        Each receiver retains its own candidates. Incoming candidates enter
        the joint safety projection at their current reference gates. The
        no-peer view instead uses a public velocity-maintenance reference
        for other receivers, so their current candidate values cannot leak.
        """
        from train_response_fusion import SafetyDerivative
        motion,obs=packet['motion'],packet['obs'];b,n=motion.shape[:2]
        enabled=peer_candidates and probe!='state-only'
        incoming=baseline.roll(1,1) if enabled and probe=='permuted' else baseline
        incoming_boids=obs[...,12:14].roll(1,1) if enabled and probe=='permuted' else obs[...,12:14]
        gate=(incoming[...,2]+bias).clamp(0.,1.)
        peer_thrust=750.*(incoming_boids+gate[...,None]*(incoming[...,:2]-incoming_boids))+250.
        if not enabled:
            heading=motion[...,2];u=heading.cos()*motion[...,3]+heading.sin()*motion[...,4]
            r=motion[...,5];drag=(100.+150.*u.abs())*u;torque=(980.+950.*r.abs())*r/1.25
            peer_thrust=torch.stack((.5*(drag+torque),.5*(drag-torque)),-1).clamp(-500.,1000.)
        own_base=baseline.roll(1,1) if permuted_own else baseline
        own_boids=obs[...,12:14].roll(1,1) if permuted_own else obs[...,12:14]
        r=(own_base[...,2]+bias).clamp(0.,1.)
        gates=(torch.stack((r,.5*r,torch.zeros_like(r),.5*(1.+r),torch.ones_like(r)),2) if curved else
            torch.stack((r,torch.zeros_like(r),torch.ones_like(r)),2))
        choices=gates.shape[-1]
        own=750.*(own_boids[:,:,None]+gates[...,None]*(own_base[...,:2]-own_boids)[:,:,None])+250.
        commands=peer_thrust[:,None,None].expand(-1,n,choices,-1,-1).clone()
        receivers=torch.arange(n,device=motion.device)
        receiver_index=receivers[None,:,None,None,None].expand(b,-1,choices,1,2)
        commands.scatter_(3,receiver_index,own[:,:,:,None])
        poses=motion[:,None,None].expand(-1,n,choices,-1,-1).reshape(-1,n,6)
        commands=commands.reshape(-1,n,2)
        filtered=torch.cat([SafetyDerivative.apply(poses[i:i+32],commands[i:i+32]) for i in range(0,len(commands),32)])
        filtered=filtered.reshape(b,n,choices,n,2)
        applied=filtered.gather(3,receiver_index).squeeze(3)
        return cls.candidate_responses(packet,baseline,applied_thrust=applied)[1]

    @staticmethod
    def candidate_reactions(packet, paths, incoming_paths=None):
        """Candidate interactions through the evader's public nominal APF.

        Other vessels follow measured velocity. For each receiver pair, the
        two candidate paths replace those kinematic paths, individually and
        jointly. Four second differences of evasion direction/magnitude at
        one and two seconds encode nonadditive reactions. This uses neither
        future disturbances nor private target dynamics/return labels.
        """
        incoming_paths=paths if incoming_paths is None else incoming_paths
        motion,obs=packet['motion'],packet['obs'];b,n=motion.shape[:2]
        angle=motion[:,0,2]+obs[:,0,3]
        target=motion[:,0,:2]+obs[:,0,2,None]*torch.stack((angle.cos(),angle.sin()),-1)
        angle=motion[:,0,2]+obs[:,0,5]
        velocity=obs[:,0,4,None]*torch.stack((angle.cos(),angle.sin()),-1)
        identity=torch.arange(n,device=motion.device);features=[]
        c,s=motion[...,2].cos()[:,:,None,None,None],motion[...,2].sin()[:,:,None,None,None]
        for t in range(2):
            destination=target+(t+1)*velocity
            reference=destination[:,None]-motion[...,:2]-(t+1)*motion[...,3:5]
            own=torch.cat((reference[:,:,None],paths[:,:,t]),2)
            peer=torch.cat((reference[:,:,None],incoming_paths[:,:,t]),2)
            force=(-.1*destination)[:,None,None,None,None].expand(-1,n,n,3,3,-1).clone()
            limit=motion.new_full((b,n,n,3,3),50.)
            for k in range(n):
                delta=reference[:,k,None,None,None,None].expand(-1,n,n,3,3,-1)
                delta=torch.where((identity==k)[None,:,None,None,None,None],own[:,:,None,:,None],delta)
                delta=torch.where((identity==k)[None,None,:,None,None,None],peer[:,None,:,None,:],delta)
                norm=delta.norm(dim=-1).clamp_min(1e-12);distance=(norm-8.).clamp_min(.1)
                active=distance<limit;size=3000.*(1./distance-1./limit)/distance.square()
                force=force+torch.where(active[...,None],size[...,None]*delta/norm[...,None],torch.zeros_like(delta))
                limit=torch.where(active,distance,limit)
            norm=force.norm(dim=-1).clamp_min(1e-12);direction=force/norm[...,None]
            response=torch.stack((c*direction[...,0]+s*direction[...,1],
                -s*direction[...,0]+c*direction[...,1],norm.log1p()/5.),-1)
            interaction=response[:,:,:,1:,1:]-response[:,:,:,1:,:1]-response[:,:,:,:1,1:]+response[:,:,:,:1,:1]
            features.append(interaction.flatten(3))
        mask=1.-torch.eye(n,device=motion.device,dtype=motion.dtype)
        return torch.cat(features,-1)*mask[None,:,:,None]

    def factors(self, packet, baseline, probe='aligned'):
        if self.potential_value:return self.potential_factors(packet,baseline,probe)
        node, candidate, geometry, state = self.public_inputs(packet, baseline, self.target_pose)
        n = baseline.shape[1]
        own = candidate[:, :, None].expand(-1, -1, n, -1)
        peer = candidate[:, None].expand(-1, n, -1, -1)
        enabled = self.peer_candidates and probe != 'state-only'
        if not enabled:
            peer = torch.zeros_like(peer)
        elif probe == 'permuted':
            peer = candidate.roll(1, 1)[:, None].expand(-1, n, -1, -1)
        edge = torch.cat((geometry, own, peer), -1)
        if self.complete_state:
            public = state[..., :-6]
            edge = torch.cat((edge, public[:, :, None].expand(-1, -1, n, -1),
                              public[:, None].expand(-1, n, -1, -1)), -1)
        if self.response_encoding:
            response, paths = ((packet['candidate_response'],packet['candidate_paths'])
                if 'candidate_response' in packet else self.candidate_responses(packet, baseline))
            incoming, peer_paths = (self.candidate_responses(packet, baseline, True)
                                    if probe == 'permuted' and enabled else (response, paths))
            if not enabled:incoming = torch.zeros_like(incoming)
            # Each receiver's two nominal responses meet both peer responses
            # at one and two seconds: separation and target angular coverage.
            left = paths[:, :, None, :, :, None, :]
            right = peer_paths[:, None, :, :, None, :, :]
            separation = (left-right).norm(dim=-1)/30.
            coverage = F.cosine_similarity(left, right, dim=-1, eps=1e-6)
            paired = torch.stack((separation, coverage), -1).flatten(3) * float(enabled)
            edge = torch.cat((edge, response[:, :, None].expand(-1, -1, n, -1),
                              incoming[:, None].expand(-1, n, -1, -1), paired), -1)
            if self.reaction_encoding:
                reaction=(packet['candidate_reaction'] if 'candidate_reaction' in packet and probe=='aligned' else
                    self.candidate_reactions(packet,paths,peer_paths)) if enabled else edge.new_zeros(*edge.shape[:-1],24)
                edge=torch.cat((edge,reaction),-1)
            node = torch.cat((node, response), -1)
        mask = 1. - torch.eye(n, device=node.device, dtype=node.dtype)
        proximity = mask / (1. + geometry[..., 6].square())
        message = (self.message(edge) * proximity[..., None]).sum(2) / proximity.sum(-1, keepdim=True).clamp_min(1.)
        unary = self.unary(torch.cat((self.node(node), message), -1))
        mixed_input = edge
        if self.response_encoding:
            context = self.coordination_context(state)
            context = torch.cat((context.mean(1), context.amax(1)), -1)
            # Pair synergy depends on the rest of the fleet's ability to
            # intercept. This context reads public state, never an unmasked
            # copy of the peer candidates used by intervention probes.
            mixed_input = torch.cat((edge,context[:,None,None].expand(-1,n,n,-1)), -1)
        mixed = self.mixed(mixed_input).reshape(*edge.shape[:-1], 2, 2)
        mixed = .5 * (mixed + mixed.transpose(1, 2).transpose(-1, -2))
        mixed = mixed * mask[None, :, :, None, None] * float(enabled)
        value = self.baseline(state)
        value = (self.global_value(torch.cat((value.mean(1), value.amax(1)), -1)).squeeze(-1)
                 if self.complete_state else value.mean(1).squeeze(-1))
        factors = (value, unary, mixed)
        return (*factors, (baseline[..., 2]+self.gate_bias).clamp(0.,1.)) if self.cardinal_gates else factors

    @staticmethod
    def energies(factors, programs):
        _, unary, mixed = factors[:3]
        if programs.ndim == 2:
            programs = programs[None].expand(len(unary), -1, -1)
        if len(factors)==4:
            reference = factors[3][:,None]
            # Cardinal candidate coordinates: coefficient units are seconds
            # for switching to Boids or learned control, even at a nearly
            # saturated source gate. No ill-conditioned inverse gate width.
            basis = torch.stack(((-programs/reference.clamp_min(1e-6)).clamp(0.,1.),
                                  (programs/(1.-reference).clamp_min(1e-6)).clamp(0.,1.)), -1)
            if unary.shape[-1]==4:
                # Quadratic cardinal interpolation at 0, half, and the pure
                # candidate on each side. Both endpoint costs may be positive
                # while an interior cooperative mixture has negative cost.
                basis=torch.stack((4.*basis*(1.-basis),basis*(2.*basis-1.)),-1).flatten(-2)
        else:basis = torch.stack((programs, programs.square()), -1)
        return (torch.einsum('bni,bkni->bk', unary, basis) +
                .5 * torch.einsum('bnmij,bkni,bkmj->bk', mixed, basis, basis))

    def forward(self, packet, baseline, programs, probe='aligned'):
        factors = self.factors(packet, baseline, probe)
        return factors[0], self.energies(factors, programs)

    def log_policy(self, packet, baseline, programs, temperature=1., reference_prior=.2, probe='aligned'):
        _, energy = self(packet, baseline, programs, probe)
        displacement = programs.square().sum(-1)
        return (-energy / temperature - reference_prior * displacement).log_softmax(-1)


class EnsembleCoordinationValue(nn.Module):
    """Bootstrap conditional values with conservative, reference-anchored cost."""
    def __init__(self, width=64, peer_candidates=True, members=5, risk=.5, target_pose=False, complete_state=False, response_encoding=False, cardinal_gates=False, gate_bias=-.05, reaction_encoding=False, potential_value=False, curved_potential=False, constrained_potential=False, coupled_potential=False):
        super().__init__()
        self.width, self.peer_candidates = width, bool(peer_candidates)
        self.register_buffer('risk_scale', torch.tensor(float(risk)))
        self.complete_state = bool(complete_state)
        self.response_encoding = bool(response_encoding)
        self.reaction_encoding = bool(reaction_encoding)
        self.potential_value = bool(potential_value)
        self.curved_potential = bool(curved_potential)
        self.constrained_potential = bool(constrained_potential)
        self.coupled_potential = bool(coupled_potential)
        self.cardinal_gates, self.gate_bias = bool(cardinal_gates), float(gate_bias)
        self.members = nn.ModuleList([CandidateCoordinationGate(width, peer_candidates, target_pose, complete_state, response_encoding, cardinal_gates, gate_bias, reaction_encoding, potential_value, curved_potential, constrained_potential, coupled_potential) for _ in range(members)])

    @property
    def risk(self):
        return float(self.risk_scale)

    @risk.setter
    def risk(self, value):
        self.risk_scale.fill_(float(value))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        # Earlier checkpoints stored this coefficient in their settings. The
        # constructor supplies that exact value; newer blocks serialize it.
        state_dict.setdefault(prefix + 'risk_scale', self.risk_scale.clone())
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def estimates(self, packet, baseline, programs, probe='aligned'):
        if self.potential_value:packet=CandidateCoordinationGate.cache_potentials(packet,baseline,self.gate_bias,self.peer_candidates,probe,self.curved_potential,self.constrained_potential,self.coupled_potential)
        if self.response_encoding and 'candidate_response' not in packet:
            response, paths = CandidateCoordinationGate.candidate_responses(packet, baseline)
            packet = dict(packet, candidate_response=response, candidate_paths=paths)
        if self.reaction_encoding and self.peer_candidates and probe=='aligned' and 'candidate_reaction' not in packet:
            packet=dict(packet,candidate_reaction=CandidateCoordinationGate.candidate_reactions(packet,packet['candidate_paths']))
        predictions = [member(packet, baseline, programs, probe) for member in self.members]
        return torch.stack([p[0] for p in predictions]), torch.stack([p[1] for p in predictions])

    def forward(self, packet, baseline, programs, probe='aligned'):
        values, energies = self.estimates(packet, baseline, programs, probe)
        return values.mean(0), energies.mean(0) + self.risk_scale * energies.std(0, unbiased=False)


class MirrorCoordinationGate(nn.Module):
    """Exact KL policy improvement in the learned coordination-factor space.

    Each frozen increment is a conditional critic fitted to the preceding
    deployed policy. Summing its unary and mixed energies implements
    log pi_new = log pi_old - eta Q_old - log Z exactly; no second network
    must rediscover the critic's conditional decisions by imitation.
    """
    def __init__(self, width=64, peer_candidates=True, increments=0, members=1, risk=.5, target_pose=False, complete_state=False, response_encoding=False, cardinal_gates=False, gate_bias=-.05, reaction_encoding=False, potential_value=False, curved_potential=False, constrained_potential=False, coupled_potential=False):
        super().__init__()
        self.width, self.peer_candidates = width, bool(peer_candidates)
        self.members, self.risk = members, risk
        self.complete_state = bool(complete_state)
        self.response_encoding = bool(response_encoding)
        self.reaction_encoding = bool(reaction_encoding)
        self.potential_value = bool(potential_value)
        self.curved_potential = bool(curved_potential)
        self.constrained_potential = bool(constrained_potential)
        self.coupled_potential = bool(coupled_potential)
        self.cardinal_gates, self.gate_bias = bool(cardinal_gates), float(gate_bias)
        def block():
            return (CandidateCoordinationGate(width, peer_candidates, target_pose, complete_state, response_encoding, cardinal_gates, gate_bias, reaction_encoding, potential_value, curved_potential, constrained_potential, coupled_potential) if members==1 else
                    EnsembleCoordinationValue(width, peer_candidates, members, risk, target_pose, complete_state, response_encoding, cardinal_gates, gate_bias, reaction_encoding, potential_value, curved_potential, constrained_potential, coupled_potential))
        self.blocks = nn.ModuleList([block() for _ in range(increments)])
        self.register_buffer('coefficients', torch.zeros(increments))

    def add_value(self, critic, step):
        count = len(critic.members) if hasattr(critic, 'members') else 1
        if (count != self.members or critic.peer_candidates != self.peer_candidates or critic.complete_state != self.complete_state or critic.response_encoding != self.response_encoding or
                critic.cardinal_gates != self.cardinal_gates or critic.gate_bias != self.gate_bias or critic.reaction_encoding != self.reaction_encoding or critic.potential_value != self.potential_value or critic.curved_potential != self.curved_potential or critic.constrained_potential != self.constrained_potential or critic.coupled_potential != self.coupled_potential):
            raise ValueError('A mirror increment must retain the serialized critic architecture.')
        self.blocks.append(copy.deepcopy(critic).eval().requires_grad_(False))
        self.coefficients = torch.cat((self.coefficients, self.coefficients.new_tensor([step])))

    def forward(self, packet, baseline, programs, probe='aligned'):
        if programs.ndim == 2:
            programs = programs[None].expand(len(baseline), -1, -1)
        energy = baseline.new_zeros(programs.shape[:2])
        for coefficient, block in zip(self.coefficients, self.blocks):
            energy = energy + coefficient * block(packet, baseline, programs, probe)[1]
        return baseline.new_zeros(len(baseline)), energy

    def prepare_factors(self, packet, baseline, probe='aligned'):
        if self.potential_value:packet=CandidateCoordinationGate.cache_potentials(packet,baseline,self.gate_bias,self.peer_candidates,probe,self.curved_potential,self.constrained_potential,self.coupled_potential)
        if self.response_encoding and 'candidate_response' not in packet:
            response, paths = CandidateCoordinationGate.candidate_responses(packet, baseline)
            packet = dict(packet, candidate_response=response, candidate_paths=paths)
        if self.reaction_encoding and self.peer_candidates and probe=='aligned' and 'candidate_reaction' not in packet:
            packet=dict(packet,candidate_reaction=CandidateCoordinationGate.candidate_reactions(packet,packet['candidate_paths']))
        prepared=[]
        for coefficient,block in zip(self.coefficients,self.blocks):
            members=block.members if hasattr(block,'members') else [block]
            prepared.append((coefficient, getattr(block,'risk_scale',0.),
                             [member.factors(packet,baseline,probe) for member in members]))
        return prepared

    @staticmethod
    def prepared_energy(prepared, programs):
        energy=programs.new_zeros(programs.shape[:2])
        for coefficient,risk,factors in prepared:
            predictions=torch.stack([CandidateCoordinationGate.energies(factor,programs) for factor in factors])
            energy=energy+coefficient*(predictions.mean(0)+risk*predictions.std(0,unbiased=False))
        return energy

    def log_policy(self, packet, baseline, programs, temperature=1., reference_prior=.2, probe='aligned'):
        _, energy = self(packet, baseline, programs, probe)
        return (-energy / temperature - reference_prior * programs.square().sum(-1)).log_softmax(-1)


class TeamValue(nn.Module):
    def __init__(self, hidden=512, relation=128, control_coordinates='raw-v1',
                 state_coordinates='dynamics-v1', architecture='mlp'):
        super().__init__()
        if architecture not in ('mlp', 'quadratic-control-v1'):
            raise ValueError('Unknown joint critic architecture.')
        if architecture == 'quadratic-control-v1' and control_coordinates != 'nominal-thrust-v2':
            raise ValueError('Structured control values require physical nominal-thrust coordinates.')
        self.architecture = architecture
        if control_coordinates not in ('raw-v1','nominal-thrust-v2'):
            raise ValueError('Unknown critic control coordinates.')
        self.control_coordinates = control_coordinates
        if state_coordinates not in ('dynamics-v1', 'execution-v2', 'controller-v3'):
            raise ValueError('Unknown critic state coordinates.')
        self.state_coordinates = state_coordinates
        if state_coordinates != 'dynamics-v1':
            self.register_buffer('state_coordinates_version', torch.tensor(3 if state_coordinates == 'controller-v3' else 2, dtype=torch.int64))
        if control_coordinates != 'raw-v1':
            self.register_buffer('control_coordinates_version',torch.tensor(2,dtype=torch.int64))
        inputs = {'dynamics-v1': 12, 'execution-v2': 16, 'controller-v3': 26}[state_coordinates]
        self.node = nn.Sequential(nn.Linear(inputs, relation), nn.LeakyReLU(),
                                  nn.Linear(relation, relation), nn.LeakyReLU())
        self.pair = nn.Sequential(nn.Linear(2 * relation + 1, relation), nn.LeakyReLU(),
                                  nn.Linear(relation, relation), nn.LeakyReLU())
        if state_coordinates == 'controller-v3':
            self.memory = nn.Sequential(nn.Linear(MEMORY_GLOBAL_DIM, relation), nn.LeakyReLU(),
                                        nn.Linear(relation, relation), nn.LeakyReLU())
        self.output = nn.Sequential(nn.Linear((3 if state_coordinates == 'controller-v3' else 2) * relation + 10, hidden), nn.LeakyReLU(),
                                    nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.Linear(hidden, 1))
        if architecture == 'quadratic-control-v1':
            self.unary_control = nn.Sequential(nn.Linear(2*relation+10, relation), nn.LeakyReLU(), nn.Linear(relation, 5))
            self.pair_control = nn.Linear(relation, 3)
            for layer in (self.unary_control[-1], self.pair_control):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, packet, action, mask=None):
        b, n = action.shape[:2]
        if mask is None:
            mask = torch.ones((b, n), dtype=torch.bool, device=action.device)
        central = packet['central']
        if central.shape[-1] != 7 * (n + 1) + 2:
            raise ValueError('Central state does not match the team size.')
        state = central[..., :7*n].reshape(b, n, 7)
        boids = packet.get('prior', packet['obs'][..., 12:14])
        controls = action
        if self.control_coordinates == 'nominal-thrust-v2':
            # Physical thrust is 750 * nominal + 250. CBF receives this
            # composition only; its arbitrary proposal/gate factorization
            # cannot change a root action value. Root entropy is outside Q.
            nominal = action[..., 2:3] * action[..., :2] + (1.-action[..., 2:3]) * boids
            controls = torch.cat((nominal, nominal[..., :1]-nominal[..., 1:2]), -1)
        node_inputs = [state, torch.zeros_like(controls) if self.architecture == 'quadratic-control-v1' else controls, boids]
        if self.state_coordinates != 'dynamics-v1':
            # The CBF consumes measured ground velocity, whereas central holds
            # the dynamics' relative-to-water velocity. Neither replaces the other.
            velocity = packet['motion'][..., 3:6].to(state.dtype)
            velocity = velocity / velocity.new_tensor([5., 5., 1.])
            # Released APF obstacle traversal is order dependent. Preserve that
            # execution order without changing the opponent or adding a network.
            # Compaction over valid nodes keeps padding out of the simulator order.
            order = mask.cumsum(-1).to(state.dtype).unsqueeze(-1) / 7.
            node_inputs.extend((velocity, order))
        if self.state_coordinates == 'controller-v3':
            node_inputs.extend((packet['memory_nodes'], packet['composition']))
        nodes = self.node(torch.cat(node_inputs, -1))
        left, right = nodes.unsqueeze(-2), nodes.unsqueeze(-3)
        distance = (state[..., :2].unsqueeze(-2) - state[..., :2].unsqueeze(-3)).norm(dim=-1, keepdim=True)
        pairs = self.pair(torch.cat((left + right, (left - right).abs(), distance), -1))
        pair_mask = mask.unsqueeze(-1) & mask.unsqueeze(-2)
        pair_mask &= ~torch.eye(n, dtype=torch.bool, device=action.device)
        pair_pool = masked_mean(pairs.flatten(1, 2), pair_mask.flatten(1, 2), 1)
        task = torch.cat((central[..., 7*n:], mask.sum(-1, keepdim=True).to(action.dtype) / 7.), -1)
        features = [masked_mean(nodes, mask, 1), pair_pool, task]
        if self.state_coordinates == 'controller-v3':
            features.append(self.memory(packet['memory_global']))
        value = self.output(torch.cat(features, -1))
        if self.architecture == 'quadratic-control-v1':
            # State value and control sensitivity have separate heads. Paired
            # returns cancel the state head exactly, including its gradients.
            # Pair products retain conditional effects of other vessels' thrust.
            context = torch.cat((pair_pool, task), -1)[:, None].expand(-1, n, -1)
            coefficients = self.unary_control(torch.cat((nodes, context), -1))
            u, v = controls[..., 0], controls[..., 1]
            basis = torch.stack((u, v, u.square(), u*v, v.square()), -1)
            unary = masked_mean((coefficients*basis).sum(-1, keepdim=True), mask, 1)
            pair_basis = torch.stack((u[:, :, None]*u[:, None, :],
                u[:, :, None]*v[:, None, :]+v[:, :, None]*u[:, None, :],
                v[:, :, None]*v[:, None, :]), -1)
            pair_value = (self.pair_control(pairs)*pair_basis).sum(-1, keepdim=True)
            value = value + unary + masked_mean(pair_value.flatten(1, 2), pair_mask.flatten(1, 2), 1)
        return value


class TwinTeamCritic(nn.Module):
    def __init__(self, hidden=512, relation=128, control_coordinates='raw-v1',
                 state_coordinates='dynamics-v1', architecture='mlp'):
        super().__init__()
        self.q1 = TeamValue(hidden, relation, control_coordinates, state_coordinates, architecture)
        self.q2 = TeamValue(hidden, relation, control_coordinates, state_coordinates, architecture)

    def forward(self, packet, action, mask=None):
        return self.q1(packet, action, mask), self.q2(packet, action, mask)


class JointReplay:
    def __init__(self, capacity=1_000_000):
        self.capacity, self.count, self.size = int(capacity), 0, 0
        self.arrays = {}
        self.contexts = []

    def store(self, before, action, executed, individual_reward, after, terminated, *, policy_cost=None, next_context=None):
        record = {**before, **{'next_' + k: v for k, v in after.items()},
                  'action': action, 'executed': executed,
                  'individual_reward': individual_reward,
                  'reward': np.asarray([np.mean(individual_reward)]),
                  'done': np.asarray([float(terminated)])}
        if policy_cost is not None:
            record['policy_cost'] = np.asarray([policy_cost])
        for key, value in record.items():
            value = np.asarray(value, dtype=np.float32)
            if not np.isfinite(value).all():
                raise FloatingPointError('Non-finite replay field: ' + key)
            if key not in self.arrays:
                self.arrays[key] = np.empty((self.capacity, *value.shape), dtype=np.float32)
            if self.arrays[key].shape[1:] != value.shape:
                raise ValueError('Cannot mix incompatible team dimensions in one replay.')
            self.arrays[key][self.count] = value
        if next_context is not None:
            if len(self.contexts) < self.capacity:
                self.contexts.append(copy.deepcopy(next_context))
            else:
                self.contexts[self.count] = copy.deepcopy(next_context)
        elif self.contexts:
            raise ValueError('Controller replay contexts cannot be omitted mid-run.')
        self.count = (self.count + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device='cpu'):
        if not self.size:
            raise ValueError('Empty replay.')
        index = np.random.randint(self.size, size=batch_size)
        batch = {k: torch.as_tensor(v[index], device=device) for k, v in self.arrays.items()}
        if self.contexts:
            batch['_next_contexts'] = [self.contexts[i] for i in index]
        return batch

    def complete_returns(self, gamma):
        """Warmup Q targets from complete real episodes, excluding root entropy."""
        if self.count != self.size or 'policy_cost' not in self.arrays:
            raise ValueError('Complete real-return warmup requires intact on-policy history and policy costs.')
        terminal = np.flatnonzero(self.arrays['done'][:self.size, 0])
        if not len(terminal):
            raise ValueError('No completed real episode is available for critic warmup.')
        stop = int(terminal[-1]) + 1
        returns = np.empty(stop, dtype=np.float64)
        for i in range(stop - 1, -1, -1):
            returns[i] = float(self.arrays['reward'][i, 0])
            if not self.arrays['done'][i, 0]:
                returns[i] += gamma * (returns[i + 1] - self.arrays['policy_cost'][i + 1, 0])
        return returns.astype(np.float32)

    def state_dict(self):
        # Adjacent transitions share the same physical state. Store next-state
        # exceptions at episode/ring boundaries, without reducing precision.
        arrays, exceptions = {}, {}
        for key, value in self.arrays.items():
            if not key.startswith('next_'):
                arrays[key] = value[:self.size].copy()
                continue
            expected = np.roll(self.arrays[key[5:]][:self.size], -1, axis=0)
            different = np.any(value[:self.size] != expected, axis=tuple(range(1, value.ndim)))
            index = np.flatnonzero(different)
            exceptions[key] = dict(index=index, values=value[index].copy())
        return dict(capacity=self.capacity, count=self.count, size=self.size,
                    arrays=arrays, next_exceptions=exceptions, contexts=copy.deepcopy(self.contexts))

    def load_state_dict(self, state):
        self.capacity, self.count, self.size = state['capacity'], state['count'], state['size']
        self.arrays = {}
        self.contexts = copy.deepcopy(state.get('contexts', []))
        for k, v in state['arrays'].items():
            self.arrays[k] = np.empty((self.capacity, *v.shape[1:]), dtype=np.float32)
            self.arrays[k][:self.size] = v
        for key, values in state.get('next_exceptions', {}).items():
            original = self.arrays[key[5:]]
            self.arrays[key] = np.empty_like(original)
            self.arrays[key][:self.size] = np.roll(original[:self.size], -1, axis=0)
            self.arrays[key][values['index']] = values['values']


def current_and_next(batch):
    keys = [k for k in (*PACKET_KEYS, *CONTROL_KEYS) if k in batch]
    return ({k: batch[k] for k in keys}, {k: batch['next_' + k] for k in keys})


class InteractionSAC:
    legacy = False

    def __init__(self, config, device='cpu'):
        if config['interaction'].get('gate_objective', 'critic-v1') != 'critic-v1':
            raise ValueError('The paper trains the adapter through the joint critic; direct gate supervision is retired.')
        if config['interaction'].get('intervention_scope', 'single') != 'single':
            raise ValueError('The paper uses single-vessel current-coefficient interventions.')
        self.config, self.device = copy.deepcopy(config), torch.device(device)
        rl = config['rl']
        self.gamma, self.tau = rl['GAMMA'], rl['TAU']
        self.batch_size = rl['batch_size']
        self.entropy_objective = config['interaction'].get('entropy_objective', 'joint-sum-v1')
        self.actor = InteractionActor(rl['hidden_dim'], config['interaction']['relation_dim'],
            **actor_kwargs(config)).to(self.device)
        self.policy_version = 0
        self.controller = None
        self.target_preparer = None
        self.critic_control_coordinates = config['interaction'].get('critic_control_coordinates','raw-v1')
        self.critic_state_coordinates = config['interaction'].get('critic_state_coordinates', 'dynamics-v1')
        self.critic = TwinTeamCritic(rl['hidden_dim'], config['interaction']['relation_dim'],
                                   self.critic_control_coordinates, self.critic_state_coordinates,
                                   config['interaction'].get('critic_architecture', 'mlp')).to(self.device)
        if config['training'].get('critic_warmup_target') == 'complete_real_state_return':
            # First fit a state-value prior, rather than initializing control
            # sensitivities from one realized trajectory at each state.
            with torch.no_grad():
                for head in (self.critic.q1,self.critic.q2):
                    head.node[0].weight[:,7:10].zero_()
        self.target = copy.deepcopy(self.critic).requires_grad_(False)
        # Historical checkpoints used the supervised critic as its own teacher.
        # Keep that interpretation readable, but new runs bootstrap exclusively
        # from a separate critic fitted to real transitions. Auxiliary gradients
        # must never reach this critic or its Polyak target.
        self.bootstrap_source = config['interaction'].get('bootstrap_source', 'coupled')
        if self.bootstrap_source not in ('coupled', 'real_td'):
            raise ValueError('Unknown bootstrap source.')
        self.bootstrap_estimator = config['interaction'].get('bootstrap_estimator', 'min')
        if self.bootstrap_estimator not in ('min', 'mean'):
            raise ValueError('Unknown bootstrap estimator.')
        self.bootstrap_critic = copy.deepcopy(self.critic) if self.bootstrap_source == 'real_td' else None
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=rl.get('actor_learning_rate',rl['learning_rate']), eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=rl['learning_rate'], eps=1e-5)
        self.bootstrap_optimizer = (torch.optim.Adam(self.bootstrap_critic.parameters(),
            lr=rl['learning_rate'], eps=1e-5) if self.bootstrap_critic is not None else None)
        self.log_alpha = torch.tensor(math.log(.2), device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=rl.get('temperature_learning_rate',rl['learning_rate']))
        self.proposal_reference = None
        self.gate_reference = None
        self.set_stage('gate')

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    def set_stage(self, stage):
        if stage not in ('gate', 'joint'):
            raise ValueError('Unknown training stage.')
        if (stage == 'joint' and self.proposal_reference is None and
                self.config['interaction'].get('proposal_kl_weight', 0.) > 0.):
            # The gate endpoint defines the continuation's behavior prior.
            # Retain the learned proposals while the new joint critic learns
            # their local sensitivity; the interaction gate remains adaptable.
            self.proposal_reference = copy.deepcopy(self.actor).eval().requires_grad_(False)
        self.stage = stage
        self.actor.freeze_proposals(stage == 'gate')

    def proposal_kl(self, packet):
        if self.proposal_reference is None or self.stage != 'joint':
            return torch.zeros((), device=self.device)
        obs = packet['obs']
        mask = torch.ones(obs.shape[:2], dtype=torch.bool, device=obs.device)
        hidden = self.actor.encode(obs, mask)
        mean = self.actor.base.mean_layer(hidden)
        log_std = self.actor.base.log_std_layer(hidden).clamp(-20, 2)
        with torch.no_grad():
            anchor = self.proposal_reference
            reference_hidden = anchor.encode(obs, mask)
            reference_mean = anchor.base.mean_layer(reference_hidden)
            reference_log_std = anchor.base.log_std_layer(reference_hidden).clamp(-20, 2)
        # KL is unchanged by the common invertible tanh transformation.
        divergence = (reference_log_std-log_std + .5 *
            (torch.exp(2.*(log_std-reference_log_std)) +
             (mean-reference_mean).square()*torch.exp(-2.*reference_log_std)-1.))
        return divergence.sum(-1).mean()

    def initialize_gate_reference(self):
        if self.gate_reference is None and self.config['interaction'].get('gate_reference_weight', 0.) > 0.:
            self.gate_reference = copy.deepcopy(self.actor).eval().requires_grad_(False)

    def gate_reference_loss(self, packet):
        if self.config['interaction'].get('gate_reference_weight', 0.) == 0.:
            return torch.zeros((), device=self.device)
        if self.gate_reference is None:
            raise RuntimeError('Initialize the gate reference after loading the common source policy.')
        with torch.no_grad():
            reference, _ = policy_action(self.gate_reference, packet, deterministic=True)
        action, _ = policy_action(self.actor, packet, deterministic=True)
        return (action[..., 2]-reference[..., 2]).square().mean()

    @torch.no_grad()
    def choose_action(self, packet, deterministic=False, noise=None):
        p = {k: v.unsqueeze(0) for k, v in tensor_packet(packet, self.device).items()}
        if noise is not None:
            noise = torch.as_tensor(noise, dtype=torch.float32, device=self.device).unsqueeze(0)
        task_policy = self.entropy_objective == 'task-return-v4'
        action, logp = policy_action(self.actor, p, deterministic=deterministic or task_policy, noise=noise)
        if task_policy and not deterministic:
            # Uniform noise has a fixed, candidate-independent scale. Neither
            # deployment values nor gradients include its density or entropy.
            interaction = self.config['interaction']
            epsilon = 2.*torch.rand_like(action)-1. if noise is None else noise.clamp(-1., 1.)
            action[..., 2] = (action[..., 2] + interaction.get('gate_exploration', .025)*epsilon[..., 2]).clamp(0., 1.)
            if self.stage == 'joint':
                action[..., :2] = (action[..., :2] + interaction.get('proposal_exploration', .025)*epsilon[..., :2]).clamp(-1., 1.)
        return action[0].cpu().numpy(), float(logp.item())


    def learn(self, replay, auxiliary=None, *, diagnostics=False, update_actor=True):
        batch = replay.sample(self.batch_size, self.device)
        packet, following = current_and_next(batch)
        if self.config.get('prediction', {}).get('enabled'):
            if self.target_preparer is None or '_next_contexts' not in batch:
                raise RuntimeError('Predictive TD requires full controller contexts and current-policy planning.')
            following = self.target_preparer(following, batch['_next_contexts'], batch['done'])
        with torch.no_grad():
            next_action, next_logp = policy_action(self.actor, following, deterministic=self.entropy_objective == 'task-return-v4')
            q1, q2 = self.target(following, next_action)
            next_value = bootstrap_value(q1, q2, self.bootstrap_estimator)
            target = batch['reward'] + self.gamma * (1. - batch['done']) * (next_value - self.alpha * next_logp)
            bootstrap_disagreement = (q1 - q2).abs().mean()
            bootstrap_value_mean = next_value.mean()
        bootstrap_loss = self.fit_bootstrap(packet, batch['action'], target)
        q1, q2 = self.critic(packet, batch['action'])
        td_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        auxiliary_loss = torch.zeros((), device=self.device)
        kind = self.config['interaction']['auxiliary']
        if auxiliary is not None and kind != 'none':
            ap = tensor_packet(auxiliary, self.device)
            a = torch.as_tensor(auxiliary['action'], device=self.device)
            cf = torch.as_tensor(auxiliary['counterfactual'], device=self.device)
            g = torch.as_tensor(auxiliary['return'], device=self.device).detach()
            gc = torch.as_tensor(auxiliary['counterfactual_return'], device=self.device).detach()
            qa, qb = self.critic(ap, a), self.critic(ap, cf)
            for left, right in zip(qa, qb):
                if kind == 'difference':
                    auxiliary_loss = auxiliary_loss + F.mse_loss(left - right, g - gc)
                elif kind == 'value':
                    auxiliary_loss = auxiliary_loss + .5 * (F.mse_loss(left, g) + F.mse_loss(right, gc))
                else:
                    raise ValueError('Unknown auxiliary objective.')
        critic_loss = td_loss + self.config['interaction']['auxiliary_weight'] * auxiliary_loss
        if not torch.isfinite(critic_loss):
            raise FloatingPointError('Non-finite critic loss.')
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        checks = {}
        def gradient_norm(module):
            gradients = [p.grad.detach() for p in module.parameters() if p.grad is not None]
            if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError('Missing or non-finite training gradients.')
            return float(torch.sqrt(sum(g.square().sum() for g in gradients)))
        if diagnostics:
            checks.update(critic_gradient_norm=gradient_norm(self.critic),
                weighted_auxiliary_loss=float((self.config['interaction']['auxiliary_weight'] * auxiliary_loss).detach()),
                td_target_abs_max=float(target.abs().max()), bootstrap_disagreement=float(bootstrap_disagreement),
                bootstrap_value_mean=float(bootstrap_value_mean), entropy_cost_mean=float((self.alpha * next_logp).mean()))
            if bootstrap_loss is not None:
                checks['bootstrap_td_loss'] = bootstrap_loss
                checks['bootstrap_gradient_norm'] = gradient_norm(self.bootstrap_critic)
            if auxiliary is not None:
                checks['model_target_abs_max'] = float(max(abs(auxiliary['return']).max(),
                                                         abs(auxiliary['counterfactual_return']).max()))
                if 'difference_standard_error' in auxiliary:
                    checks['auxiliary_difference_standard_error']=float(np.mean(auxiliary['difference_standard_error']))
        self.critic_optimizer.step()
        if diagnostics and self.bootstrap_critic is not None:
            storage=lambda optimizer:{(str(v.device),v.data_ptr()) for values in optimizer.state.values()
                for v in values.values() if isinstance(v,torch.Tensor)}
            aliases=len(storage(self.critic_optimizer)&storage(self.bootstrap_optimizer))
            checks['bootstrap_optimizer_aliases']=aliases
            if aliases:
                raise RuntimeError('Main and bootstrap optimizers must have independent state storage.')
            if kind=='none':
                delta=max(float((left-right).detach().abs().max()) for left,right in
                          zip(self.critic.parameters(),self.bootstrap_critic.parameters()))
                checks['same_info_critic_max_difference']=delta
                if delta!=0.:
                    raise RuntimeError('Same-info main and bootstrap critics diverged despite identical objectives.')
        self.critic.requires_grad_(False)
        if not update_actor:
            actor_loss = torch.zeros((), device=self.device)
        else:
            action, logp = policy_action(self.actor, packet, deterministic=self.entropy_objective == 'task-return-v4')
            q1, q2 = self.critic(packet, action)
            actor_loss = (self.alpha * logp - torch.minimum(q1, q2)).mean()
            proposal_kl = self.proposal_kl(packet)
            actor_loss = actor_loss + self.config['interaction'].get('proposal_kl_weight', 0.) * proposal_kl
            reference_loss = self.gate_reference_loss(packet)
            actor_loss = actor_loss + self.config['interaction'].get('gate_reference_weight', 0.) * reference_loss
            if diagnostics:
                checks['proposal_kl'] = float(proposal_kl.detach())
                checks['gate_reference_mse'] = float(reference_loss.detach())
        if not torch.isfinite(actor_loss):
            raise FloatingPointError('Non-finite policy loss.')
        self.actor_optimizer.zero_grad(set_to_none=True)
        if update_actor:
            actor_loss.backward()
        if diagnostics and update_actor:
            checks['actor_gradient_norm'] = gradient_norm(self.actor)
            checks['adapter_gradient_norm'] = gradient_norm(self.actor.base.adap_layer)
            checks['candidate_gate_gradient_norm'] = gradient_norm(self.actor.relation_gate)
        if update_actor:
            self.actor_optimizer.step()
            self.policy_version += 1
        self.critic.requires_grad_(True)
        if (update_actor and self.entropy_objective != 'task-return-v4' and
                (self.stage == 'joint' or self.entropy_objective == 'stage-mean-v2')):
            target_entropy = (-2 if self.entropy_objective == 'proposal-mean-v3' else
                ((-1 if self.stage == 'gate' else -3)
                 if self.entropy_objective == 'stage-mean-v2' else -3 * action.shape[-2]))
            alpha_loss = -(self.log_alpha * (logp.detach() + target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()
        with torch.no_grad():
            source = self.bootstrap_critic if self.bootstrap_critic is not None else self.critic
            for target_parameter, parameter in zip(self.target.parameters(), source.parameters()):
                target_parameter.lerp_(parameter, self.tau)
        return dict(td_loss=float(td_loss.detach()), auxiliary_loss=float(auxiliary_loss.detach()),
                    actor_loss=float(actor_loss.detach()), actor_updated=bool(update_actor), critic_updated=True,
                    bootstrap_updated=self.bootstrap_critic is not None,
                    alpha=float(self.alpha), **checks)

    def initialize_critic_from_bootstrap(self):
        if self.bootstrap_critic is None:
            raise ValueError('A separate bootstrap critic is required.')
        self.critic.load_state_dict(self.bootstrap_critic.state_dict())
        # Adam load_state_dict can retain same-device tensor storage, including
        # CPU step counters for CUDA optimizers. Values must match, storage must not.
        self.critic_optimizer.load_state_dict(copy.deepcopy(self.bootstrap_optimizer.state_dict()))

    def fit_bootstrap(self, packet, action, target, *, state_only=False):
        """Only real-transition Bellman targets may update the bootstrap critic."""
        if self.bootstrap_critic is None:
            return None
        q1, q2 = self.bootstrap_critic(packet, action)
        loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite real-transition bootstrap loss.')
        self.bootstrap_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if state_only:
            for head in (self.bootstrap_critic.q1,self.bootstrap_critic.q2):
                head.node[0].weight.grad[:,7:10].zero_()
                if head.architecture == 'quadratic-control-v1':
                    for module in (head.unary_control, head.pair_control):
                        for parameter in module.parameters():
                            parameter.grad.zero_()
        self.bootstrap_optimizer.step()
        return float(loss.detach())

    def learn_bootstrap(self, replay):
        """Recondition a legacy teacher with fixed actor and real replay only."""
        if self.bootstrap_critic is None:
            raise ValueError('A separate real-transition bootstrap critic is required.')
        batch = replay.sample(self.batch_size, self.device)
        packet, following = current_and_next(batch)
        with torch.no_grad():
            action, logp = policy_action(self.actor, following, deterministic=self.entropy_objective == 'task-return-v4')
            q1, q2 = self.target(following, action)
            next_value = bootstrap_value(q1, q2, self.bootstrap_estimator)
            target = batch['reward'] + self.gamma * (1. - batch['done']) * (next_value - self.alpha * logp)
        loss = self.fit_bootstrap(packet, batch['action'], target)
        with torch.no_grad():
            for slow, current in zip(self.target.parameters(), self.bootstrap_critic.parameters()):
                slow.lerp_(current, self.tau)
        return dict(bootstrap_td_loss=loss, bootstrap_disagreement=float((q1-q2).abs().mean()),
                    bootstrap_value_mean=float(next_value.mean()), td_target_abs_max=float(target.abs().max()))

    def learn_bootstrap_returns(self, replay, returns):
        """Initialize values before policy learning, without a learned value tail."""
        index = np.random.randint(len(returns), size=self.batch_size)
        packet = {k: torch.as_tensor(replay.arrays[k][index], device=self.device) for k in (*PACKET_KEYS, *CONTROL_KEYS) if k in replay.arrays}
        action = torch.as_tensor(replay.arrays['action'][index], device=self.device)
        target = torch.as_tensor(returns[index, None], device=self.device)
        loss = self.fit_bootstrap(packet, action, target,
            state_only=self.config['training'].get('critic_warmup_target') == 'complete_real_state_return')
        with torch.no_grad():
            for slow, current in zip(self.target.parameters(), self.bootstrap_critic.parameters()):
                slow.lerp_(current, self.tau)
        return dict(bootstrap_mc_loss=loss, complete_return_samples=len(returns),
                    real_return_target_mean=float(target.mean()), real_return_target_abs_max=float(target.abs().max()))

    def state_dict(self):
        state = dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), target=self.target.state_dict(), policy_version=self.policy_version,
                    actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
                    log_alpha=self.log_alpha.detach(), alpha_optimizer=self.alpha_optimizer.state_dict(), stage=self.stage,
                    critic_control_coordinates=self.critic_control_coordinates,
                    critic_state_coordinates=self.critic_state_coordinates,
                    entropy_objective=self.entropy_objective, bootstrap_estimator=self.bootstrap_estimator)
        if self.config['interaction'].get('critic_architecture', 'mlp') != 'mlp':
            state['critic_architecture'] = self.config['interaction']['critic_architecture']
        if self.bootstrap_critic is not None:
            state.update(bootstrap_critic=self.bootstrap_critic.state_dict(),
                         bootstrap_optimizer=self.bootstrap_optimizer.state_dict())
        if self.proposal_reference is not None:
            state['proposal_reference'] = self.proposal_reference.state_dict()
        if self.gate_reference is not None:
            state['gate_reference'] = self.gate_reference.state_dict()
        return state

    def load_state_dict(self, state):
        self.policy_version = state.get('policy_version', 0)
        saved_architecture = state.get('critic_architecture',
            'quadratic-control-v1' if any(k.startswith('q1.unary_control.') for k in state['critic']) else 'mlp')
        if saved_architecture != self.config['interaction'].get('critic_architecture', 'mlp'):
            raise ValueError('Critic architecture changed; a fresh matched training run is required.')
        if state.get('critic_state_coordinates', 'dynamics-v1') != self.critic_state_coordinates:
            raise ValueError('Critic state coordinates changed; a fresh matched training run is required.')
        if state.get('bootstrap_estimator', 'min') != self.bootstrap_estimator:
            raise ValueError('Bootstrap estimator changed; a fresh matched training run is required.')
        if state.get('entropy_objective', 'joint-sum-v1') != self.entropy_objective:
            raise ValueError('Entropy objective changed; a fresh matched training run is required.')
        if state.get('critic_control_coordinates','raw-v1') != self.critic_control_coordinates:
            raise ValueError('Critic control coordinates changed; a fresh matched training run is required.')
        if (self.bootstrap_critic is not None) != ('bootstrap_critic' in state):
            raise ValueError('Bootstrap protocol changed; use explicit checkpoint reconditioning.')
        for name in ('actor', 'critic', 'target', 'actor_optimizer', 'critic_optimizer', 'alpha_optimizer'):
            value=copy.deepcopy(state[name]) if name.endswith('optimizer') else state[name]
            getattr(self, name).load_state_dict(value)
        if self.bootstrap_critic is not None:
            self.bootstrap_critic.load_state_dict(state['bootstrap_critic'])
            self.bootstrap_optimizer.load_state_dict(copy.deepcopy(state['bootstrap_optimizer']))
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'])
        self.set_stage(state['stage'])
        if 'proposal_reference' in state:
            if self.proposal_reference is None:
                raise ValueError('A proposal-reference checkpoint requires its original KL setting.')
            self.proposal_reference.load_state_dict(state['proposal_reference'])
        if 'gate_reference' in state:
            self.initialize_gate_reference()
            if self.gate_reference is None:
                raise ValueError('A gate-reference checkpoint requires its original regularization setting.')
            self.gate_reference.load_state_dict(state['gate_reference'])


class LegacyCBFSAC:
    """Released individual-critic SAC, with the common physical safety layer."""
    legacy = True

    def __init__(self, config, device='cpu'):
        from policy.SAC import SAC
        from utils.config import _dict_to_namespace
        self.config, self.device = copy.deepcopy(config), torch.device(device)
        self.sac = SAC(_dict_to_namespace(config), 6, 8, 3, adaptive=True, device=self.device)
        self.actor = self.sac.actor
        with torch.no_grad():
            self.sac.log_alpha.fill_(math.log(.2))
        self.sac.alpha = self.sac.log_alpha.exp().detach().to(self.device)
        self.set_stage('gate')

    def set_stage(self, stage):
        self.stage = stage
        self.sac.adaptive_alpha = stage == 'joint'
        for name, parameter in self.actor.named_parameters():
            if name.split('.')[0] in ('f1', 'f2', 't1', 'l1', 'l2', 'mean_layer', 'log_std_layer'):
                parameter.requires_grad_(stage == 'joint')

    def choose_action(self, packet, deterministic=False, noise=None):
        action = self.sac.choose_action(packet['obs'], deterministic).reshape(-1, 3)
        if not deterministic:
            action[:, 2] = np.clip(action[:, 2] + np.random.normal(0., .1, len(action)), 0., 1.)
        return action, 0.

    def learn(self, replay, auxiliary=None):
        class LocalView:
            def sample(self, batch_size, device):
                n = replay.arrays['action'].shape[1]
                index = np.random.randint(replay.size * n, size=batch_size)
                row, boat = index // n, index % n
                fields = replay.arrays
                data = (fields['obs'][row, boat], fields['action'][row, boat],
                        fields['individual_reward'][row, boat, None], fields['next_obs'][row, boat],
                        fields['done'][row])
                return tuple(torch.as_tensor(value, device=device) for value in data)
        self.sac.learn(LocalView())
        return dict(alpha=float(self.sac.alpha.detach()))

    def state_dict(self):
        return dict(actor=self.actor.state_dict(), critic=self.sac.critic.state_dict(),
                    target=self.sac.critic_target.state_dict(), actor_optimizer=self.sac.actor_optimizer.state_dict(),
                    critic_optimizer=self.sac.critic_optimizer.state_dict(), log_alpha=self.sac.log_alpha.detach(),
                    alpha_optimizer=self.sac.alpha_optimizer.state_dict(), stage=self.stage)

    def load_state_dict(self, state):
        mapping = dict(actor='actor', critic='critic', target='critic_target', actor_optimizer='actor_optimizer',
                       critic_optimizer='critic_optimizer', alpha_optimizer='alpha_optimizer')
        for source, target in mapping.items():
            getattr(self.sac, target).load_state_dict(state[source])
        with torch.no_grad():
            self.sac.log_alpha.copy_(state['log_alpha'].cpu())
        self.sac.alpha = self.sac.log_alpha.exp().detach().to(self.device)
        self.set_stage(state['stage'])


def make_agent(config, device='cpu'):
    return (LegacyCBFSAC if config['interaction']['arm'] == 'arboids_cbf' else InteractionSAC)(config, device)
