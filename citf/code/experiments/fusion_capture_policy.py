"""Candidate-conditioned distribution learning for covered joint pursuit.

Each configuration specifies both thruster fusion coefficients and feedback
rule targets. The same configuration executor is used by real branches and
deployment. Learned synchronous distance distributions directly rank the bank.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from citf.bridge import Controller
from citf.contracts import canonical_hash, file_hash, validate_predictor_protocol
from citf.model import Attention
from citf.risk import rectified_gaussian_risk
from experiments import batched_feedback as feedback
from experiments.contact_event_policy import retained_teacher
from experiments.interception_reserve import tracking_thrust
from experiments.joint_viability_projection import ViabilityController
from experiments.synchronized_configuration_policy import joint


@dataclass(frozen=True)
class FusionConfig:
    revision: str = 'CITF-joint-feedback-capture-distribution-v1'
    width: int = 64
    heads: int = 4
    history: int = 8
    horizon: int = 32
    decision_ticks: int = 5
    context: bool = True
    teacher_checkpoint: str = ''
    predictor: str = 'learned'
    collision_weight: float = 64.
    breach_weight: float = 64.
    capture_weight: float = 20.
    progress_weight: float = 1.
    failure_weight: float = 200.
    distribution: str = 'rectified_gaussian'

    def __post_init__(self):
        if min(self.width,self.heads,self.history,self.horizon,self.decision_ticks) < 1 or self.width % self.heads:
            raise ValueError('Fusion dimensions and decision period must be positive; width must divide into heads')
        if self.distribution not in ('rectified_gaussian','hurdle') or self.predictor not in ('learned','nominal'):
            raise ValueError('Unknown fusion distribution or predictor')


def feedback_rules(state, boids, parameters):
    """parameters = lead seconds, lateral m, radial m, guard flag, tracking flag."""
    enemy = state[:, -1, :2]
    radial = enemy / enemy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    velocity = state[:, -1, 3:5]
    heading = torch.where(velocity.norm(dim=-1, keepdim=True) > .3,
                          velocity / velocity.norm(dim=-1, keepdim=True).clamp_min(1e-6), radial)
    side = torch.stack((-heading[:, 1], heading[:, 0]), -1)
    waypoint = (enemy[:, None] + parameters[..., :1] * velocity[:, None]
                + parameters[..., 1:2] * side[:, None] + parameters[..., 2:3] * radial[:, None])
    waypoint = torch.where(parameters[..., 3:4] > .5, radial[:, None] * 12.5, waypoint)
    tracked = tracking_thrust(state, waypoint)
    return torch.where((parameters[..., 4:5] > .5) | (parameters[..., 3:4] > .5),
                       (tracked - 250) / 750, boids)


def configuration_bank(world, source, teacher, guard):
    _, trace = teacher.act(world)
    learned, boids, prior = source.commands(world.env)
    n = len(learned)
    gates = torch.stack([torch.as_tensor(trace['gates']), prior,
                        torch.ones_like(prior), torch.zeros_like(prior),
                        *[torch.zeros_like(prior) for _ in range(3)],
                        torch.full_like(prior, .5), torch.full_like(prior, .25),
                        torch.full_like(prior, .25), prior.clone(), prior.clone()])
    parameters = torch.zeros(12, n, 5)
    parameters[4:, :, 4] = 1
    parameters[5, :, 0] = 2
    parameters[6, :, 0] = 4
    parameters[7:, :, 0] = 2
    state = torch.as_tensor(world.state(), dtype=torch.float32)
    enemy = state[-1, :2]
    radial = enemy / enemy.norm().clamp_min(1e-6)
    side = torch.stack((-radial[1], radial[0]))
    lateral = (state[:-1, :2] - enemy) @ side
    eligible = [i for i in range(n) if i != guard]
    left = max(eligible, key=lambda i: float(lateral[i]))
    right = min(eligible, key=lambda i: float(lateral[i]))
    parameters[8, left, 1] = parameters[10, left, 1] = 8
    parameters[8, right, 1] = parameters[10, right, 1] = -8
    outer_point = enemy + radial * 12 + state[-1, 3:5] * 2
    outer = min(eligible, key=lambda i: float((state[i, :2] - outer_point).norm()))
    parameters[9, outer, 2] = 12
    distance = (state[:-1, :2] - enemy).norm(dim=-1)
    gates[11] = torch.sigmoid((distance - distance.median()) / 10)[:, None].expand(-1, 2)
    parameters[:, guard] = parameters.new_tensor([0, 0, 0, 1, 1])
    gates[:, guard] = 0
    return gates, parameters, learned, boids, trace


class PlanReference:
    def __init__(self, source):
        self.source = source
        self.gates = self.parameters = None

    def act(self, world):
        learned, boids, _ = self.source.commands(world.env)
        state = torch.as_tensor(world.state(), dtype=torch.float32)[None]
        rules = feedback_rules(state, boids[None], self.parameters[None])[0]
        force = 750 * (self.gates * learned + (1 - self.gates) * rules) + 250
        return force.numpy(), dict(forward_batches=0, evaluated_unique=1)


class ConfigurationExecutor(ViabilityController):
    """A shared executor; it does not query the selector or another policy."""
    def __init__(self, source):
        self.source = source
        self.teacher = PlanReference(source)
        self.model = SimpleNamespace(cfg=SimpleNamespace(guard_margin=6., guard_weight=16.))
        self.horizon, self.iterations = 10, 12

    def configure(self, gates, parameters):
        self.teacher.gates = gates.cpu()
        self.teacher.parameters = parameters.cpu()


class NominalStep(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, state, legacy, gates, parameters, override, first):
        observations, boids = feedback.source_observations(state, legacy)
        learned = feedback.source_actions(self.network, observations)
        rules = feedback_rules(state, boids, parameters)
        command = 750 * (gates * learned + (1 - gates) * rules) + 250
        command = torch.where(first > .5, override, command)
        thrust = torch.cat((command, feedback.attacker_thrust(state, 2.25)[:, None]), 1)
        return feedback.integrate(state, thrust)


def nominal_paths(world, source, gates, parameters, actual_first, horizon):
    """Known nominal feedback; future projection corrections are learned residuals."""
    c, n = gates.shape[:2]
    device = next(source.net.parameters()).device
    state = torch.as_tensor(world.state(), dtype=torch.float32, device=device)[None].expand(c, -1, -1).clone()
    gates, parameters, actual_first = (value.to(device) for value in (gates, parameters, actual_first))
    legacy = state[..., 3:5].clone()
    first = state.new_ones(c, 1, 1)
    key = (c, n, str(device))
    if not hasattr(source, '_fusion_nominal'):
        source._fusion_nominal = {}
    if key not in source._fusion_nominal:
        step = NominalStep(source.net).eval()
        # Compilation for the actual research workload; no synthetic validation run.
        source._fusion_nominal[key] = torch.jit.freeze(torch.jit.trace(
            step, (state, legacy, gates, parameters, actual_first, first), check_trace=False))
    step = source._fusion_nominal[key]
    paths = []
    for tick in range(horizon):
        state, legacy = step(state, legacy, gates, parameters, actual_first,
                             first if tick == 0 else torch.zeros_like(first))
        paths.append(state[..., :2])
    return torch.stack(paths, -2).cpu()


@torch.no_grad()
def make_input(world, source, teacher, executor, guard, cfg):
    gates, parameters, learned, boids, reference_trace = configuration_bank(world, source, teacher, guard)
    forces = []
    for gate, param in zip(gates, parameters):
        executor.configure(gate, param)
        force, _ = executor.act(world)
        forces.append(torch.as_tensor(force))
    forces = torch.stack(forces)
    horizon = min(cfg.horizon, int(math.ceil((world.protocol.time_limit-world.env.Current_T-1e-8)/.2)))
    nominal = nominal_paths(world, source, gates, parameters, forces, horizon)
    if horizon < cfg.horizon:
        nominal = torch.cat((nominal, nominal[..., -1:, :].expand(-1, -1, cfg.horizon-horizon, -1)), -2)
    history, position, history_valid, rotation = world.observation()
    rules = feedback_rules(torch.as_tensor(world.state(), dtype=torch.float32)[None].expand(12, -1, -1),
                           boids[None].expand(12, -1, -1), parameters)
    candidate = torch.cat((gates, learned[None].expand(12, -1, -1), rules,
                           (forces-250)/750, parameters / parameters.new_tensor([4, 12, 12, 1, 1])), -1)
    candidate = torch.cat((candidate, candidate.new_zeros(12, 1, 13)), 1)
    batch = dict(history=history[None], positions=position[None], history_valid=history_valid[None],
        entity_valid=torch.ones(1, len(position), dtype=torch.bool), candidate=candidate[None],
        nominal=(nominal @ torch.as_tensor(rotation, dtype=torch.float32))[None],
        known_time_valid=(torch.arange(cfg.horizon)<horizon)[None],
        time_context=torch.tensor([[max(0.,world.protocol.time_limit-world.env.Current_T)/world.protocol.time_limit,
                                    horizon/cfg.horizon]], dtype=torch.float32))
    return batch, gates, parameters, forces, rotation, reference_trace


def nominal_statistics(batch):
    nominal, known = batch['nominal'], batch['known_time_valid']
    entities = nominal.shape[2]
    i, j = torch.triu_indices(entities, entities, 1, device=nominal.device)
    distance = (nominal[:, :, i]-nominal[:, :, j]).norm(dim=-1)
    # Apply the public physical stopping rule to the nominal comparison too.
    radial = nominal[:, :, -1].norm(dim=-1)
    stop = ((distance[:, :, j < entities-1].amin(-2) < 5)
            | (distance[:, :, j == entities-1].amin(-2) < 5) | (radial < 15))
    previous_stop = F.pad(stop[..., :-1].long().cumsum(-1), (1, 0)) > 0
    valid = known[:, None] & ~previous_stop
    distance = distance.masked_fill(~valid[:, :, None], torch.inf)
    pair = distance.amin(-1)
    team_path = distance[:, :, j == entities-1].amin(-2)
    team, at = team_path.min(-1)
    radius = radial.masked_fill(~valid, torch.inf).amin(-1)
    return pair, team, radius, (at.float()+1)/known.sum(-1)[:, None], i, j


def hurdle_parameters(nominal, parameters, threshold):
    """A physical-event mixture: beta below the boundary, lognormal excess above."""
    event_logit=((threshold-nominal)/.7).clamp(-8,8)+parameters[...,0]
    probability=event_logit.sigmoid()
    alpha=F.softplus(parameters[...,1])+.1
    beta=F.softplus(parameters[...,2])+.1
    logmu=((nominal-threshold).clamp_min(1).log()+parameters[...,3]).clamp(-5,7)
    scale=(F.softplus(parameters[...,4])+.05).clamp_max(1.5)
    excess=(logmu+.5*scale.square()).exp()
    total=alpha+beta
    inside_mean=threshold*alpha/total
    deficit_second=threshold**2*beta*(beta+1)/(total*(total+1))
    mean=probability*inside_mean+(1-probability)*(threshold+excess)
    second=(probability*threshold**2*alpha*(alpha+1)/(total*(total+1))
            +(1-probability)*(threshold**2+2*threshold*excess+(2*logmu+2*scale.square()).exp()))
    return dict(mu=mean,sigma=(second-mean.square()).clamp_min(1e-6).sqrt(),
        probability=probability,deficit_second=deficit_second,excess_mean=excess,
        event_logit=event_logit,alpha=alpha,beta=beta,excess_logmu=logmu,excess_sigma=scale)


def capture_probability(output):
    if 'team_probability' in output:return output['team_probability']
    return torch.special.ndtr((5-output['team_mu'])/output['team_sigma'])


class FusionOutcomeTransformer(nn.Module):
    def __init__(self, cfg=FusionConfig()):
        super().__init__();self.cfg=cfg;w=cfg.width
        self.history_embed=nn.Linear(8,w)
        self.temporal_position=nn.Parameter(torch.randn(cfg.history,w)*.01)
        self.temporal=nn.ModuleList([Attention(w,cfg.heads),Attention(w,cfg.heads)])
        self.history_context=Attention(w,cfg.heads)
        self.time_embed=nn.Linear(2,w)
        self.action_embed=nn.Sequential(nn.Linear(13+2*cfg.horizon,w),nn.GELU(),nn.Linear(w,w))
        self.context=nn.ModuleList([Attention(w,cfg.heads,True),Attention(w,cfg.heads,True)])
        self.pair=nn.Sequential(nn.Linear(2*w+3,w),nn.GELU(),nn.Linear(w,5 if cfg.distribution=='hurdle' else 2))
        self.team=nn.Sequential(nn.Linear(3*w+3,w),nn.GELU(),nn.Linear(w,11 if cfg.distribution=='hurdle' else 5))
        self.motion=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,2*cfg.horizon))
        for head in [self.pair[-1],self.team[-1],self.motion[-1]]:
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        with torch.no_grad():
            if cfg.distribution=='hurdle':
                self.pair[-1].bias[[1,2,4]]=torch.tensor([9.,.5413,-2.])
                self.team[-1].bias[[1,2,4,6,7,9]]=torch.tensor([9.,.5413,-2.,9.,.5413,-2.])
            else:
                self.pair[-1].bias[1]=-1.5;self.team[-1].bias[[1,3]]=-1.5

    def validate_runtime(self,source,world):
        identity=getattr(self,'source_identity',{})
        for key,actual in [('source_sha256',source.sha256),
                           ('source_implementation_sha256',source.implementation_sha256)]:
            if identity.get(key) and identity[key] != actual:
                raise ValueError('Fusion predictor and runtime source differ: '+key)
        if world is not None:
            if world.protocol.history_steps != self.cfg.history:
                raise ValueError('Fusion runtime history differs from the trained model')
            if hasattr(self,'physical_protocol'):
                validate_predictor_protocol(self.physical_protocol,world.protocol)
            if identity.get('environment_sha256') and identity['environment_sha256'] != world.environment_sha256:
                raise ValueError('Fusion predictor and runtime environment implementation differ')

    def forward(self,batch):
        history=batch['history'];B,E,L,_=history.shape
        valid=batch['entity_valid'];hv=batch['history_valid'].reshape(B*E,L)
        h=(self.history_embed(history)+self.temporal_position).reshape(B*E,L,-1)
        for layer in self.temporal:h=layer(h,hv,causal=True)
        h=h[:,-1].reshape(B,E,-1)
        if self.cfg.context:h=self.history_context(h,valid)
        h=h+self.time_embed(batch['time_context'])[:,None]
        nominal=batch['nominal'];C=nominal.shape[1];known=batch['known_time_valid']
        displacement=(nominal-batch['positions'][:,None,:,None])/20
        displacement=displacement*known[:,None,None,:,None]
        z=(h[:,None]+self.action_embed(torch.cat((batch['candidate'],displacement.flatten(-2)),-1))).reshape(B*C,E,-1)
        position=batch['positions'][:,None].expand(-1,C,-1,-1).reshape(B*C,E,2)
        delta=position[:,:,None]-position[:,None,:];radius=position.norm(dim=-1)
        edge=torch.cat((delta/20,(delta.norm(dim=-1)/20)[...,None],
                         ((radius[:,:,None]-radius[:,None,:])/20)[...,None],
                         torch.ones_like(delta[...,:1])),-1)
        mask=valid[:,None].expand(-1,C,-1).reshape(B*C,E)
        if self.cfg.context:
            for layer in self.context:z=layer(z,mask,edge)
        z=z.reshape(B,C,E,-1)
        nom_pair,nom_team,nom_radius,nom_time,i,j=nominal_statistics(batch)
        pair_features=torch.cat((z[:,:,i]+z[:,:,j],(z[:,:,i]-z[:,:,j]).abs(),
             (nom_pair/20)[...,None],(batch['positions'][:,i]-batch['positions'][:,j]).norm(dim=-1)[:,None,:,None].expand(-1,C,-1,-1)/20,
             (j==E-1)[None,None,:,None].expand(B,C,-1,-1).to(z.dtype)),-1)
        q=self.pair(pair_features)
        pooled=torch.cat((z[:,:,:-1].mean(-2),z[:,:,:-1].amax(-2),z[:,:,-1],
                          torch.stack((nom_team/20,nom_radius/60,nom_time),-1)),-1)
        team=self.team(pooled)
        if self.cfg.distribution=='hurdle':
            pair_distribution=hurdle_parameters(nom_pair,q,5.)
            predicted_time=torch.sigmoid(torch.logit(nom_time.clamp(.02,.98))+team[...,10])
            if getattr(self.cfg,'time_prediction','nominal_window')=='remaining_task':
                remaining=batch['time_context'][:,0,None]
                predicted_time=1-remaining+remaining*team[...,10].sigmoid()
            output={name+'_'+key:value for name,values in [
                ('pair',pair_distribution),('team',hurdle_parameters(nom_team,team[...,:5],5.)),
                ('radius',hurdle_parameters(nom_radius,team[...,5:10],15.))] for key,value in values.items()}
            output.update(mu=pair_distribution['mu'],sigma=pair_distribution['sigma'],
                team_parameters=team,
                configuration_embedding=pooled,state_embedding=h.mean(-2),
                time_fraction=predicted_time,
                xy=nominal+6*torch.tanh(self.motion(z).reshape(B,C,E,self.cfg.horizon,2)))
            return output
        time_fraction=torch.sigmoid(torch.logit(nom_time.clamp(.02,.98))+team[...,4])
        return dict(mu=nom_pair+q[...,0],sigma=(F.softplus(q[...,1])+.05).clamp_max(8),
            configuration_embedding=pooled,state_embedding=h.mean(-2),
            team_mu=nom_team+team[...,0],team_sigma=(F.softplus(team[...,1])+.05).clamp_max(8),
            radius_mu=nom_radius+team[...,2],radius_sigma=(F.softplus(team[...,3])+.05).clamp_max(8),
            time_fraction=time_fraction,xy=nominal+6*torch.tanh(self.motion(z).reshape(B,C,E,self.cfg.horizon,2)))


def physical_output(batch):
    pair,team,radius,at,_,_=nominal_statistics(batch)
    return dict(mu=pair,sigma=torch.full_like(pair,.25),team_mu=team,team_sigma=torch.full_like(team,.25),
                radius_mu=radius,radius_sigma=torch.full_like(radius,.25),time_fraction=at,xy=batch['nominal'])


def control_score(output,cfg,entities):
    _,j=torch.triu_indices(entities,entities,1,device=output['mu'].device)
    if 'pair_probability' in output:
        collision=(output['pair_probability'][...,j<entities-1]*output['pair_deficit_second'][...,j<entities-1]).sum(-1)
        breach=output['radius_probability']*output['radius_deficit_second']
        failure=output['pair_probability'][...,j<entities-1].sum(-1)+output['radius_probability']
        probability=output['team_probability']
        progress=(1-probability)*output['team_excess_mean']
        return (cfg.collision_weight*collision+cfg.breach_weight*breach+cfg.failure_weight*failure
                +cfg.progress_weight*progress-cfg.capture_weight*probability+2*probability*output['time_fraction'])
    collision=rectified_gaussian_risk(output['mu'][...,j<entities-1],output['sigma'][...,j<entities-1],5.).sum(-1)
    breach=rectified_gaussian_risk(output['radius_mu'],output['radius_sigma'],15.)
    probability=torch.special.ndtr((5-output['team_mu'])/output['team_sigma'])
    failure=(torch.special.ndtr((5-output['mu'][...,j<entities-1])/output['sigma'][...,j<entities-1]).sum(-1)
             +torch.special.ndtr((15-output['radius_mu'])/output['radius_sigma']))
    gap=output['team_mu']-5
    scaled=gap/output['team_sigma']
    progress=(gap*torch.special.ndtr(scaled)
              +output['team_sigma']*torch.exp(-.5*scaled.square())/math.sqrt(2*math.pi))
    return (cfg.collision_weight*collision+cfg.breach_weight*breach+cfg.failure_weight*failure
            +cfg.progress_weight*progress-cfg.capture_weight*probability+2*probability*output['time_fraction'])


def load_policy(path,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    model=FusionOutcomeTransformer(FusionConfig(**payload['config']))
    model.load_state_dict(payload['state']);model.physical_protocol=payload['physical_protocol']
    model.teacher_sha256=payload.get('teacher_sha256','')
    model.source_identity=payload.get('source_identity',{})
    return model.to(device).eval()


class FusionCaptureController(Controller):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cfg=self.model.cfg
        teacher=retained_teacher(self.cfg.teacher_checkpoint,getattr(self.model,'teacher_sha256',''))
        for key,actual in [('source_sha256',self.source.sha256),
                           ('source_implementation_sha256',self.source.implementation_sha256)]:
            if teacher.checkpoint_protocol.get(key) != actual:
                raise ValueError('Fusion teacher and runtime source differ: '+key)
        self.teacher=joint.JointPolicyController(self.source,teacher,'cpu',12,96,1)
        self.executor=ConfigurationExecutor(self.source)
        self.guard=None;self.last_time=None;self.ticks=0;self.current=None

    def act(self,world):
        self.model.validate_runtime(self.source,world)
        started=time.perf_counter();now=world.env.Current_T
        if self.guard is None or self.last_time is not None and now<=self.last_time:
            state=world.state();enemy=state[-1,:2];target=enemy/max(np.linalg.norm(enemy),1e-6)*12.5
            self.guard=int(np.linalg.norm(state[:-1,:2]-target,axis=-1).argmin());self.ticks=0
        if self.ticks%self.cfg.decision_ticks==0:
            batch,gates,parameters,forces,_,_=make_input(world,self.source,self.teacher,self.executor,self.guard,self.cfg)
            with torch.no_grad():
                if self.cfg.predictor=='nominal':output=physical_output(batch)
                else:output=self.model({k:v.to(self.device) for k,v in batch.items()})
                scores=control_score(output,self.cfg,len(world.state()))[0].cpu()
            choice=int(scores.argmin());self.current=(gates[choice],parameters[choice]);force=forces[choice].numpy()
            self.decision=dict(evaluated_unique=len(gates),forward_batches=1,selected_configuration=choice,
                scores=scores.tolist(),gates=gates[choice].tolist(),feedback_parameters=parameters[choice].tolist(),
                predicted_team_distance=output['team_mu'][0].cpu().tolist(),
                predicted_capture_probability=capture_probability(output)[0].cpu().tolist(),
                candidate_first_forces=forces.tolist(),coverage_guard=self.guard,predictor=self.cfg.predictor)
            trace=dict(self.decision,replanned=True)
        else:
            self.executor.configure(*self.current);force,trace=self.executor.act(world)
            projection_calls=trace['forward_batches']
            trace.update(self.decision,replanned=False,evaluated_unique=0,forward_batches=projection_calls)
        self.last_time=now;self.ticks+=1
        trace.update(revision=self.cfg.revision,physical_thrust=force.tolist(),compute_seconds=time.perf_counter()-started)
        return force,trace


def main():
    import citf.evaluate as evaluation
    original=evaluation.runtime_identity
    def identity(*args,**kwargs):
        value=original(*args,**kwargs)
        checkpoint=args[2] if len(args)>2 else kwargs['checkpoint']
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        value.update(method='CITF-joint-fusion-capture-distribution-development',revision=payload['config']['revision'])
        value['controller'].update(configuration=payload['config'],
            action='joint left/right learned-source and feedback-rule fusion; persistent inner support; common joint projection',
            prediction='candidate Transformer predicts all synchronous pair minima,team encounter minimum,target radius minimum and contact time',
            readout='predicted distance distributions directly determine every bank score; no independent action-logit head',
            supervision='real feedback branches with common futures,actual stopping and actual projected first command',
            nominal='fixed2.25 observed APF feedback; projection residual learned; true agility and future actions are labels only')
        value['runtime_code']=canonical_hash(dict(core=value['runtime_code'],fusion_capture=file_hash(__file__)))
        return value
    def load(checkpoint,source,world,device):
        model=load_policy(checkpoint,device)
        validate_predictor_protocol(model.physical_protocol,world.protocol)
        return model
    evaluation.runtime_identity=identity;evaluation.load_behavior=load;evaluation.Controller=FusionCaptureController
    evaluation.main()


if __name__=='__main__':main()
