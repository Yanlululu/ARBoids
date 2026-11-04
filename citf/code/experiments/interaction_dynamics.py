"""Observed-state interaction dynamics for current-configuration planning.

The attacker moves before defenders in the real environment. Consequently its
current propulsion prediction reads only joint observed history. A candidate
changes its later responses through the rolled joint state, never by exposing
future attacker actions. Executed attacker propulsion is an auxiliary label.
"""
import argparse
from dataclasses import asdict, dataclass
import datetime
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from citf.contracts import (Protocol, canonical_hash, immutable_run_contract,
    require_same_contract, validate_task_banks, validate_predictor_protocol)
from citf.train import atomic_save
from experiments.batched_feedback import integrate
from experiments.interaction_trajectories import atomic_json, load_collection


@dataclass(frozen=True)
class DynamicsConfig:
    revision: str = 'CITF-observed-interaction-dynamics-v1'
    history: int = 8
    width: int = 64
    layers: int = 2
    heads: int = 4
    horizon: int = 16
    dt: float = .2
    context: bool = True
    geometry_training: bool = True


class InteractionLayer(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.relative = nn.Sequential(nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, heads))
        self.output = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))

    def forward(self, value, relative, context):
        batch, entities, width = value.shape
        q, k, v = self.qkv(self.norm1(value)).reshape(batch, entities, 3, self.heads, width // self.heads).unbind(2)
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
        logits = q @ k.transpose(-1, -2) / math.sqrt(width // self.heads)
        logits = logits + self.relative(relative).permute(0, 3, 1, 2)
        if not context:
            diagonal = torch.eye(entities, dtype=torch.bool, device=value.device)
            logits = logits.masked_fill(~diagonal[None, None], -torch.inf)
        attended = (logits.softmax(-1) @ v).transpose(1, 2).reshape(batch, entities, width)
        value = value + self.output(attended)
        return value + self.ff(self.norm2(value))


class InteractionDynamics(nn.Module):
    def __init__(self, cfg=DynamicsConfig()):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Sequential(nn.Linear(cfg.history * 8 + 8, cfg.width), nn.SiLU(), nn.Linear(cfg.width, cfg.width))
        self.layers = nn.ModuleList([InteractionLayer(cfg.width, cfg.heads) for _ in range(cfg.layers)])
        self.attacker = nn.Sequential(nn.LayerNorm(cfg.width), nn.Linear(cfg.width, cfg.width), nn.SiLU(), nn.Linear(cfg.width, 4))
        self.residual = nn.Sequential(nn.Linear(cfg.width + 2, cfg.width), nn.SiLU(), nn.Linear(cfg.width, 6))
        nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)
        self.register_buffer('residual_scale', torch.tensor([.10, .10, .05, .5, .5, .15]))
        self.training_contract = None

    def validate_runtime(self, source, world):
        if self.training_contract is None:
            raise ValueError('Dynamics checkpoint lacks a recorded training contract; retain legacy evaluations separately')
        parameters = self.training_contract['parameters']
        if source is not None:
            expected = parameters['source']
            if (expected['source_sha256'] != source.sha256
                    or expected['source_implementation_sha256'] != source.implementation_sha256):
                raise ValueError('Dynamics checkpoint and online source differ')
        if world is not None:
            if parameters['source']['environment_sha256'] != world.environment_sha256:
                raise ValueError('Dynamics checkpoint and online environment differ')
            validate_predictor_protocol(parameters['physical_protocol'], world.protocol)

    def encode(self, history, remaining):
        batch, length, entities, _ = history.shape
        angle = torch.atan2(-history[:, -1, :, 1], -history[:, -1, :, 0])
        if self.cfg.context:
            angle = angle[:, -1:].expand(-1, entities)
        # The no-context control must also choose its frame from its own state.
        c, s = angle.cos()[:, None, :], angle.sin()[:, None, :]
        x, y = history[..., 0], history[..., 1]
        vx, vy = history[..., 3], history[..., 4]
        yaw = history[..., 2] - angle[:, None, :]
        role = torch.zeros_like(x); role[:, :, -1] = 1
        features = torch.stack(((c*x+s*y)/60, (-s*x+c*y)/60,
            (c*vx+s*vy)/5, (-s*vx+c*vy)/5, yaw.sin(), yaw.cos(), history[..., 5], role), -1)
        position = features[:, -1, :, :2] * 60
        delta = position - position[:, -1:]
        distance = delta.norm(dim=-1)
        index = torch.arange(entities, device=history.device, dtype=history.dtype)[None].expand(batch, -1)
        extra = torch.stack((delta[..., 0]/20, delta[..., 1]/20, distance/20, 1/(distance+1),
            (index/8).sin(), (index/8).cos(), remaining[:, None].expand(-1, entities)/80,
            (remaining[:, None].expand(-1, entities)/3.2).clamp_max(1)), -1)
        if not self.cfg.context:
            # Other entities cannot enter a token indirectly through target-relative features.
            extra = extra.clone(); extra[..., :4] = 0
        value = self.embedding(torch.cat((features.transpose(1, 2).flatten(2), extra), -1))
        pair_delta = position[:, :, None] - position[:, None, :]
        pair_distance = pair_delta.norm(dim=-1)
        radius = position.norm(dim=-1)
        relative = torch.cat((pair_delta/20, (pair_distance/20)[..., None],
            (1/(pair_distance+1))[..., None], ((radius[:, :, None]-radius[:, None, :])/20)[..., None]), -1)
        for layer in self.layers:
            value = layer(value, relative, self.cfg.context)
        return value

    def forward(self, history, defender_command, remaining, attacker_noise=None):
        encoded = self.encode(history, remaining)
        raw = self.attacker(encoded[:, -1])
        attacker_mean = 750 + 2250 * raw[:, :2].tanh()
        attacker_log_std = (-1.5 + raw[:, 2:]).clamp(-4, .4)
        attacker_command = attacker_mean
        if attacker_noise is not None:
            attacker_command = (attacker_mean + 2500 * attacker_log_std.exp() * attacker_noise).clamp(-1500, 3000)
        command = torch.cat((defender_command, attacker_command[:, None]), 1)
        state, legacy_velocity = integrate(history[:, -1], command)
        local = self.residual(torch.cat((encoded, command/2500), -1)).tanh() * self.residual_scale
        yaw = history[:, -1, :, 2]
        c, s = yaw.cos(), yaw.sin()
        residual = torch.stack((c*local[..., 0]-s*local[..., 1], s*local[..., 0]+c*local[..., 1],
            local[..., 2], c*local[..., 3]-s*local[..., 4], s*local[..., 3]+c*local[..., 4], local[..., 5]), -1)
        state = state + residual
        # Source observations expect the physical heading in [0, 2*pi).
        state = torch.cat((state[..., :2], state[..., 2:3].remainder(2 * math.pi), state[..., 3:]), -1)
        # Use the matching correction for legacy velocity passed to frozen source feedback.
        legacy_velocity = legacy_velocity + residual[..., 3:5]
        return dict(state=state, legacy_velocity=legacy_velocity,
                    attacker_mean=attacker_mean, attacker_log_std=attacker_log_std)


def state_error(predicted, target):
    error = predicted - target
    angular = torch.atan2(error[..., 2].sin(), error[..., 2].cos())
    return torch.cat((error[..., :2], angular[..., None], error[..., 3:]), -1)


def geometry(state):
    position = state[..., :2]
    distance = (position[..., :-1, :] - position[..., -1:, :]).norm(dim=-1)
    radius = position.norm(dim=-1)
    radial = radius[..., :-1] - radius[..., -1:]
    n = position.shape[-2] - 1
    i, j = torch.triu_indices(n, n, 1, device=state.device)
    pair = (position[..., i, :] - position[..., j, :]).norm(dim=-1)
    return distance, radial, pair


class Trajectories:
    def __init__(self, directory, device, role, bundle=None):
        directory = Path(directory)
        protocol, report, self.rows = bundle if bundle is not None else load_collection(directory, role)
        self.collection_contract = protocol['collection_contract']
        self.bank = self.collection_contract['parameters']['task_bank']
        validate_task_banks((role, self.bank))
        self.protocol = Protocol(**self.bank['protocol'])
        self.data_identity = canonical_hash(dict(contract=self.collection_contract, episodes=self.rows))
        loaded = [np.load(directory/'episodes'/row['episode']) for row in self.rows]
        lengths = np.asarray([len(x['command']) for x in loaded], np.int64)
        entities, longest = loaded[0]['state'].shape[1], int(lengths.max())
        states = np.empty((len(loaded), longest+1, entities, 6), np.float32)
        commands = np.zeros((len(loaded), longest, entities-1, 2), np.float32)
        attacker = np.zeros((len(loaded), longest, 2), np.float32)
        for i, data in enumerate(loaded):
            n = lengths[i]
            if (n != self.rows[i]['steps'] or data['state'].shape != (n+1, entities, 6)
                    or data['command'].shape != (n, entities-1, 2)
                    or data['attacker_thrust_label'].shape != (n, 2)
                    or entities != self.rows[i]['task']['defenders'] + 1):
                raise ValueError('Trajectory payload dimensions differ from its committed metadata')
            if any(not np.isfinite(data[key]).all() for key in ('state', 'command', 'attacker_thrust_label')):
                raise ValueError('Nonfinite observed trajectory')
            states[i, :n+1] = data['state']; states[i, n+1:] = data['state'][-1]
            commands[i, :n] = data['command']; attacker[i, :n] = data['attacker_thrust_label']
            data.close()
        self.state = torch.tensor(states, device=device)
        self.command = torch.tensor(commands, device=device)
        self.attacker = torch.tensor(attacker, device=device)
        self.lengths = torch.tensor(lengths, device=device)
        self.roots = torch.tensor([(i,t) for i,n in enumerate(lengths) for t in range(n)], device=device)
        self.report, self.device = report, device

    def remaining(self, tick):
        return (self.protocol.time_limit - self.protocol.control_dt * tick).clamp_min(0)

    def sample(self, count, generator, history=8, roots=None):
        if roots is None:
            indices = torch.randint(len(self.roots), (count,), generator=generator, device=self.device)
            roots = self.roots[indices]
        episode, tick = roots.unbind(-1)
        at = (tick[:, None] + torch.arange(-history+1, 1, device=self.device)[None]).clamp_min(0)
        hist = self.state[episode[:, None], at]
        return episode, tick, hist


def loss_one(model, data, sample):
    episode, tick, history = sample
    result = model(history, data.command[episode, tick], data.remaining(tick))
    scale = history.new_tensor([.08, .08, .03, .3, .3, .08])
    error = state_error(result['state'], data.state[episode, tick+1])/scale
    state_loss = F.smooth_l1_loss(error, torch.zeros_like(error))
    action_error = (result['attacker_mean'] - data.attacker[episode, tick])/2500
    log_std = result['attacker_log_std']
    propulsion = (.5*(action_error/log_std.exp()).square() + log_std).mean()
    return state_loss + .15*propulsion + .5*action_error.square().mean(), dict(
        state=float(state_loss.detach()), propulsion=float(propulsion.detach()),
        attacker_thrust_mae=float(action_error.detach().abs().mean()*2500))


def loss_rollout(model, data, sample, horizon=8):
    episode, tick, history = sample
    total = history.new_zeros(())
    for step in range(horizon):
        at = (tick+step).clamp_max(data.command.shape[1]-1)
        valid = tick+step < data.lengths[episode]
        result = model(history, data.command[episode, at], data.remaining(tick+step))
        predicted = result['state']; target = data.state[episode, at+1]
        scale = history.new_tensor([.15+.1*step, .15+.1*step, .08+.02*step, .7, .7, .2])
        loss = F.smooth_l1_loss(state_error(predicted, target)/scale,
                              torch.zeros_like(predicted), reduction='none').mean((1, 2))
        if model.cfg.geometry_training:
            pg, tg = geometry(predicted), geometry(target)
            close = (tg[0].amin(-1)<15).to(loss.dtype)
            geometry_loss = sum(F.smooth_l1_loss(p, t, reduction='none').mean(-1) for p,t in zip(pg,tg))
            loss = loss + .5*(1+2*close)*geometry_loss
        total = total + (loss*valid).sum()/valid.sum().clamp_min(1)
        history = torch.cat((history[:, 1:], predicted[:, None]), 1)
    return total/horizon


@torch.no_grad()
def prediction_metrics(model, data):
    model.eval()
    count, position, velocity, thrust = 0, 0., 0., 0.
    for roots in data.roots.split(512):
        episode, tick, history = data.sample(len(roots), None, model.cfg.history, roots)
        result = model(history, data.command[episode, tick], data.remaining(tick))
        error = state_error(result['state'], data.state[episode, tick+1])
        position += float(error[..., :2].norm(dim=-1).mean(-1).sum())
        velocity += float(error[..., 3:5].norm(dim=-1).mean(-1).sum())
        thrust += float((result['attacker_mean']-data.attacker[episode,tick]).abs().mean(-1).sum())
        count += len(roots)
    selected = data.roots[::16]
    rollout_position, rollout_geometry, valid_total = 0., 0., 0
    for roots in selected.split(128):
        episode, tick, history = data.sample(len(roots), None, model.cfg.history, roots)
        for step in range(model.cfg.horizon):
            at = (tick+step).clamp_max(data.command.shape[1]-1)
            valid = tick+step < data.lengths[episode]
            result = model(history, data.command[episode,at], data.remaining(tick+step))
            predicted, target = result['state'], data.state[episode,at+1]
            if step == model.cfg.horizon - 1:
                rollout_position += float(((predicted[...,:2]-target[...,:2]).norm(dim=-1).mean(-1)*valid).sum())
                pg,tg=geometry(predicted),geometry(target)
                rollout_geometry += float((sum((p-t).abs().mean(-1) for p,t in zip(pg,tg))/3*valid).sum())
                valid_total += int(valid.sum())
            history = torch.cat((history[:,1:],predicted[:,None]),1)
    model.train()
    return dict(transitions=count,position_error_m=position/count,velocity_error_mps=velocity/count,
        attacker_thrust_mae_N=thrust/count,rollout_roots=valid_total,
        **{f'rollout{model.cfg.horizon}_position_error_m': rollout_position/max(1,valid_total),
           f'rollout{model.cfg.horizon}_geometry_mae_m': rollout_geometry/max(1,valid_total)},
        scope='held-out trajectories; real defender commands condition open-loop physical prediction; not task performance')


def load(path, device='cpu', source=None, world=None):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = InteractionDynamics(DynamicsConfig(**checkpoint['config'])).to(device)
    model.training_contract = checkpoint.get('training_contract')
    if source is not None or world is not None:
        model.validate_runtime(source, world)
    model.load_state_dict(checkpoint['state']); model.eval()
    return model


def main():
    parser=argparse.ArgumentParser()
    for key in ('train','validation','out'):
        parser.add_argument('--'+key,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--seed',type=int,default=9201)
    parser.add_argument('--updates',type=int,default=3000)
    parser.add_argument('--batch',type=int,default=256)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--no-context',action='store_true')
    parser.add_argument('--ordinary',action='store_true')
    args=parser.parse_args(); out=Path(args.out)
    if args.updates < 1 or args.batch < 4:
        raise ValueError('Positive updates and at least four samples per mixed training batch required')
    bundles = [load_collection(path, role) for path, role in
               ((args.train, 'train'), (args.validation, 'validation'))]
    contracts = [bundle[0]['collection_contract'] for bundle in bundles]
    validate_task_banks(*[(role, contract['parameters']['task_bank']) for role, contract in
                         zip(('train', 'validation'), contracts)])
    for key in ('source', 'behavior_config'):
        require_same_contract(contracts[0]['parameters'][key], contracts[1]['parameters'][key], 'Trajectory ' + key)
    require_same_contract(contracts[0]['code'], contracts[1]['code'], 'Trajectory collector implementation')
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
    generator=torch.Generator(device=args.device).manual_seed(args.seed+101)
    train=Trajectories(args.train,args.device,'train',bundles[0])
    validation=Trajectories(args.validation,args.device,'validation',bundles[1])
    cfg=DynamicsConfig(context=not args.no_context,geometry_training=not args.ordinary,
        history=train.protocol.history_steps,horizon=train.protocol.prediction_steps,dt=train.protocol.control_dt)
    training_contract=immutable_run_contract(dict(arguments=vars(args),config=asdict(cfg),
        source=contracts[0]['parameters']['source'],physical_protocol=asdict(train.protocol),
        train=train.data_identity,validation=validation.data_identity), code=(__file__,))
    out.mkdir(parents=True,exist_ok=False)
    model=InteractionDynamics(cfg).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    protocol=dict(arguments=vars(args),config=asdict(cfg),train=train.report,validation=validation.report,
        training_contract=training_contract,
        stages=[1000,2000,3000],auxiliary_label='executed physical attacker thrust, never online input',
        optimizer='three one-step updates followed by one eight-step open-loop geometry update',
        control_selection='independent256 tasks, successes/collisions/capped capture/earlier stage',
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic_json(protocol,out/'protocol.json'); started=time.time()
    with (out/'learning.jsonl').open('a',encoding='utf-8') as log:
        for update in range(1,args.updates+1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            if update%4:
                loss,detail=loss_one(model,train,train.sample(args.batch,generator,cfg.history))
            else:
                loss=loss_rollout(model,train,train.sample(args.batch//4,generator,cfg.history))
                detail=dict(rollout_loss=float(loss.detach()))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite interaction-model research objective')
            loss.backward(); gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            if update%100==0:
                row=dict(update=update,loss=float(loss.detach()),gradient=float(gradient),
                    elapsed_seconds=time.time()-started,**detail)
                log.write(json.dumps(row)+'\n'); log.flush(); atomic_json(dict(completed=False,**row),out/'status.json')
                print(json.dumps(row),flush=True)
            if update%1000==0 or update==args.updates:
                metrics=prediction_metrics(model,validation)
                payload=dict(state=model.state_dict(),config=asdict(cfg),update=update,optimizer=optimizer.state_dict(),
                    training_contract=training_contract,
                    torch_rng=torch.get_rng_state(),sample_rng=generator.get_state(),arguments=vars(args),validation=metrics)
                atomic_save(payload,out/f'stage_{update:06d}.pt'); atomic_save(payload,out/'last.pt')
                atomic_json(metrics,out/f'prediction_{update:06d}.json')
                print(json.dumps(dict(update=update,validation=metrics)),flush=True)
    report=dict(completed=True,updates=args.updates,train_transitions=len(train.roots),
        train_episodes=len(train.rows),train_layouts=train.report['layouts'],validation_transitions=len(validation.roots),
        elapsed_seconds=time.time()-started,task_performance='not evaluated by prediction metrics')
    atomic_json(report,out/'summary.json'); atomic_json(report,out/'status.json')


if __name__=='__main__':
    main()
