"""Learn contact-event configuration corrections around a retained controller.

Short trajectories train candidate-response physics under the same retained
feedback. Ordinary task policy gradients train a current action at observed
contact/miss events. There is no critic, outcome predictor, or CV target.
"""
import argparse
import concurrent.futures
from dataclasses import asdict, dataclass, replace
import datetime
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from citf.bridge import Controller, Source, World, factorize, inputs
from citf.contracts import Protocol, canonical_hash, file_hash
from citf.model import Attention
from citf.train import atomic_save
from experiments import synchronized_configuration_policy as synchronized
from experiments.candidate_response import (CandidateResponseModel, ResponseConfig,
    add_reference, load_response_data, response_objective, response_validation, stopped_positions)
from experiments.capture_window_diagnostic import geometry


@dataclass(frozen=True)
class ContactEventConfig(ResponseConfig):
    revision: str = 'CITF-retained-contact-event-response-v1'
    teacher_checkpoint: str = ''
    teacher_sha256: str = ''
    event_strategy: str = 'learned'
    event_schedule: str = 'first8_first6_firstmiss_recovery2'
    response_reference: str = 'current candidate one tick then frozen teacher feedback'
    candidate_domain: str = 'original_physical_bank'
    deployment: str = 'policy'


class ContactEventModel(CandidateResponseModel):
    def __init__(self, cfg=ContactEventConfig()):
        super().__init__(cfg)
        self.capture_interaction = Attention(cfg.width, cfg.heads, True)
        self.capture_strength = nn.Parameter(torch.zeros(()))
        self.phase_embed = nn.Sequential(nn.Linear(10, 4 * cfg.width + 16), nn.Tanh())

    def encode_candidates(self, batch, nominal, candidates, history):
        z = super().encode_candidates(batch, nominal, candidates, history)
        if not self.cfg.candidate_context:
            return z
        b, c, e, d = z.shape
        pos = batch['positions'][:, None].expand(b, c, e, 2).reshape(b * c, e, 2)
        velocity = ((nominal[..., 0, :] - batch['positions'][:, None]) / self.cfg.dt).reshape(b * c, e, 2)
        delta = pos[:, None] - pos[:, :, None]
        relative_velocity = velocity[:, None] - velocity[:, :, None]
        distance = delta.norm(dim=-1).clamp_min(.1)
        product = (delta * relative_velocity).sum(-1)
        closing = -product / distance
        lateral = (delta[..., 0] * relative_velocity[..., 1] - delta[..., 1] * relative_velocity[..., 0]) / distance
        closest_time = (-product / relative_velocity.square().sum(-1).clamp_min(.01)).clamp(0, self.cfg.horizon * self.cfg.dt)
        edge = torch.cat((delta / 20, (closing / 5)[..., None], (lateral / 5)[..., None],
                          (closest_time / (self.cfg.horizon * self.cfg.dt))[..., None]), -1)
        valid = batch['entity_valid'][:, None].expand(b, c, e).reshape(b * c, e)
        flat = z.reshape(b * c, e, d)
        correction = self.capture_interaction(flat, valid, edge) - flat
        return (flat + self.capture_strength.tanh() * correction).reshape(b, c, e, d)

    def policy_logits(self, batch, features, baseline, encoded):
        if 'teacher_logits' in batch:
            features = features + self.phase_embed(batch['event_features'])[:, None]
            return batch['teacher_logits'] + self.cfg.policy_logit_scale * self.actor(features).squeeze(-1)
        return super().policy_logits(batch, features, baseline, encoded)


def contact_objective(output, batch, cfg):
    result = response_objective(output, batch, cfg)
    prediction = output['xy']
    truth = stopped_positions(batch)[:, :, :cfg.modes]
    relative_pred = (prediction[..., :-1, :, :] - prediction[..., -1:, :, :]).mean(2)
    relative_true = (truth[..., :-1, :, :] - truth[..., -1:, :, :]).mean(2)
    # Capture-zone errors matter most for the boats that can physically reach it.
    distance = (truth[..., :-1, :, :] - truth[..., -1:, :, :]).norm(dim=-1).mean(2)
    weight = 1 + 3 * torch.exp(-(distance - 5).clamp_min(0) / 3)
    valid = (batch['known_time_valid'][:, None, None]
             & batch['candidate_valid'][:, :, None, None]
             & batch['entity_valid'][:, None, :-1, None])
    weight = weight.masked_fill(~valid, 0)
    def weighted(error, mask):
        return (F.smooth_l1_loss(error, torch.zeros_like(error), reduction='none') * mask[..., None]).sum() / (2 * mask.sum()).clamp_min(1)
    position = weighted(relative_pred - relative_true, weight)
    response = weighted(((relative_pred - relative_pred[:, :1]) - (relative_true - relative_true[:, :1])) / .25, weight)
    velocity_error = (relative_pred[..., 1:, :] - relative_pred[..., :-1, :] - relative_true[..., 1:, :] + relative_true[..., :-1, :]) / cfg.dt
    velocity = weighted(velocity_error, weight[..., 1:] * batch['known_time_valid'][:, None, None, :-1])
    if cfg.supervision == 'response':
        result['total'] = result['total'] + .25 * (position + response + velocity)
    result.update(capture_relative_position=position, capture_relative_response=response, capture_relative_velocity=velocity)
    return result


def teacher_fingerprint(path, expected=''):
    fingerprint = file_hash(path)
    if expected and fingerprint != expected:
        raise ValueError('Frozen contact teacher checkpoint has changed')
    return fingerprint


def pin_teacher(cfg, protocol):
    """New artifacts pin teacher contents; legacy artifacts keep their history."""
    fingerprint = teacher_fingerprint(cfg.teacher_checkpoint, cfg.teacher_sha256)
    teacher = torch.load(cfg.teacher_checkpoint, map_location='cpu', weights_only=False)
    for key in ('source_sha256', 'source_implementation_sha256', 'environment_sha256'):
        if not protocol.get(key) or teacher['protocol'].get(key) != protocol[key]:
            raise ValueError(f'Contact teacher and student differ: {key}')
    return replace(cfg, teacher_sha256=fingerprint)


def retained_teacher(path, expected_sha256=''):
    synchronized.install_semantics()
    fingerprint = teacher_fingerprint(path, expected_sha256)
    value = torch.load(path, map_location='cpu', weights_only=False)
    model = synchronized.joint.JointConfigurationPolicy(synchronized.SynchronizedPolicyConfig(**value['config']))
    model.load_state_dict(value['state'])
    model.checkpoint_sha256 = fingerprint
    model.checkpoint_protocol = value['protocol']
    return model.eval().requires_grad_(False)


class ContactEventController(Controller):
    def __init__(self, *args, stochastic=False, teacher=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher if teacher is not None else retained_teacher(
            self.model.cfg.teacher_checkpoint, self.model.cfg.teacher_sha256)
        if self.model.cfg.teacher_sha256 and self.teacher.checkpoint_sha256 != self.model.cfg.teacher_sha256:
            raise ValueError('Supplied contact teacher differs from the checkpoint identity')
        for key, actual in (('source_sha256', self.source.sha256),
                            ('source_implementation_sha256', self.source.implementation_sha256)):
            if self.teacher.checkpoint_protocol.get(key) != actual:
                raise ValueError(f'Contact teacher differs from the runtime source: {key}')
        self.teacher_controller = self.child_controller(synchronized.joint.JointPolicyController, self.teacher)
        self.stochastic = stochastic
        self.seen = set()
        self.minimum = float('inf')
        self.previous = float('inf')
        self.receding_steps = 0
        self.missed_time = None
        self.transition = None

    @torch.no_grad()
    def act(self, world):
        if self.teacher.checkpoint_protocol.get('environment_sha256') != world.environment_sha256:
            raise ValueError('Contact teacher differs from the runtime environment')
        geo = geometry(world.state())
        distance = geo['distance']; elapsed = float(world.env.Current_T)
        self.minimum = min(self.minimum, distance)
        self.receding_steps = self.receding_steps + 1 if distance > self.previous + .01 else 0
        conditions = dict(contact8=distance <= 8, capture6=distance <= 6,
            missed_window=self.minimum <= 8 and distance >= self.minimum + .15 and self.receding_steps >= 2,
            recovery2=self.missed_time is not None and elapsed >= self.missed_time + 2)
        phases = [k for k, value in conditions.items() if value and k not in self.seen]
        self.previous = distance
        self.transition = None
        force, trace = self.teacher_controller.act(world)
        trace.update(contact_event=phases, retained_teacher_indices=trace['selected_indices'])
        if not phases:
            return force, trace
        self.seen.update(phases)
        if 'missed_window' in phases:
            self.missed_time = elapsed
        f = factorize(world, self.source)
        plans = np.asarray(trace['configurations'], dtype=np.int64)
        physical = next(i for i, p in enumerate(plans) if p.tolist() == trace['physical_selected_indices'])
        order = [physical] + [i for i in range(len(plans)) if i != physical]
        plans = plans[order]
        teacher_logits = -np.asarray(trace['scores'], dtype=np.float32)[order]
        batch, gates, _ = inputs(world, f, plans)
        batch = add_reference(batch)
        event_features = [float(k in phases) for k in conditions] + [distance / 10,
            geo['closing_speed'] / 5, geo['radial_gap'] / 20, geo['separation'] / 10,
            self.minimum / 10, elapsed / 80]
        batch.update(teacher_logits=torch.from_numpy(teacher_logits[None]),
                     event_features=torch.tensor([event_features], dtype=torch.float32))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        logits = self.model(batch)['logits'][0]
        distribution = torch.distributions.Categorical(logits=logits)
        if self.model.cfg.event_strategy == 'physical':
            choice = 0
        elif self.model.cfg.event_strategy == 'teacher':
            choice = min(range(len(plans)), key=lambda i: (-float(teacher_logits[i]), tuple(plans[i])))
        elif self.model.cfg.event_strategy == 'learned':
            choice = int(distribution.sample()) if self.stochastic else min(
                range(len(plans)), key=lambda i: (-float(logits[i]), tuple(plans[i])))
        else:
            raise ValueError('Unknown declared event strategy')
        teacher_choice = min(range(len(plans)), key=lambda i: (-float(teacher_logits[i]), tuple(plans[i])))
        self.transition = dict(inputs={k: v[0].cpu().numpy().copy() for k, v in batch.items()},
            action=choice, old_logp=float(distribution.log_prob(torch.tensor(choice, device=logits.device))),
            changed=int(choice != teacher_choice), entropy=float(distribution.entropy()), phases=phases)
        gate = gates[choice]
        force = (750 * (gate * f['learned'] + (1 - gate) * f['rules']) + 250).numpy()
        trace.update(configurations=plans.tolist(), scores=(-logits).cpu().tolist(),
            selected_indices=plans[choice].tolist(), gates=gate.tolist(), physical_thrust=force.tolist(),
            teacher_logits=teacher_logits.tolist(), event_features=event_features,
            selection='one current correction at observed contact event; retained teacher between events')
        return force, trace


worker = {}


def initialize(project, source, config):
    torch.set_num_threads(1)
    cfg = ContactEventConfig(**config)
    worker.update(project=project, source=Source(project, source), model=ContactEventModel(cfg).eval(),
                  teacher=retained_teacher(cfg.teacher_checkpoint, cfg.teacher_sha256))


@torch.no_grad()
def rollout(job):
    model = worker['model']
    model.load_state_dict({k: torch.from_numpy(v) for k, v in job['state'].items()})
    torch.manual_seed(job['policy_seed'])
    task = job['task']
    world = World(worker['project'], Protocol(**job['protocol']), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    controller = ContactEventController(worker['source'], model, teacher=worker['teacher'], stochastic=job['training'])
    transitions, steps, changed, entropies = [], 0, 0, []
    while not world.done:
        if steps % 20 == 0 and (time.time() >= job['deadline'] or Path(job['stop_file']).exists()):
            return dict(completed=False, task=task, steps=steps, reason='deadline or cooperative stop')
        force, trace = controller.act(world)
        transition = controller.transition
        if transition is not None:
            changed += transition['changed']; entropies.append(transition['entropy'])
            if job['training']:
                transitions.append(transition)
        result = world.step(force); steps += 1
    duration = result['duration'] if result['capture'] else 80.
    reward = (1. if result['success'] else -1.) - .25 * result['collision'] + .02 * result['capture'] - .0005 * duration
    record = dict(completed=True, task=task, policy_seed=job['policy_seed'], steps=steps,
        decision_events=len(entropies), changed_events=changed, execution='sample' if job['training'] else 'greedy',
        reward=reward, reward_duration='capped_capture', entropy=float(np.mean(entropies)) if entropies else 0., **result)
    return dict(completed=True, record=record, transitions=transitions)


def input_batch(items, device):
    count = max(len(x['candidate_valid']) for x in items)
    candidate_names = {'nominal', 'candidate', 'unary', 'candidate_valid', 'teacher_logits'}
    output = {}
    for name in items[0]:
        values = []
        for item in items:
            value = item[name]
            if name in candidate_names:
                pad = np.repeat(value[:1], count - len(value), axis=0)
                if name == 'candidate_valid':
                    pad[:] = False
                value = np.concatenate((value, pad), axis=0)
            values.append(value)
        output[name] = torch.from_numpy(np.stack(values)).to(device)
    return output


def contact_data_files(path, role):
    """All files actually consumed by either supported replay representation."""
    path = Path(path)
    if path.is_dir():
        return [path / 'status.json', *sorted(path.glob('*.pt'))]
    index = json.loads(path.read_text(encoding='utf-8'))
    return [path, *(Path(name) for name in index['roots'][role])]


def load_contact_data(path, role):
    path = Path(path)
    if path.is_dir():
        return load_response_data(path, role)
    index = json.loads(path.read_text())
    values = [torch.load(file, map_location='cpu', weights_only=False) for file in index['roots'][role]]
    if any(value['metadata']['split'] != role for value in values):
        raise ValueError('Contact response index split mismatch')
    names = ['history', 'positions', 'history_valid', 'entity_valid', 'nominal', 'candidate',
             'unary', 'candidate_valid', 'time_context', 'known_time_valid', 'real_positions', 'time_valid']
    data = {name: torch.stack([value['tensors'][name] for value in values]) for name in names}
    return add_reference(data), [value['metadata'] for value in values]


def prepare_main():
    parser = argparse.ArgumentParser()
    for name in ['pretrained', 'teacher', 'out']:
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--train')
    parser.add_argument('--validation')
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--seed', type=int, default=9101)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-context', action='store_true')
    parser.add_argument('--ordinary', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(1); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    value = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    cfg = ContactEventConfig(**dict(value['config'], revision=ContactEventConfig().revision,
        teacher_checkpoint=args.teacher, teacher_sha256='', response_reference=ContactEventConfig().response_reference,
        candidate_domain='original_physical_bank', deployment='policy',
        candidate_context=not args.no_context, supervision='ordinary' if args.ordinary else 'response'))
    cfg = pin_teacher(cfg, value['protocol'])
    model = ContactEventModel(cfg).to(args.device)
    model.load_state_dict(value['state'], strict=False)
    nn.init.zeros_(model.actor[-1].weight); nn.init.zeros_(model.actor[-1].bias)
    actor_ids = {id(p) for p in model.actor.parameters()}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if id(p) not in actor_ids], lr=3e-4, weight_decay=1e-4)
    protocol = dict(value['protocol'], arguments=vars(args), source_pretrained=args.pretrained,
        response_replay=args.train, teacher=args.teacher, event_schedule=cfg.event_schedule,
        physical_pretraining='short candidate response under frozen teacher feedback; no complete outcome targets',
        actor_initialization='zero additive correction on frozen teacher logits; exact greedy teacher start',
        response_representation='relative closing/lateral velocity and time of closest approach in candidate attention',
        task_policy='ordinary complete task return at observed events; no critic or terminal predictor',
        training_step_semantics='policy updates count event decisions, environment steps separately in episodes')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    train, validation, roots, validation_roots = None, None, [], []
    if args.steps:
        train, roots = load_contact_data(args.train, 'train')
        validation, validation_roots = load_contact_data(args.validation, 'validation')
        if {x['geometry_family_id'] for x in roots} & {x['geometry_family_id'] for x in validation_roots}:
            raise ValueError('Physical response train/validation layout overlap')
    rng = np.random.default_rng(args.seed)
    loss_value = None
    for step in range(1, args.steps + 1):
        indices = rng.integers(0, len(roots), 8)
        batch = {k: v[indices].to(args.device) for k, v in train.items()}
        loss = contact_objective(model(batch), batch, cfg)['total']
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
        loss_value = float(loss.detach())
        if step % 100 == 0:
            status = dict(completed=False, step=step, total_steps=args.steps, loss=loss_value,
                updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            (out / 'status.json').write_text(json.dumps(status, indent=2)); print(json.dumps(status), flush=True)
    metrics = response_validation(model.eval(), validation, args.device) if validation is not None else None
    protocol.update(train_roots=len(roots), validation_roots=len(validation_roots), physical_validation=metrics)
    record = dict(config=asdict(cfg), state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                  protocol=protocol, seed=args.seed, step=args.steps)
    atomic_save(record, out / 'prepared.pt')
    (out / 'status.json').write_text(json.dumps(dict(completed=True, step=args.steps, loss=loss_value,
        validation=metrics, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2))


def train_main():
    # Reuse the existing ordinary PPO trainer and actual full-episode validation.
    # Only event transitions enter its actor update; physical replay is unchanged.
    from experiments import response_policy as training
    training.ResponseConfig = ContactEventConfig
    training.CandidateResponseModel = ContactEventModel
    training.initialize = initialize
    training.rollout = rollout
    training.input_batch = input_batch
    training.response_objective = contact_objective
    training.load_response_data = load_contact_data
    training.main()


def load_model(path, source, world, device):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value['protocol']['source_sha256'] != source.sha256 or value['protocol']['environment_sha256'] != world.environment_sha256:
        raise ValueError('Contact-event checkpoint source/environment mismatch')
    model = ContactEventModel(ContactEventConfig(**value['config']))
    model.load_state_dict(value['state'])
    return model.to(device).eval()


def contact_runtime_identity(value, stored):
    """Include external teacher contents and the shared inference dependency chain."""
    config = stored['config']
    value['controller'].update(config)
    value['controller']['teacher_sha256'] = teacher_fingerprint(
        config['teacher_checkpoint'], config.get('teacher_sha256', ''))
    dependencies = dict(value['runtime_dependencies'])
    value['contact_runtime_dependencies'] = dependencies
    value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'], contact=dependencies))
    return value


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ('prepare', 'train'):
        mode = sys.argv.pop(1)
        return prepare_main() if mode == 'prepare' else train_main()
    import citf.evaluate as evaluation
    evaluation.load_behavior = load_model
    evaluation.Controller = ContactEventController
    original = evaluation.runtime_identity
    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        checkpoint = args[2] if len(args) > 2 else kwargs['checkpoint']
        stored = torch.load(checkpoint, map_location='cpu', weights_only=False)
        value['method'] = 'CITF-retained-contact-event-response-development'
        return contact_runtime_identity(value, stored)
    evaluation.runtime_identity = identity
    evaluation.main()


if __name__ == '__main__':
    main()
