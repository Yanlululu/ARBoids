"""Transfer a retained controller into bank-independent response features.

The frozen teacher supplies current-action labels on teacher or student trajectories.
Task outcomes are recorded for provenance, never used as prediction targets.
Every state retains the fixed physical reference before candidate subsampling.
"""
import argparse
import concurrent.futures
from dataclasses import asdict, replace
import datetime
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from citf.bridge import World, inputs
from citf.contracts import Protocol
from citf.train import atomic_save
from experiments import root_response_diagnostic as diagnostic
from experiments.candidate_response import (CandidateResponseModel, ResponseConfig,
    add_reference, load_response_data, response_objective)
from experiments.response_policy import input_batch


def initialize(project, source, teacher, out, deadline, role, stride, candidates,
               behavior=None):
    diagnostic.initialize(project, source, teacher, out, deadline)
    diagnostic.worker.update(role=role, stride=stride, candidates=candidates,
        teacher_checkpoint=teacher, behavior_checkpoint=behavior or teacher,
        behavior_model=None)
    if behavior is not None:
        value = torch.load(behavior, map_location='cpu', weights_only=False)
        cfg = ResponseConfig(**value['config'])
        if cfg.deployment != 'policy' or cfg.candidate_domain != 'original_physical_bank':
            raise ValueError('Student aggregation requires the declared original-bank policy')
        model = CandidateResponseModel(cfg).eval()
        model.load_state_dict(value['state'])
        diagnostic.worker['behavior_model'] = model


@torch.no_grad()
def collect_task(job):
    task, index = job['task'], job['index']
    w = diagnostic.worker
    world = World(w['project'], Protocol(), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    rng = np.random.default_rng(task['scene_seed'] + 71719)
    samples, step, events, disagreements = [], 0, set(), 0
    while not world.done:
        if time.time() >= w['deadline']:
            return dict(completed=False, task=task, steps=step, states=len(samples), reason='deadline')
        f, plans, physical_scores, reference, _ = diagnostic.physical_bank(world, w['source'], w['cfg'])
        b, _, _ = inputs(world, f, plans)
        logits = w['model'](b)['logits'][0]
        action = min(range(len(plans)), key=lambda k: (-float(logits[k]), tuple(plans[k])))
        reference_index = next(k for k, row in enumerate(plans) if np.array_equal(row, reference))
        executed = action
        if w['behavior_model'] is not None:
            order = [reference_index] + [k for k in range(len(plans)) if k != reference_index]
            candidate_keys = {'nominal', 'candidate', 'unary', 'candidate_valid'}
            student_batch = {k: v[:, order] if k in candidate_keys else v for k, v in b.items()}
            student_logits = w['behavior_model'](add_reference(student_batch))['logits'][0]
            chosen = min(range(len(order)), key=lambda k: (-float(student_logits[k]),
                int(k != 0) if w['behavior_model'].cfg.policy_prior_mode == 'reference_saturated' else 0,
                tuple(plans[order[k]])))
            executed = order[chosen]
        disagreements += int(executed != action)
        state = world.state()
        distance = float(np.linalg.norm(state[:-1, :2] - state[-1, :2], axis=-1).min())
        gap = float(np.linalg.norm(state[:-1, :2], axis=-1).min() - np.linalg.norm(state[-1, :2]))
        triggers = {name for name, yes in [('contact12', distance <= 12),
            ('contact8', distance <= 8), ('radial6', gap >= -6)] if yes and name not in events}
        if step % w['stride'] == 0 or triggers:
            selected = []
            ranking = sorted(range(len(plans)), key=lambda k: (-float(logits[k]), tuple(plans[k])))
            for k in [reference_index, action, executed, *ranking[:4], *np.argsort(physical_scores)[:3], *rng.permutation(len(plans))]:
                if int(k) not in selected:
                    selected.append(int(k))
                if len(selected) == w['candidates']:
                    break
            candidate_keys = {'nominal', 'candidate', 'unary', 'candidate_valid'}
            item = {k: (v[0, selected] if k in candidate_keys else v[0]).numpy().copy()
                    for k, v in b.items()}
            item.update(reference_candidate=item['candidate'][0].copy(),
                        reference_nominal=item['nominal'][0].copy(), reference_unary=item['unary'][0].copy())
            samples.append(dict(inputs=item, teacher_logits=logits[selected].numpy().copy(),
                teacher_action=selected.index(action), behavior_action=selected.index(executed), root_step=step,
                observed_distance=distance, observed_radial_gap=gap,
                configurations=plans[selected].tolist(), original_candidate_count=len(plans)))
            events.update(triggers)
        result = world.step(diagnostic.forces(f, plans[executed])); step += 1
    record = dict(completed=True, task=task, steps=step, states=len(samples),
        teacher_disagreements=disagreements, behavior_checkpoint=w['behavior_checkpoint'], **result)
    atomic_save(dict(samples=samples, metadata=dict(split=w['role'], task=task,
        geometry_family_id=task['geometry_family_id'], source_sha256=w['source'].sha256,
        environment_sha256=world.environment_sha256, teacher_checkpoint=w['teacher_checkpoint'],
        behavior_checkpoint=w['behavior_checkpoint']), outcome_audit_only=record),
        w['out'] / f'task_{index:04d}.pt')
    return record


def collect_main(args):
    torch.set_num_threads(1)
    bank = json.loads(Path(args.bank).read_text())
    if bank['role'] not in ('train', 'validation'):
        raise ValueError('Teacher transfer requires train or validation layouts')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    protocol = dict(arguments=vars(args), role=bank['role'],
        supervision='frozen teacher current-action logits and choices; no outcome labels',
        candidates='reference, teacher and behavior actions, leading teacher and physical choices, then seeded random negatives',
        state_sampling='every declared stride plus first12m,8m and radial-minus6m states',
        complete_teacher_trajectories=args.behavior is None,
        complete_student_trajectories=args.behavior is not None, retain_successes_and_failures=True)
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    status = dict(completed=False, finished_jobs=0, expected_jobs=len(bank['tasks']), states=0, jobs=[])
    (out / 'status.json').write_text(json.dumps(status, indent=2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(args.project, args.source, args.teacher, args.out, deadline,
                      bank['role'], args.stride, args.candidates, args.behavior)) as pool:
        pending = {pool.submit(collect_task, dict(index=i, task=t)): i for i, t in enumerate(bank['tasks'])}
        for future in concurrent.futures.as_completed(pending):
            index = pending[future]
            try:
                result = future.result()
            except Exception as error:
                result = dict(completed=False, task=bank['tasks'][index], error=repr(error))
            with (out / 'records.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(index=index, **result)) + '\n')
            status['jobs'].append(dict(index=index, status='completed' if result['completed'] else 'incomplete'))
            status['finished_jobs'] += int(result['completed']); status['states'] += result.get('states', 0)
            (out / 'status.json').write_text(json.dumps(status, indent=2))
            print(json.dumps(dict(index=index, **result)), flush=True)
    status['completed'] = status['finished_jobs'] == status['expected_jobs']
    status['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (out / 'status.json').write_text(json.dumps(status, indent=2))


def load_samples(path, role):
    path = Path(path)
    if not json.loads((path / 'status.json').read_text())['completed']:
        raise ValueError('Incomplete teacher trajectory collection')
    samples, layouts = [], set()
    for p in sorted(path.glob('task_*.pt')):
        value = torch.load(p, map_location='cpu', weights_only=False)
        if value['metadata']['split'] != role:
            raise ValueError('Incorrect teacher transfer split')
        samples.extend(value['samples']); layouts.add(value['metadata']['geometry_family_id'])
    return samples, layouts


def teacher_batch(samples, device):
    batch = input_batch([x['inputs'] for x in samples], device)
    teacher = torch.full(batch['candidate_valid'].shape, -torch.inf, device=device)
    for i, sample in enumerate(samples):
        teacher[i, :len(sample['teacher_logits'])] = torch.as_tensor(sample['teacher_logits'], device=device)
    action = torch.tensor([x['teacher_action'] for x in samples], device=device)
    return batch, teacher, action


def distill_loss(logits, teacher, action, valid):
    hard = F.cross_entropy(logits, action)
    probability = teacher.softmax(-1)
    log_prediction = logits.log_softmax(-1).masked_fill(~valid, 0)
    soft = -(probability * log_prediction).sum(-1).mean()
    # Center on the declared teacher action, never on the candidate pool.
    prediction_difference = logits - logits.gather(1, action[:, None])
    teacher_difference = teacher - teacher.gather(1, action[:, None])
    relative = F.smooth_l1_loss(prediction_difference[valid], teacher_difference[valid])
    return .5 * hard + .5 * soft + .2 * relative


def train_main(args):
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    train, train_layouts = load_samples(args.train, 'train')
    original_train_states = len(train)
    if args.augment_train:
        augmented, augmented_layouts = load_samples(args.augment_train, 'train')
        train.extend(augmented); train_layouts |= augmented_layouts
    validation, validation_layouts = load_samples(args.validation, 'validation')
    if train_layouts & validation_layouts:
        raise ValueError('Teacher train/validation layouts overlap')
    value = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    cfg = replace(ResponseConfig(**value['config']), deployment='policy',
        policy_prior_mode='reference_saturated', candidate_domain='original_physical_bank',
        revision=('CITF-student-state-response-policy-transfer-v2' if args.augment_train
                  else 'CITF-bank-independent-response-policy-transfer-v1'))
    model = CandidateResponseModel(cfg).to(args.device); model.load_state_dict(value['state'])
    actor_ids = {id(p) for p in model.actor.parameters()}
    optimizer = torch.optim.AdamW([dict(params=model.actor.parameters(), lr=args.lr),
        dict(params=[p for p in model.parameters() if id(p) not in actor_ids], lr=args.encoder_lr)], weight_decay=1e-4)
    replay_path = (value['protocol'].get('response_replay')
                   or value['protocol']['arguments']['train'])
    replay, replay_roots = load_response_data(replay_path, 'train')
    if {x['geometry_family_id'] for x in replay_roots} & validation_layouts:
        raise ValueError('Response replay and teacher validation overlap')
    protocol = dict(value['protocol'], arguments=vars(args), response_pretrained=args.pretrained,
        response_replay=replay_path, objective='current-action policy transfer plus observed short-response replay',
        complete_task_outcomes_used_for_training=False, train_states=len(train), validation_states=len(validation),
        original_train_states=original_train_states, augmented_train_states=len(train)-original_train_states,
        selection='held-out action agreement then distillation loss; full closed-loop selection remains separate')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    rng = np.random.default_rng(args.seed); deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    step, best = 0, None
    def save(path):
        atomic_save(dict(config=asdict(cfg), state={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), protocol=protocol, seed=args.seed, step=step, best=best,
            sampler_rng=rng.bit_generator.state), path)
    def status(completed=False):
        (out / 'status.json').write_text(json.dumps(dict(completed=completed, step=step,
            total_steps=args.steps, best=best, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2))
    @torch.no_grad()
    def validate():
        nonlocal best
        model.eval(); agreement, loss = 0, 0.
        for start in range(0, len(validation), args.batch_size):
            items = validation[start:start + args.batch_size]
            b, teacher, action = teacher_batch(items, args.device); prediction = model(b)['logits']
            agreement += int((prediction.argmax(-1) == action).sum())
            loss += float(distill_loss(prediction, teacher, action, b['candidate_valid'])) * len(items)
        key = [agreement, -loss / len(validation)]
        row = dict(step=step, N=len(validation), action_agreement=agreement,
            agreement_rate=agreement / len(validation), loss=loss / len(validation))
        if best is None or tuple(key) > tuple(best['selection_key']):
            best = dict(step=step, selection_key=key, **{k:v for k,v in row.items() if k != 'step'})
            save(out / 'best.pt')
        save(out / f'stage_{step:06d}.pt'); save(out / 'last.pt'); status()
        with (out / 'learning.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True); model.train()
    validate()
    while step < args.steps and time.time() < deadline:
        selected = rng.integers(0, len(train), args.batch_size)
        b, teacher, action = teacher_batch([train[i] for i in selected], args.device)
        loss = distill_loss(model(b)['logits'], teacher, action, b['candidate_valid'])
        if args.replay_weight:
            ri = rng.integers(0, len(replay['history']), args.replay_batch)
            rb = {k:v[ri].to(args.device) for k,v in replay.items()}
            loss = loss + args.replay_weight * response_objective(model(rb), rb, cfg)['total']
        optimizer.zero_grad(set_to_none=True); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); step += 1
        if step % 100 == 0:
            status(); print(json.dumps(dict(step=step, training_loss=float(loss.detach()))), flush=True)
        if step % args.validation_every == 0 or step == args.steps:
            validate()
    save(out / 'last.pt'); status(step == args.steps)


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_subparsers(dest='mode', required=True)
    collection = modes.add_parser('collect')
    for name in ('project', 'source', 'teacher', 'bank', 'out', 'deadline'):
        collection.add_argument('--' + name, required=True)
    collection.add_argument('--workers', type=int, default=2)
    collection.add_argument('--stride', type=int, default=8)
    collection.add_argument('--candidates', type=int, default=12)
    collection.add_argument('--behavior', help='Frozen response policy visiting states for teacher relabeling')
    training = modes.add_parser('train')
    for name in ('pretrained', 'train', 'validation', 'out', 'deadline'):
        training.add_argument('--' + name, required=True)
    training.add_argument('--seed', type=int, required=True)
    training.add_argument('--augment-train', help='Additional teacher-labeled student states from train layouts')
    training.add_argument('--steps', type=int, default=1200)
    training.add_argument('--validation-every', type=int, default=300)
    training.add_argument('--batch-size', type=int, default=24)
    training.add_argument('--lr', type=float, default=3e-4)
    training.add_argument('--encoder-lr', type=float, default=3e-5)
    training.add_argument('--replay-weight', type=float, default=.1)
    training.add_argument('--replay-batch', type=int, default=4)
    training.add_argument('--device', default='cuda')
    args = parser.parse_args()
    (collect_main if args.mode == 'collect' else train_main)(args)


if __name__ == '__main__':
    main()
