"""Task optimization of fixed-reference candidate response representations.

Ordinary policy gradients select the current team configuration. There is no
critic, complete-task outcome predictor, or future sequence of fusion actions.
The replay objective remains short response geometry under common recourse.
"""
import argparse
import concurrent.futures
from dataclasses import asdict, replace
import datetime
import json
import math
import multiprocessing
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from citf.bridge import Source, World, inputs
from citf.contracts import Protocol
from citf.train import atomic_save
from experiments.candidate_response import (ResponseConfig, CandidateResponseModel,
    add_reference, load_response_data, response_objective)
from experiments.response_data import candidate_bank


worker = {}


def initialize(project, source, config):
    torch.set_num_threads(1)
    model = CandidateResponseModel(ResponseConfig(**config)).eval()
    worker.update(project=project, source=Source(project, source), model=model)


@torch.no_grad()
def rollout(job):
    model = worker['model']
    model.load_state_dict({k: torch.from_numpy(v) for k, v in job['state'].items()})
    torch.manual_seed(job['policy_seed'])
    task = job['task']
    world = World(worker['project'], Protocol(**job['protocol']), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    transitions, steps, changed, entropies = [], 0, 0, []
    while not world.done:
        if steps % 20 == 0 and (time.time() >= job['deadline'] or Path(job['stop_file']).exists()):
            return dict(completed=False, task=task, steps=steps, reason='deadline or cooperative stop')
        f, plans, detail = candidate_bank(world, worker['source'], model.cfg)
        if model.cfg.candidate_domain == 'original_physical_bank':
            plans = plans[:detail['original_count']]
        batch, gates, _ = inputs(world, f, plans); batch = add_reference(batch)
        output = model(batch)
        logits = output['logits'][0]
        distribution = torch.distributions.Categorical(logits=logits)
        action = int(distribution.sample()) if job['training'] else min(
            range(len(plans)), key=lambda k: (-float(logits[k]),
                int(k != 0) if model.cfg.policy_prior_mode == 'reference_saturated' else 0,
                tuple(plans[k])))
        physical = int(output['physical'][0].argmin())
        changed += int(action != physical); entropies.append(float(distribution.entropy()))
        if job['training']:
            transitions.append(dict(inputs={k: v[0].numpy().copy() for k, v in batch.items()},
                action=action, old_logp=float(distribution.log_prob(torch.tensor(action)))))
        gate = gates[action]
        force = (750 * (gate * f['learned'] + (1 - gate) * f['rules']) + 250).numpy()
        result = world.step(force); steps += 1
    reward = (1. if result['success'] else -1.) - .25 * result['collision']
    reward += .02 * result['capture'] - .0005 * result['duration']
    record = dict(completed=True, task=task, policy_seed=job['policy_seed'], steps=steps,
                  reward=reward, changed_fraction=changed / steps, entropy=float(np.mean(entropies)), **result)
    return dict(completed=True, record=record, transitions=transitions)


def input_batch(items, device):
    count = max(len(x['candidate_valid']) for x in items)
    candidate_names = {'nominal', 'candidate', 'unary', 'candidate_valid'}
    result = {}
    for name in items[0]:
        if name not in candidate_names:
            result[name] = torch.from_numpy(np.stack([x[name] for x in items])).to(device)
            continue
        values = []
        for item in items:
            value = item[name]; size = len(value)
            pad = np.repeat(value[:1], count - size, axis=0)
            if name == 'candidate_valid':
                pad[:] = False
            values.append(np.concatenate((value, pad), axis=0))
        result[name] = torch.from_numpy(np.stack(values)).to(device)
    return result


def main():
    parser = argparse.ArgumentParser()
    for key in ['project', 'source', 'pretrained', 'train-bank', 'validation-bank', 'out', 'deadline']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=160)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--groups', type=int, default=4)
    parser.add_argument('--group-size', type=int, default=4)
    parser.add_argument('--validation-every', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--encoder-lr', type=float, default=3e-5)
    parser.add_argument('--replay-weight', type=float, default=.1)
    parser.add_argument('--replay-batch', type=int, default=4)
    parser.add_argument('--entropy', type=float, default=.002)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--original-bank', action='store_true')
    parser.add_argument('--reference-prior', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(1); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    train_bank = json.loads(Path(args.train_bank).read_text())
    validation = json.loads(Path(args.validation_bank).read_text())
    if train_bank['role'] != 'train' or validation['role'] != 'validation':
        raise ValueError('Incorrect task split roles')
    if {x['geometry_family_id'] for x in train_bank['tasks']} & {x['geometry_family_id'] for x in validation['tasks']}:
        raise ValueError('Training/validation layout overlap')
    value = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    cfg = replace(ResponseConfig(**value['config']), deployment='policy',
                  candidate_domain='original_physical_bank' if args.original_bank else 'physical_bank_plus_joint_pairs')
    if args.reference_prior:
        cfg = replace(cfg, revision='CITF-reference-saturated-recourse-response-v2',
                      policy_prior_mode='reference_saturated')
    model = CandidateResponseModel(cfg).to(args.device); model.load_state_dict(value['state'])
    actor_ids = {id(p) for p in model.actor.parameters()}
    optimizer = torch.optim.AdamW([dict(params=model.actor.parameters(), lr=args.lr),
        dict(params=[p for p in model.parameters() if id(p) not in actor_ids], lr=args.encoder_lr)], weight_decay=1e-4)
    replay, replay_roots = load_response_data(value['protocol']['arguments']['train'], 'train')
    if {x['geometry_family_id'] for x in replay_roots} & {x['geometry_family_id'] for x in validation['tasks']}:
        raise ValueError('Replay/validation layout overlap')
    rng = np.random.default_rng(args.seed); replay_rng = np.random.default_rng(args.seed + 13000)
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    stop_file = str(out.parent / 'STOP_RESPONSE_POLICY')
    protocol = dict(value['protocol'], arguments=vars(args), response_pretrained=args.pretrained,
        response_replay=value['protocol']['arguments']['train'], source_unchanged=True,
        objective='ordinary clipped complete-task policy gradient plus matched short-response replay',
        no_critic_or_terminal_prediction=True, candidate_set_normalization=False,
        policy_prior_mode=cfg.policy_prior_mode,
        selection='complete fixed validation tasks: success then fewer collisions then capped time')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    update, episodes, total_steps, best = 0, 0, 0, None
    def state():
        return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}
    def job(task, parameters, training, seed):
        return dict(task=task, state=parameters, training=training, policy_seed=int(seed),
                    protocol=train_bank.get('protocol', {}), deadline=deadline, stop_file=stop_file)
    def save(path):
        atomic_save(dict(config=asdict(cfg), state={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), protocol=protocol, update=update, selected=best, seed=args.seed,
            training_episodes=episodes, training_steps=total_steps, sampler_rng=rng.bit_generator.state,
            replay_rng=replay_rng.bit_generator.state, torch_rng=torch.get_rng_state()), path)
    def status(completed=False):
        record = dict(completed=completed, update=update, total_updates=args.iterations,
            training_episodes=episodes, training_steps=total_steps, best=best,
            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (out / 'status.json').write_text(json.dumps(record, indent=2))
    status()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(args.project, args.source, asdict(cfg))) as pool:
        def validate():
            nonlocal best
            model.eval(); parameters = state()
            results = list(pool.map(rollout, [job(t, parameters, False, args.seed + i)
                                              for i, t in enumerate(validation['tasks'])]))
            if any(not r['completed'] for r in results):
                with (out / 'validation.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(update=update, completed=False, results=results)) + '\n')
                return False
            records = [r['record'] for r in results]
            success = sum(r['success'] for r in records); collision = sum(r['collision'] for r in records)
            capped = sum(r['duration'] if r['capture'] else 80 for r in records) / len(records)
            selection = [success, -collision, -capped]
            row = dict(update=update, completed=True, N=len(records), success=success, collision=collision,
                       capped_capture_time=capped, tasks=records)
            with (out / 'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            if best is None or tuple(selection) > tuple(best['selection_key']):
                best = dict(update=update, selection_key=selection, success=success, collision=collision, N=len(records))
                save(out / 'best.pt')
            save(out / f'stage_{update:06d}.pt'); save(out / 'last.pt'); status()
            print(json.dumps(dict(validation={k: v for k, v in row.items() if k != 'tasks'}, best=best)), flush=True)
            model.train(); return True
        proceed = validate()
        while proceed and update < args.iterations and time.time() < deadline and not Path(stop_file).exists():
            parameters = state(); jobs = []
            for _ in range(args.groups):
                task = dict(train_bank['tasks'][int(rng.integers(len(train_bank['tasks'])))])
                task['future_seed'] = int(rng.integers(200000000, 2000000000))
                for _ in range(args.group_size):
                    jobs.append(job(task, parameters, True, rng.integers(1, 2000000000)))
            results = list(pool.map(rollout, jobs))
            with (out / 'episodes.jsonl').open('a') as stream:
                for result in results:
                    stream.write(json.dumps(dict(update=update + 1, **result.get('record', {k:v for k,v in result.items() if k != 'transitions'}))) + '\n')
            if any(not r['completed'] for r in results):
                break
            rewards = np.asarray([r['record']['reward'] for r in results]).reshape(args.groups, args.group_size)
            advantages = ((rewards - rewards.mean(1, keepdims=True)) * args.group_size / (args.group_size - 1)).reshape(-1)
            transitions, values = [], []
            for result, advantage in zip(results, advantages):
                transitions.extend(result['transitions']); values.extend([advantage] * len(result['transitions']))
            count = len(transitions); episodes += len(results); total_steps += count
            action = torch.tensor([r['action'] for r in transitions], device=args.device)
            old_logp = torch.tensor([r['old_logp'] for r in transitions], device=args.device)
            advantage = torch.tensor(values, dtype=torch.float32, device=args.device)
            losses, entropies, kls, replay_losses = [], [], [], []
            for epoch in range(args.epochs):
                indices = rng.permutation(count)
                for start in range(0, count, args.batch_size):
                    ids = indices[start:start + args.batch_size]
                    batch = input_batch([transitions[i]['inputs'] for i in ids], args.device)
                    output = model(batch); distribution = torch.distributions.Categorical(logits=output['logits'])
                    logp = distribution.log_prob(action[ids]); logratio = logp - old_logp[ids]; ratio = logratio.exp()
                    loss = -torch.minimum(ratio * advantage[ids], ratio.clamp(.8, 1.2) * advantage[ids]).mean()
                    entropy = distribution.entropy().mean(); loss = loss - args.entropy * entropy
                    if args.replay_weight:
                        ri = replay_rng.integers(0, len(replay['history']), args.replay_batch)
                        rb = {k: v[ri].to(args.device) for k, v in replay.items()}
                        auxiliary = response_objective(model(rb), rb, cfg)['total']
                        loss = loss + args.replay_weight * auxiliary; replay_losses.append(float(auxiliary.detach()))
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                    losses.append(float(loss.detach())); entropies.append(float(entropy.detach()))
                    kls.append(float(((ratio - 1) - logratio).mean().detach()))
                if np.mean(kls[-max(1, math.ceil(count / args.batch_size)):]) > .08:
                    break
            update += 1
            row = dict(update=update, training_episodes=episodes, environment_steps=total_steps,
                success=sum(r['record']['success'] for r in results), collision=sum(r['record']['collision'] for r in results),
                N=len(results), reward_mean=float(rewards.mean()), advantage_abs_mean=float(np.abs(advantages).mean()),
                policy_loss=float(np.mean(losses)), entropy=float(np.mean(entropies)), kl=float(np.mean(kls)),
                replay_loss=float(np.mean(replay_losses)) if replay_losses else None)
            with (out / 'learning.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True); status()
            if update % args.validation_every == 0 or update == args.iterations:
                proceed = validate()
            else:
                save(out / 'last.pt')
        save(out / 'last.pt')
    status(update == args.iterations)


if __name__ == '__main__':
    main()
