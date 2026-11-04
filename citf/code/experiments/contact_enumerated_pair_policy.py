"""Enumerate the two current proposals under common future continuations.

Only the current observed event is varied. Complete returns are averaged over
the same two futures before forming the actor objective. The continuation is
the current greedy controller; there is no critic or return-prediction head.
"""
import argparse
import concurrent.futures
from dataclasses import asdict, dataclass
import datetime
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

from citf.bridge import Source, World
from citf.contracts import Protocol
from citf.search import candidate_grid
from citf.train import atomic_save
from experiments import contact_event_policy as contact
from experiments import contact_pair_policy as pair
from experiments.response_policy import validated_training_config


@dataclass(frozen=True)
class EnumeratedPairConfig(pair.ContactPairConfig):
    revision: str = 'CITF-enumerated-current-contact-proposal-policy-v1'
    training_action_sampling: str = 'both current proposals; two common futures; mean before clipping'


ModelConfig = EnumeratedPairConfig
COMPARISON_PROPOSALS = ('teacher', 'physical')
PROTOCOL_OVERRIDES = {}
ACTOR_OBJECTIVE = None
worker = {}


def initialize(project, source, config):
    torch.set_num_threads(1)
    cfg = ModelConfig(**config)
    worker.update(project=project, source=Source(project, source),
        model=contact.ContactEventModel(cfg).eval(),
        teacher=contact.retained_teacher(cfg.teacher_checkpoint, cfg.teacher_sha256))


@torch.no_grad()
def rollout(job):
    model = worker['model']
    model.load_state_dict({k: torch.from_numpy(v) for k, v in job['state'].items()})
    torch.manual_seed(job['policy_seed'])
    task = job['task']
    world = World(worker['project'], Protocol(**job['protocol']), task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    controller = pair.ContactPairController(worker['source'], model, teacher=worker['teacher'])
    probe = int(np.random.default_rng(int(task['future_seed']) ^ 0x5EED9101).integers(4))
    transitions, steps, events, changed = [], 0, 0, 0
    intervention_step = None
    while not world.done:
        if steps % 20 == 0 and (time.time() >= job['deadline'] or Path(job['stop_file']).exists()):
            return dict(completed=False, task=task, steps=steps, reason='deadline or cooperative stop')
        force, trace = controller.act(world)
        transition = controller.transition
        if transition is not None:
            if job['training'] and events == probe:
                action = trace['retained_proposal_index'] if job['proposal'] == 'teacher' else 0
                batch = {k: torch.from_numpy(v[None]) for k, v in transition['inputs'].items()}
                distribution = torch.distributions.Categorical(logits=model(batch)['logits'][0])
                transition = dict(transition, action=action,
                    old_logp=float(distribution.log_prob(torch.tensor(action))),
                    old_probabilities=distribution.probs.numpy().copy(),
                    changed=int(action != trace['retained_proposal_index']))
                plans = np.asarray(trace['configurations'], dtype=np.int64)
                learned, rules, prior = worker['source'].commands(world.env)
                grid, _ = candidate_grid(prior)
                gate = grid[torch.arange(len(plans[action])), torch.as_tensor(plans[action])]
                force = (750 * (gate * learned + (1 - gate) * rules) + 250).numpy()
                transitions.append(transition)
                intervention_step = steps
                # Chosen from current observations before exposing a future draw.
                world.rng = np.random.RandomState(job['branch_future_seed'])
            changed += transition['changed']
            events += 1
        result = world.step(force)
        steps += 1
    duration = result['duration'] if result['capture'] else 80.
    reward = (1. if result['success'] else -1.) - .25 * result['collision'] + .02 * result['capture'] - .0005 * duration
    record = dict(completed=True, task=task, policy_seed=job['policy_seed'], steps=steps,
        decision_events=events, changed_events=changed, sampled_events=len(transitions),
        proposal=job.get('proposal'), future_replica=job.get('future_replica'),
        root_group=job.get('root_group'), probe_event_ordinal=probe if job['training'] else None,
        intervention_step=intervention_step,
        branch_future_seed=job.get('branch_future_seed') if job['training'] else None,
        sampled_action=transitions[0]['action'] if transitions else None,
        execution='enumerated_current_event' if job['training'] else 'greedy',
        reward=reward, reward_duration='capped_capture', **result)
    return dict(completed=True, record=record, transitions=transitions)


def event_examples(results):
    """Each group is teacher/physical crossed with two identical future draws."""
    examples, missing = [], 0
    for start in range(0, len(results), 4):
        group = results[start:start + 4]
        present = [bool(r['transitions']) for r in group]
        if not any(present):
            missing += 1
            continue
        if not all(present):
            raise ValueError('The common greedy prefix reached inconsistent intervention events')
        inputs = group[0]['transitions'][0]['inputs']
        probabilities = group[0]['transitions'][0]['old_probabilities']
        returns = np.zeros(len(probabilities), dtype=np.float32)
        counts = np.zeros(len(probabilities), dtype=np.int64)
        for result in group:
            transition = result['transitions'][0]
            action = transition['action']
            returns[action] += result['record']['reward']
            counts[action] += 1
        if np.any(counts == 0):
            raise ValueError('Every available current proposal needs the declared common futures')
        returns /= counts
        examples.append(dict(inputs=inputs, probabilities=probabilities,
            returns=returns, advantage=returns - np.dot(probabilities, returns),
            teacher_return=float(np.mean([x['record']['reward'] for x in group[:2]])),
            comparison_return=float(np.mean([x['record']['reward'] for x in group[2:]]))))
    return examples, missing


def train_main():
    parser = argparse.ArgumentParser()
    for key in ['project', 'source', 'pretrained', 'train-bank', 'validation-bank', 'training-schedule', 'out', 'deadline']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--seed', type=int, default=9101)
    parser.add_argument('--iterations', type=int, default=80)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--validation-every', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--replay-weight', type=float, default=.2)
    parser.add_argument('--replay-batch', type=int, default=8)
    parser.add_argument('--entropy', type=float, default=.002)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--optimizer-schedule')
    parser.add_argument('--task-replay-batch', type=int, default=0)
    parser.add_argument('--task-replay-weight', type=float, default=.5)
    args = parser.parse_args()
    if args.task_replay_batch < 0 or not 0 <= args.task_replay_weight <= 1:
        raise ValueError('Invalid task-comparison replay configuration')
    if args.task_replay_batch and ACTOR_OBJECTIVE is None:
        raise ValueError('Task-comparison replay requires a declared ranking objective')
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    train_bank = json.loads(Path(args.train_bank).read_text())
    validation = json.loads(Path(args.validation_bank).read_text())
    if train_bank['role'] != 'train' or validation['role'] != 'validation':
        raise ValueError('Incorrect task split roles')
    train_families = {x['geometry_family_id'] for x in train_bank['tasks']}
    validation_families = {x['geometry_family_id'] for x in validation['tasks']}
    if train_families & validation_families:
        raise ValueError('Training and validation geometry overlap')
    schedule = {}
    for line in Path(args.training_schedule).read_text().splitlines():
        row = json.loads(line)
        if not row.get('completed') or row['task']['geometry_family_id'] not in train_families:
            raise ValueError('The common training schedule must contain completed training tasks')
        schedule.setdefault(row['update'], []).append(row)
    if any(len(schedule.get(i, [])) != 16 for i in range(1, args.iterations + 1)):
        raise ValueError('Require four roots with four original draws at every update')
    optimization_schedule = None
    if args.optimizer_schedule:
        optimization_schedule = {r['update']: r['optimization_batches'] for r in
            (json.loads(line) for line in Path(args.optimizer_schedule).read_text().splitlines())}
        if any(i not in optimization_schedule for i in range(1, args.iterations + 1)):
            raise ValueError('The optimization schedule must cover every declared update')
    value = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    cfg = ModelConfig(**dict(value['config'], revision=ModelConfig().revision,
        decision_domain=ModelConfig().decision_domain,
        training_action_sampling=ModelConfig().training_action_sampling))
    cfg = validated_training_config(args.project, args.source, value, cfg)
    model = contact.ContactEventModel(cfg).to(args.device)
    model.load_state_dict(value['state'])
    actor_ids = {id(p) for p in model.actor.parameters()}
    optimizer = torch.optim.AdamW([dict(params=model.actor.parameters(), lr=args.lr),
        dict(params=[p for p in model.parameters() if id(p) not in actor_ids], lr=args.encoder_lr)], weight_decay=1e-4)
    replay_path = value['protocol'].get('response_replay') or value['protocol']['arguments']['train']
    replay, replay_roots = contact.load_contact_data(replay_path, 'train')
    if {x['geometry_family_id'] for x in replay_roots} & validation_families:
        raise ValueError('Response replay and task validation geometry overlap')
    replay_rng = np.random.default_rng(args.seed + 13000)
    task_replay_rng = np.random.default_rng(args.seed + 27000)
    task_comparisons = []
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    stop_file = str(out.parent / 'STOP_RESPONSE_POLICY')
    protocol = dict(value['protocol'], arguments=vars(args), response_pretrained=args.pretrained,
        response_replay=replay_path, source_unchanged=True, no_critic_or_terminal_prediction=True,
        objective='enumerated current-event expected complete return; old-policy weighted clipped actor surrogate; physical response replay',
        state_distribution='current greedy prefix and continuation; only the assigned current event is varied',
        future_averaging='average two actual common future returns per proposal before clipping',
        budget='same 320 root/prefix groups and 1280 real episodes as original80; two proposals by two futures; declared optimizer schedule preserves update and replay counts, while batch composition differs',
        training_action_sampling=cfg.training_action_sampling,
        selection='complete fixed64 validation: success then fewer collisions then capped capture time')
    protocol.update(PROTOCOL_OVERRIDES)
    protocol['task_comparison_replay'] = dict(
        enabled=bool(args.task_replay_batch), batch=args.task_replay_batch,
        weight=args.task_replay_weight, sampling='uniform with replacement over all earlier observed roots; no outcome filtering',
        insertion='after current update optimization; current roots are not sampled as historical roots',
        labels='actual mean paired returns and sampling propensity under the recorded historical greedy continuation',
        interpretation='historical preference supervision; not an unbiased objective for the latest continuation',
        budget='same environment episodes and optimizer count; additional historical actor examples increase forward/backward work')
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    update, episodes, environment_steps, event_roots, best = 0, 0, 0, 0, None

    def state():
        return {k: v.detach().cpu().numpy().copy() for k, v in model.state_dict().items()}

    def save(path):
        payload = dict(config=asdict(cfg), state={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), protocol=protocol, update=update, selected=best, seed=args.seed,
            training_episodes=episodes, environment_steps=environment_steps, event_roots=event_roots,
            replay_rng=replay_rng.bit_generator.state, torch_rng=torch.get_rng_state())
        if args.task_replay_batch:
            payload['task_comparison_replay'] = dict(examples=task_comparisons,
                rng=task_replay_rng.bit_generator.state)
        atomic_save(payload, path)

    def status(completed=False):
        (out / 'status.json').write_text(json.dumps(dict(completed=completed, update=update,
            total_updates=args.iterations, training_episodes=episodes, environment_steps=environment_steps,
            event_roots=event_roots, best=best,
            task_replay_roots=len(task_comparisons),
            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2))

    def job(task, parameters, training, seed, **extra):
        return dict(task=task, state=parameters, training=training, policy_seed=int(seed),
            protocol=train_bank.get('protocol', {}), deadline=deadline, stop_file=stop_file, **extra)

    status()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=initialize,
            initargs=(args.project, args.source, asdict(cfg))) as pool:
        def validate():
            nonlocal best
            model.eval()
            parameters = state()
            results = list(pool.map(rollout, [job(t, parameters, False, args.seed + i)
                for i, t in enumerate(validation['tasks'])]))
            if any(not r['completed'] for r in results):
                with (out / 'validation.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(update=update, completed=False,
                        results=[{k: v for k, v in r.items() if k != 'transitions'} for r in results])) + '\n')
                return False
            rows = [r['record'] for r in results]
            success = sum(r['success'] for r in rows)
            collision = sum(r['collision'] for r in rows)
            capped = sum(r['duration'] if r['capture'] else 80 for r in rows) / len(rows)
            selection = [success, -collision, -capped]
            with (out / 'validation.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(update=update, completed=True, N=len(rows), success=success,
                    collision=collision, capped_capture_time=capped, tasks=rows)) + '\n')
            if best is None or tuple(selection) > tuple(best['selection_key']):
                best = dict(update=update, selection_key=selection, success=success, collision=collision, N=len(rows))
                save(out / 'best.pt')
            save(out / f'stage_{update:06d}.pt')
            save(out / 'last.pt')
            status()
            print(json.dumps(dict(validation_update=update, success=success, collision=collision, best=best)), flush=True)
            model.train()
            return True

        proceed = validate()
        while proceed and update < args.iterations and time.time() < deadline and not Path(stop_file).exists():
            parameters, jobs = state(), []
            records = schedule[update + 1]
            for group in range(4):
                original = records[4 * group:4 * group + 4]
                task = original[0]['task']
                if any(x['task'] != task for x in original):
                    raise ValueError('Original common-future root group is inconsistent')
                future_rng = np.random.default_rng(np.random.SeedSequence([args.seed, update + 1, group, int(task['future_seed'])]))
                futures = future_rng.integers(1, 2000000000, 2)
                for role in COMPARISON_PROPOSALS:
                    for replica, future in enumerate(futures):
                        jobs.append(job(dict(task), parameters, True, original[0]['policy_seed'],
                            proposal=role, future_replica=replica, branch_future_seed=int(future), root_group=group))
            results = list(pool.map(rollout, jobs))
            with (out / 'episodes.jsonl').open('a') as stream:
                for result in results:
                    stream.write(json.dumps(dict(update=update + 1,
                        **result.get('record', {k: v for k, v in result.items() if k != 'transitions'}))) + '\n')
            if any(not r['completed'] for r in results):
                break
            examples, missing = event_examples(results)
            episodes += len(results)
            environment_steps += sum(r['record']['steps'] for r in results)
            event_roots += len(examples)
            losses, kls, clipping, replay_losses = [], [], [], []
            task_replay_losses, current_actor_losses, task_replay_draws = [], [], 0
            if examples:
                batch = contact.input_batch([x['inputs'] for x in examples], args.device)
                candidates = batch['candidate_valid'].shape[1]
                old = torch.tensor(np.stack([np.pad(x['probabilities'], (0, candidates - len(x['probabilities'])))
                    for x in examples]), device=args.device)
                weights = torch.tensor(np.stack([np.pad(x.get('objective_weights', x['probabilities']),
                    (0, candidates - len(x['probabilities']))) for x in examples]), device=args.device)
                advantage = torch.tensor(np.stack([np.pad(x['advantage'], (0, candidates - len(x['advantage'])))
                    for x in examples]), device=args.device)
                valid = batch['candidate_valid']
            optimization_steps = optimization_schedule[update + 1] if optimization_schedule else args.epochs
            for _ in range(optimization_steps):
                if examples:
                    distribution = torch.distributions.Categorical(logits=model(batch)['logits'])
                    probabilities = distribution.probs
                    ratio = probabilities / old.clamp_min(1e-12)
                    surrogate = torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage)
                    if ACTOR_OBJECTIVE is None:
                        loss = -(weights * surrogate).sum(-1).mean() - args.entropy * distribution.entropy().mean()
                    else:
                        loss = ACTOR_OBJECTIVE(distribution.logits, examples)
                    current_actor_losses.append(float(loss.detach()))
                    with torch.no_grad():
                        kls.append(float((old * (old.clamp_min(1e-12).log() - probabilities.clamp_min(1e-12).log())).sum(-1).mean()))
                        if ACTOR_OBJECTIVE is None:
                            clipping.append(float((((ratio - 1).abs() > .2) & valid).sum() / valid.sum()))
                else:
                    loss = torch.zeros((), device=args.device)
                if args.task_replay_batch and task_comparisons:
                    indices = task_replay_rng.integers(0, len(task_comparisons), args.task_replay_batch)
                    old_examples = [task_comparisons[i] for i in indices]
                    old_batch = contact.input_batch([x['inputs'] for x in old_examples], args.device)
                    historical_loss = ACTOR_OBJECTIVE(model(old_batch)['logits'], old_examples)
                    mix = args.task_replay_weight if examples else 1.
                    loss = (1 - mix) * loss + mix * historical_loss
                    task_replay_losses.append(float(historical_loss.detach()))
                    task_replay_draws += len(old_examples)
                if args.replay_weight:
                    indices = replay_rng.integers(0, len(replay['history']), args.replay_batch)
                    rb = {k: v[indices].to(args.device) for k, v in replay.items()}
                    auxiliary = contact.contact_objective(model(rb), rb, cfg)['total']
                    loss = loss + args.replay_weight * auxiliary
                    replay_losses.append(float(auxiliary.detach()))
                if loss.requires_grad:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
                    losses.append(float(loss.detach()))
            if args.task_replay_batch:
                task_comparisons.extend(dict(x, source_update=update + 1) for x in examples)
            update += 1
            deltas = [x['comparison_return'] - x['teacher_return'] for x in examples]
            row = dict(update=update, training_episodes=episodes, environment_steps=environment_steps,
                N=len(results), success=sum(r['record']['success'] for r in results),
                collision=sum(r['record']['collision'] for r in results),
                observed_event_roots=len(examples), missed_event_roots=missing,
                distinct_proposal_roots=sum(len(x['probabilities']) > 1 for x in examples),
                policy_candidate_counts=[len(x['probabilities']) for x in examples],
                comparison_minus_teacher_return=deltas, optimization_batches=len(losses),
                loss=float(np.mean(losses)) if losses else 0., kl=float(np.mean(kls)) if kls else 0.,
                clip_fraction=float(np.mean(clipping)) if clipping else None,
                replay_loss=float(np.mean(replay_losses)) if replay_losses else None,
                current_actor_loss=float(np.mean(current_actor_losses)) if current_actor_losses else None,
                task_replay_loss=float(np.mean(task_replay_losses)) if task_replay_losses else None,
                task_replay_roots=len(task_comparisons), task_replay_draws=task_replay_draws)
            with (out / 'learning.jsonl').open('a') as stream:
                stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            status()
            if update % args.validation_every == 0 or update == args.iterations:
                proceed = validate()
            else:
                save(out / 'last.pt')
        save(out / 'last.pt')
    status(update == args.iterations and proceed)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        return train_main()
    pair.ContactPairConfig = EnumeratedPairConfig
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'], enumerated_pair=file_hash(__file__)))
        return value

    evaluation.runtime_identity = identity
    pair.main()


if __name__ == '__main__':
    main()
