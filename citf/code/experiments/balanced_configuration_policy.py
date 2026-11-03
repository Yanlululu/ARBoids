"""Train current configuration choice with balanced layouts and larger groups.

The actor, candidate interaction encoder, source and task return are retained.
Complete shuffled training-layout cycles and 32-episode batches expose more
independent configurations before each update. A declared categorical
sampling temperature controls exploration; greedy evaluation is unchanged.
"""
import argparse
import concurrent.futures
from dataclasses import asdict, dataclass
import datetime
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

from citf.contracts import canonical_hash, file_hash
from citf.train import atomic_save, load_shards
from experiments import joint_configuration_policy as joint
from experiments import joint_stopped_motion as motion

policy = joint.policy
rollout = joint.rollout
input_batch = joint.input_batch


@dataclass(frozen=True)
class BalancedPolicyConfig(joint.JointPolicyConfig):
    revision: str = 'CITF-balanced-current-configuration-policy-v1'
    policy_temperature: float = .5
    training_signal: str = 'ordinary task policy gradient on balanced layout cycles plus stopped-geometry replay'


class BalancedConfigurationPolicy(joint.JointConfigurationPolicy):
    def action_logits(self, features, baseline, valid):
        return super().action_logits(features, baseline, valid) / self.cfg.policy_temperature


_original_worker = joint.initialize_worker


def initialize_worker(*args):
    joint.JointPolicyConfig = BalancedPolicyConfig
    joint.JointConfigurationPolicy = BalancedConfigurationPolicy
    _original_worker(*args)


def train_main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'source', 'pretrained', 'train-bank', 'validation-bank', 'out', 'deadline'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=160)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--groups', type=int, default=8)
    parser.add_argument('--group-size', type=int, default=4)
    parser.add_argument('--validation-every', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=96)
    parser.add_argument('--policy-temperature', type=float, default=.5)
    parser.add_argument('--stage-every', type=int, default=40)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--encoder-lr', type=float, default=1e-5)
    parser.add_argument('--entropy', type=float, default=.002)
    parser.add_argument('--replay-weight', type=float, default=.05)
    parser.add_argument('--replay-batch', type=int, default=8)
    parser.add_argument('--freeze-encoder', action='store_true')
    parser.add_argument('--proposal-checkpoint')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    replay_rng = np.random.default_rng(args.seed + 100000)
    train_bank = json.loads(Path(args.train_bank).read_text())
    validation = json.loads(Path(args.validation_bank).read_text())
    layout_rng = np.random.default_rng(args.seed + 200000)
    layout_order = layout_rng.permutation(len(train_bank['tasks']))
    layout_position = 0
    layout_exposure = np.zeros(len(train_bank['tasks']), dtype=np.int64)
    if train_bank['role'] != 'train' or validation['role'] != 'validation':
        raise ValueError('Policy training and validation task roles differ')
    if {x['geometry_family_id'] for x in train_bank['tasks']} & {x['geometry_family_id'] for x in validation['tasks']}:
        raise ValueError('Training and validation layouts overlap')
    pretrained = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    proposal = torch.load(args.proposal_checkpoint, map_location='cpu', weights_only=False) if args.proposal_checkpoint else None
    cfg = BalancedPolicyConfig(**dict(pretrained['config'], revision=BalancedPolicyConfig().revision,
        encoder_trainable=not args.freeze_encoder, policy_temperature=args.policy_temperature,
        training_signal=BalancedPolicyConfig().training_signal,
        action_bank='learned_experts' if proposal is not None else 'physical_search',
        proposal_config=proposal['config'] if proposal is not None else None))
    model = BalancedConfigurationPolicy(cfg).to(args.device)
    model.encoder.load_state_dict(pretrained['state']); model.eval()
    if proposal is not None:
        model.proposal.load_state_dict(proposal['state'])
    groups = [dict(params=model.actor.parameters(), lr=args.lr)]
    if cfg.encoder_trainable:
        groups.append(dict(params=model.encoder.parameters(), lr=args.encoder_lr))
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    replay_paths = pretrained['protocol']['arguments']['train']
    replay = None
    if cfg.encoder_trainable and args.replay_weight:
        replay, roots, blobs = load_shards(replay_paths, 'train')
        if {x['geometry_family_id'] for x in roots} & {x['geometry_family_id'] for x in validation['tasks']}:
            raise ValueError('Supervised training replay and policy validation layouts overlap')
        del roots, blobs
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    stop_file = str(out.parent / 'STOP_TRAINING')
    protocol = dict(pretrained['protocol'], config=asdict(cfg), arguments=vars(args),
        pretrained_geometry_checkpoint=args.pretrained, pretrained_geometry_sha256=file_hash(args.pretrained),
        objective='ordinary complete-task clipped policy gradient plus stopped-geometry replay; no value head',
        source_unchanged=True, encoder_frozen=not cfg.encoder_trainable, supervised_replay_paths=replay_paths,
        layout_sampling='shuffled complete cycles, preserving every declared training layout',
        proposal_checkpoint=args.proposal_checkpoint,
        proposal_sha256=file_hash(args.proposal_checkpoint) if args.proposal_checkpoint else None,
        selection='greedy training-validation successes, fewer collisions, shorter capped capture time',
        runtime_code=canonical_hash(dict(base=file_hash(policy.__file__),
            joint_policy=file_hash(joint.__file__), balanced_policy=file_hash(__file__))))
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    best, update, episodes, steps = None, 0, 0, 0

    def policy_state():
        return dict(actor={k: v.detach().cpu().numpy().copy() for k, v in model.actor.state_dict().items()},
                    encoder={k: v.detach().cpu().numpy().copy() for k, v in model.encoder.state_dict().items()})

    def job(task, state, training, policy_seed):
        return dict(task=task, **state, training=training, policy_seed=int(policy_seed),
                    protocol=train_bank['protocol'], deadline=deadline, stop_file=stop_file)

    def checkpoint(path, selected=None):
        atomic_save(dict(state=model.state_dict(), config=asdict(cfg), protocol=protocol,
            selected=selected, update=update, completed=update == args.iterations,
            optimizer=optimizer.state_dict(), sampler_rng=rng.bit_generator.state,
            replay_rng=replay_rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            training_episodes=episodes, training_steps=steps, layout_rng=layout_rng.bit_generator.state,
            layout_order=layout_order.tolist(), layout_position=layout_position,
            layout_exposure=layout_exposure.tolist()), path)

    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
            initializer=initialize_worker,
            initargs=(args.project, args.source, asdict(cfg), pretrained['state'],
                      proposal['state'] if proposal is not None else None)) as pool:
        def validate():
            nonlocal best
            state = policy_state()
            results = list(pool.map(rollout, [job(task, state, False, args.seed + i)
                                           for i, task in enumerate(validation['tasks'])]))
            if any('record' not in result for result in results): return False
            records = [r['record'] for r in results]
            success = sum(r['success'] for r in records); collision = sum(r['collision'] for r in records)
            capped = sum(r['duration'] if r['capture'] else 80 for r in records) / len(records)
            row = dict(update=update, N=len(records), success=success, collision=collision,
                       capped_capture_time=capped, tasks=records)
            with (out / 'validation.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
            key = (success, -collision, -capped)
            if best is None or key > tuple(best['selection_key']):
                best = dict(update=update, selection_key=list(key), success=success, N=len(records),
                            collision=collision, capped_capture_time=capped)
                checkpoint(out / 'best.pt', best)
            if update == 0:
                checkpoint(out / 'initial.pt', best)
            checkpoint(out / 'last.pt', best)
            if update % args.stage_every == 0:
                checkpoint(out / f'stage_{update:06d}.pt', best)
            print(json.dumps(dict(validation=row | {'tasks': None}, best=best)), flush=True)
            return True

        proceed = validate()
        while proceed and update < args.iterations and time.time() < deadline and not Path(stop_file).exists():
            state = policy_state(); jobs = []
            for _ in range(args.groups):
                if layout_position == len(layout_order):
                    layout_order = layout_rng.permutation(len(train_bank['tasks']))
                    layout_position = 0
                index = int(layout_order[layout_position])
                layout_position += 1
                layout_exposure[index] += 1
                task = dict(train_bank['tasks'][index])
                task['future_seed'] = int(rng.integers(100000000, 2000000000))
                for _ in range(args.group_size):
                    jobs.append(job(task, state, True, rng.integers(1, 2000000000)))
            results = list(pool.map(rollout, jobs))
            with (out / 'episodes.jsonl').open('a') as stream:
                for result in results:
                    stream.write(json.dumps(dict(update=update + 1, **result.get('record', result))) + '\n')
            if any('record' not in result for result in results): break
            rewards = np.asarray([r['record']['reward'] for r in results]).reshape(args.groups, args.group_size)
            advantages = ((rewards - rewards.mean(1, keepdims=True)) * args.group_size / (args.group_size - 1)).reshape(-1)
            data = policy.collate_transitions(results, advantages)
            raw_inputs = [transition['inputs'] for result in results for transition in result['transitions']]
            episodes += len(results); steps += len(data['action'])
            count = len(raw_inputs); losses, entropies, kls, supervised = [], [], [], []
            for epoch in range(args.epochs):
                indices = rng.permutation(count)
                for start in range(0, count, args.batch_size):
                    ids = indices[start:start + args.batch_size]
                    batch = input_batch([raw_inputs[i] for i in ids], args.device)
                    output = model(batch)
                    dist = torch.distributions.Categorical(logits=output['logits'])
                    logp = dist.log_prob(data['action'][ids].to(args.device))
                    advantage = data['advantage'][ids].to(args.device)
                    logratio = logp - data['old_logp'][ids].to(args.device); ratio = logratio.exp()
                    loss = -torch.minimum(ratio * advantage, ratio.clamp(.8, 1.2) * advantage).mean()
                    entropy = dist.entropy().mean(); loss = loss - args.entropy * entropy
                    if replay is not None:
                        ri = replay_rng.integers(0, len(replay['history']), args.replay_batch)
                        rb = {k: v[ri].to(args.device) for k, v in replay.items()}
                        auxiliary = policy.geometry.task_objective(model.encoder(rb), rb,
                            clearance=cfg.clearance, mode='full', risk_weight=1., rank_weight=.5, temperature=2.)['total']
                        model._cover, model._pairs = None, None
                        loss = loss + args.replay_weight * auxiliary
                        supervised.append(float(auxiliary.detach()))
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                    losses.append(float(loss.detach())); entropies.append(float(entropy.detach()))
                    kls.append(float(((ratio - 1) - logratio).mean().detach()))
                if np.mean(kls[-max(1, math.ceil(count / args.batch_size)):]) > .04: break
            update += 1
            row = dict(update=update, episodes=episodes, environment_steps=steps,
                covered_layouts=int((layout_exposure > 0).sum()), minimum_layout_visits=int(layout_exposure.min()),
                success=sum(r['record']['success'] for r in results), N=len(results),
                collision=sum(r['record']['collision'] for r in results),
                reward_mean=float(rewards.mean()), advantage_abs_mean=float(np.abs(advantages).mean()),
                policy_loss=float(np.mean(losses)), entropy=float(np.mean(entropies)), kl=float(np.mean(kls)),
                supervised_geometry_loss=float(np.mean(supervised)) if supervised else None)
            with (out / 'learning.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            if update % args.validation_every == 0 or update == args.iterations: proceed = validate()
            else: checkpoint(out / 'last.pt', best)
        checkpoint(out / 'last.pt', best)
    (out / 'status.json').write_text(json.dumps(dict(completed=update == args.iterations,
        update=update, total_updates=args.iterations, best=best,
        training_episodes=episodes, environment_steps=steps, covered_layouts=int((layout_exposure > 0).sum()),
        minimum_layout_visits=int(layout_exposure.min()), maximum_layout_visits=int(layout_exposure.max())), indent=2))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1)
        train_main()
        return
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        result['controller']['policy_training'] = BalancedPolicyConfig().training_signal
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            balanced_configuration_policy=file_hash(__file__)))
        return result

    joint.JointPolicyConfig = BalancedPolicyConfig
    joint.JointConfigurationPolicy = BalancedConfigurationPolicy
    evaluation.runtime_identity = identity
    joint.main()


if __name__ == '__main__':
    main()
