"""Jointly refine candidate interaction and current-configuration action choice.

The actor observes absolute interaction features and their standardized
differences over the same available configuration bank. Real task returns
train categorical current actions; supervised short-branch replay retains
the geometry prediction targets. There is no critic or terminal-value head.
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

from citf.bridge import Source, factorize, inputs
from citf.contracts import canonical_hash, file_hash
from citf.risk import signed_log1p
from citf.search import nominal_beam, initialize_bank, whole_configuration_search
from citf.train import atomic_save, load_shards
from experiments import configuration_policy as policy
from experiments import joint_stopped_motion as motion


@dataclass(frozen=True)
class JointPolicyConfig(policy.PolicyConfig):
    revision: str = 'CITF-joint-current-configuration-policy-v1'
    encoder_trainable: bool = True
    relative_feature_floor: float = .05
    training_signal: str = 'ordinary task policy gradient plus observed short-branch geometry replay'
    action_features: str = 'absolute and standardized within-bank candidate interaction features'
    action_bank: str = 'physical_search'
    proposal_config: dict | None = None


class JointConfigurationPolicy(policy.ConfigurationPolicy):
    def __init__(self, cfg=JointPolicyConfig()):
        super().__init__(cfg)
        self.encoder.requires_grad_(cfg.encoder_trainable)
        size = 4 * cfg.width + 8
        self.actor = nn.Sequential(nn.LayerNorm(2 * size), nn.Linear(2 * size, cfg.width),
                                   nn.GELU(), nn.Linear(cfg.width, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.capture_inputs = False
        self.rollout_inputs = None
        self.proposal = None
        if cfg.proposal_config is not None:
            self.proposal = motion.StoppedMotionPredictor(motion.StoppedMotionConfig(**cfg.proposal_config))
            self.proposal.requires_grad_(False)

    def features(self, batch, history_cache=None):
        out = self.encoder(batch, history_cache=history_cache)
        roots, candidates = batch['candidate_valid'].shape
        cover = self._cover.reshape(roots, candidates, -1, self.cfg.width)
        pairs = self._pairs.reshape(roots, candidates, -1, self.cfg.width)
        self._cover, self._pairs = None, None

        def pool(values, mask):
            mean = (values * mask[:, None, :, None]).sum(-2) / mask.sum(-1)[:, None, None].clamp_min(1)
            maximum = values.masked_fill(~mask[:, None, :, None], -torch.inf).amax(-2)
            return torch.cat((mean, maximum), -1)

        physical, collision, gap = policy.physical_components(batch, self.cfg)
        signed_log = signed_log1p
        scalar = torch.stack((batch['unary'] / 10, torch.log1p(collision), signed_log(gap / 10),
            torch.log1p((out['risk'] * out['pair_valid'][:, None]).sum(-1)),
            signed_log(out['team_mu'] / 10), torch.log1p(out['team_risk']),
            torch.log1p(out['pursuit_risk'].amin(-1)),
            batch['time_context'][:, :1].expand(-1, candidates)), -1)
        feature = torch.cat((pool(cover, batch['entity_valid'][:, :-1]),
                             pool(pairs, out['all_pair_valid']), scalar), -1)
        valid = batch['candidate_valid'][..., None]
        count = valid.sum(1, keepdim=True).clamp_min(1)
        center = (feature * valid).sum(1, keepdim=True) / count
        difference = feature - center
        scale = ((difference.square() * valid).sum(1, keepdim=True) / count).sqrt()
        relative = difference / scale.detach().clamp_min(self.cfg.relative_feature_floor)
        best = physical.masked_fill(~batch['candidate_valid'], torch.inf).amin(-1, keepdim=True)
        baseline = -torch.log1p((physical - best).clamp_min(0) / self.cfg.policy_prior_scale)
        return torch.cat((feature, relative), -1), baseline

    def forward(self, batch, history_cache=None):
        if self.capture_inputs:
            self.rollout_inputs = {k: v[0].detach().cpu().numpy().copy() for k, v in batch.items()}
        return super().forward(batch, history_cache)


class JointPolicyController(policy.PolicyController):
    @torch.no_grad()
    def act(self, world):
        self.model.capture_inputs = self.stochastic
        result = self.expert_act(world) if self.model.cfg.action_bank == 'learned_experts' else super().act(world)
        if self.stochastic:
            self.transition['inputs'] = self.model.rollout_inputs
            self.model.rollout_inputs = None
        return result

    @torch.no_grad()
    def expert_act(self, world):
        started = time.perf_counter()
        f, cfg = factorize(world, self.source), self.model.cfg
        ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), cfg.objective_risk_weight, 128)
        initial, centre = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(),
                                          f['prior'].numpy(), self.bank, self.budget)
        choices = [row.nonzero()[:, 0].numpy() for row in f['valid']]

        def physical_score(plans):
            batch, _, _ = inputs(world, f, plans)
            return policy.physical_components(batch, cfg)[0][0].numpy()

        physical_choice, physical_trace = whole_configuration_search(
            initial, physical_score, choices, self.budget, self.rounds)
        first, _, _ = inputs(world, f, initial[:1])
        first = {k: v.to(self.device) for k, v in first.items()}
        history_cache = self.model.proposal.encode_history(first)

        def learned_score(plans):
            batch, _, _ = inputs(world, f, plans)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            prediction = self.model.proposal(batch, history_cache=history_cache)
            return policy.geometry.costs(prediction, batch, cfg.clearance)[0].cpu().numpy()

        learned_choice, learned_trace = whole_configuration_search(
            initial, learned_score, choices, self.budget, self.rounds)
        plans = np.asarray(list(dict.fromkeys(tuple(map(int, row))
            for row in [physical_choice, learned_choice, centre])), dtype=np.int64)
        batch, _, _ = inputs(world, f, plans)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        output = self.model(batch)
        logits = output['logits'][0]
        distribution = torch.distributions.Categorical(logits=logits)
        choice = int(distribution.sample()) if self.stochastic else min(
            range(len(plans)), key=lambda k: (-float(logits[k]), tuple(plans[k])))
        indices = plans[choice]
        gates = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * f['learned'] + (1 - gates) * f['rules']) + 250).numpy()
        self.transition = dict(features=output['features'][0].cpu().half(),
            baseline=output['baseline'][0].cpu(), action=choice,
            log_probability=float(distribution.log_prob(torch.as_tensor(choice, device=logits.device))),
            changed=int(not np.array_equal(indices, physical_choice)), entropy=float(distribution.entropy()))
        all_scored = set(map(tuple, physical_trace['configurations'])) | set(map(tuple, learned_trace['configurations']))
        trace = dict(cost=float(-logits[choice]), evaluated_unique=len(plans),
            forward_batches=physical_trace['forward_batches']+learned_trace['forward_batches']+1,
            configurations=plans.tolist(), scores=(-logits).cpu().tolist(),
            gates=gates.tolist(), selected_indices=indices.tolist(), physical_thrust=force.tolist(),
            physical_selected_indices=physical_choice.tolist(), learned_selected_indices=learned_choice.tolist(),
            source_center_indices=centre.tolist(), source_center_scored=centre.tolist() in plans.tolist(),
            initial_configurations=initial.tolist(), selection='learned current choice of physical, motion, and source proposals',
            physical_scored_unique=physical_trace['evaluated_unique'],
            predictor_scored_unique=learned_trace['evaluated_unique'],
            proposal_scored_unique_union=len(all_scored), compute_seconds=time.perf_counter() - started)
        return force, trace


def initialize_worker(project, source, config, encoder_state, proposal_state):
    torch.set_num_threads(1)
    policy.PolicyController = JointPolicyController
    model = JointConfigurationPolicy(JointPolicyConfig(**config))
    model.encoder.load_state_dict(encoder_state)
    if proposal_state is not None:
        model.proposal.load_state_dict(proposal_state)
    model.eval()
    policy._worker.update(project=project, source=Source(project, source, 'cpu'), model=model)


def rollout(job):
    job = dict(job)
    state = {k: torch.from_numpy(v) for k, v in job.pop('encoder').items()}
    policy._worker['model'].encoder.load_state_dict(state)
    job['actor'] = {k: torch.from_numpy(v) for k, v in job['actor'].items()}
    return policy.rollout(job)


def input_batch(rows, device):
    candidate_keys = {'nominal', 'candidate', 'unary', 'candidate_valid'}
    output = {}
    for key in rows[0]:
        values = [torch.from_numpy(row[key]) for row in rows]
        if key in candidate_keys:
            value = nn.utils.rnn.pad_sequence(values, batch_first=True)
        else:
            value = torch.stack(values)
        output[key] = value.to(device)
    return output


def train_main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'source', 'pretrained', 'train-bank', 'validation-bank', 'out', 'deadline'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=160)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--groups', type=int, default=2)
    parser.add_argument('--group-size', type=int, default=4)
    parser.add_argument('--validation-every', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=48)
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
    if train_bank['role'] != 'train' or validation['role'] != 'validation':
        raise ValueError('Policy training and validation task roles differ')
    if {x['geometry_family_id'] for x in train_bank['tasks']} & {x['geometry_family_id'] for x in validation['tasks']}:
        raise ValueError('Training and validation layouts overlap')
    pretrained = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    proposal = torch.load(args.proposal_checkpoint, map_location='cpu', weights_only=False) if args.proposal_checkpoint else None
    cfg = JointPolicyConfig(**dict(pretrained['config'], revision=JointPolicyConfig().revision,
        encoder_trainable=not args.freeze_encoder,
        training_signal=JointPolicyConfig().training_signal,
        action_bank='learned_experts' if proposal is not None else 'physical_search',
        proposal_config=proposal['config'] if proposal is not None else None))
    model = JointConfigurationPolicy(cfg).to(args.device)
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
        proposal_checkpoint=args.proposal_checkpoint,
        proposal_sha256=file_hash(args.proposal_checkpoint) if args.proposal_checkpoint else None,
        selection='greedy training-validation successes, fewer collisions, shorter capped capture time',
        runtime_code=canonical_hash(dict(base=file_hash(policy.__file__), joint_policy=file_hash(__file__))))
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    best, update, episodes, steps = None, 0, 0, 0

    def policy_state():
        return dict(actor={k: v.detach().cpu().numpy().copy() for k, v in model.actor.state_dict().items()},
                    encoder={k: v.detach().cpu().numpy().copy() for k, v in model.encoder.state_dict().items()})

    def job(task, state, training, policy_seed):
        return dict(task=task, **state, training=training, policy_seed=int(policy_seed),
                    protocol=train_bank['protocol'], deadline=deadline, stop_file=stop_file)

    validated_update = None

    def checkpoint(path, selected=None):
        atomic_save(dict(state=model.state_dict(), config=asdict(cfg), protocol=protocol,
            selected=selected, update=update,
            completed=update == args.iterations and validated_update == update,
            validated_update=validated_update,
            optimizer=optimizer.state_dict(), sampler_rng=rng.bit_generator.state,
            replay_rng=replay_rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            training_episodes=episodes, training_steps=steps), path)

    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
            initializer=initialize_worker,
            initargs=(args.project, args.source, asdict(cfg), pretrained['state'],
                      proposal['state'] if proposal is not None else None)) as pool:
        def validate():
            nonlocal best, validated_update
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
            validated_update = update
            key = (success, -collision, -capped)
            if best is None or key > tuple(best['selection_key']):
                best = dict(update=update, selection_key=list(key), success=success, N=len(records),
                            collision=collision, capped_capture_time=capped)
                checkpoint(out / 'best.pt', best)
            if update == 0:
                checkpoint(out / 'initial.pt', best)
            checkpoint(out / 'last.pt', best)
            print(json.dumps(dict(validation=row | {'tasks': None}, best=best)), flush=True)
            return True

        proceed = validate()
        while proceed and update < args.iterations and time.time() < deadline and not Path(stop_file).exists():
            state = policy_state(); jobs = []
            for _ in range(args.groups):
                task = dict(train_bank['tasks'][int(rng.integers(len(train_bank['tasks'])))])
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
                if np.mean(kls[-max(1, math.ceil(count / args.batch_size)):]) > .08: break
            update += 1
            row = dict(update=update, episodes=episodes, environment_steps=steps,
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
    (out / 'status.json').write_text(json.dumps(dict(
        completed=update == args.iterations and validated_update == update,
        validated_update=validated_update,
        update=update, total_updates=args.iterations, best=best,
        training_episodes=episodes, environment_steps=steps), indent=2))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1); train_main(); return
    import citf.collect as collection
    import citf.evaluate as evaluation
    collection.Config, collection.Predictor = JointPolicyConfig, JointConfigurationPolicy
    evaluation.Controller = JointPolicyController
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        checkpoint = args[2] if len(args) > 2 else kwargs.get('checkpoint')
        cfg = torch.load(checkpoint, map_location='cpu', weights_only=False)['config']
        result['method'] = 'CITF-JointCurrentConfigurationPolicy-development'
        result['controller'].update(action_head_role=JointPolicyConfig().action_head_role,
            action_features=JointPolicyConfig().action_features,
            action_bank=cfg['action_bank'], encoder_trainable=cfg['encoder_trainable'],
            selection='greedy learned logit on the declared current configuration proposals')
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            configuration_policy=file_hash(policy.__file__), joint_configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    evaluation.main()


if __name__ == '__main__':
    main()
