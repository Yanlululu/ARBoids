"""Train current whole-configuration choice with ordinary task policy gradients.

The frozen supervised candidate-interaction encoder retains its geometric
meaning. A separate categorical action head learns from actual complete task
returns using clipped policy gradients and a within-layout sampled-return
baseline. There is no terminal-value, success-probability or CV predictor.
The monotone physical score initializes the policy; an unrestricted learned
logit can override it. All configurations in the common physical search bank
remain available. Training and greedy evaluation use the same bank.
"""
import argparse
import concurrent.futures
from dataclasses import asdict, dataclass, replace
import datetime
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn

from citf.bridge import Controller, Source, World, factorize, inputs
from citf.contracts import Protocol, canonical_hash, file_hash
from citf.search import nominal_beam, initialize_bank, whole_configuration_search
from citf.train import atomic_save
from experiments import joint_task_geometry as geometry


@dataclass(frozen=True)
class PolicyConfig(geometry.TaskGeometryConfig):
    revision: str = 'CITF-current-configuration-policy-v1'
    policy_prior_scale: float = 2.
    policy_logit_scale: float = 4.
    action_head_role: str = 'categorical current whole-configuration action, not a value prediction'
    training_signal: str = 'ordinary complete-task return with sampled group baseline; frozen geometry encoder'


def physical_components(batch, cfg):
    radius = batch['nominal'].norm(dim=-1)
    curve = radius[:, :, :-1] - radius[:, :, -1:]
    maximum = curve.masked_fill(~batch['known_time_valid'][:, None, None], -torch.inf).amax(-1)
    gap = maximum.masked_fill(~batch['entity_valid'][:, None, :-1], torch.inf).amin(-1)
    collision = geometry.guard.physical_pair_cost(batch, cfg.clearance)
    cost = batch['unary'] + cfg.objective_risk_weight * collision + cfg.guard_weight * (gap + cfg.guard_margin).clamp_min(0).square()
    return cost, collision, gap


class ConfigurationPolicy(nn.Module):
    def __init__(self, cfg=PolicyConfig()):
        super().__init__()
        self.cfg = cfg
        geometry.pair.pair_indices = geometry.all_pairs
        geometry.guard.nominal_gaps = geometry.team.nominal_team
        self.encoder = geometry.TaskGeometryPredictor(cfg)
        self.encoder.requires_grad_(False)
        self.actor = nn.Sequential(nn.LayerNorm(4 * cfg.width + 8),
            nn.Linear(4 * cfg.width + 8, cfg.width), nn.GELU(),
            nn.Linear(cfg.width, 1))
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self._cover, self._pairs = None, None
        self.encoder.guard_head.register_forward_pre_hook(self._capture_cover)
        self.encoder.distance_head.register_forward_pre_hook(self._capture_pairs)

    def _capture_cover(self, module, arguments):
        self._cover = arguments[0]

    def _capture_pairs(self, module, arguments):
        self._pairs = arguments[0]

    def encode_history(self, batch):
        return self.encoder.encode_history(batch)

    @torch.no_grad()
    def features(self, batch, history_cache=None):
        out = self.encoder(batch, history_cache=history_cache)
        roots, candidates = batch['candidate_valid'].shape
        width = self.cfg.width
        cover = self._cover.reshape(roots, candidates, -1, width)
        pairs = self._pairs.reshape(roots, candidates, -1, width)
        self._cover, self._pairs = None, None

        def pool(values, mask):
            mean = (values * mask[:, None, :, None]).sum(-2) / mask.sum(-1)[:, None, None].clamp_min(1)
            maximum = values.masked_fill(~mask[:, None, :, None], -torch.inf).amax(-2)
            return torch.cat((mean, maximum), -1)

        physical, collision, gap = physical_components(batch, self.cfg)
        signed_log = lambda value: value.sign() * torch.log1p(value.abs())
        scalar = torch.stack((batch['unary'] / 10, torch.log1p(collision), signed_log(gap / 10),
            torch.log1p((out['risk'] * out['pair_valid'][:, None]).sum(-1)),
            signed_log(out['team_mu'] / 10), torch.log1p(out['team_risk']),
            torch.log1p(out['pursuit_risk'].amin(-1)),
            batch['time_context'][:, :1].expand(-1, candidates)), -1)
        feature = torch.cat((pool(cover, batch['entity_valid'][:, :-1]),
                             pool(pairs, out['all_pair_valid']), scalar), -1)
        best = physical.masked_fill(~batch['candidate_valid'], torch.inf).amin(-1, keepdim=True)
        baseline = -torch.log1p((physical - best).clamp_min(0) / self.cfg.policy_prior_scale)
        return feature, baseline

    def action_logits(self, features, baseline, valid):
        residual = self.actor(features.float()).squeeze(-1) * self.cfg.policy_logit_scale
        return (baseline + residual).masked_fill(~valid, -torch.inf)

    def forward(self, batch, history_cache=None):
        features, baseline = self.features(batch, history_cache)
        return dict(features=features, baseline=baseline,
                    logits=self.action_logits(features, baseline, batch['candidate_valid']))


class PolicyController(Controller):
    def __init__(self, *args, stochastic=False, **kwargs):
        super().__init__(*args, **kwargs)
        if self.model is None:
            raise ValueError('The registered configuration policy needs its checkpoint')
        self.stochastic, self.transition = stochastic, None

    @torch.no_grad()
    def act(self, world):
        started = time.perf_counter()
        f = factorize(world, self.source)
        cfg = self.model.cfg
        ranked, _ = nominal_beam(f['unary'].numpy(), f['pair'].numpy(), cfg.objective_risk_weight, 128)
        initial, centre = initialize_bank(ranked, f['grid'].numpy(), f['valid'].numpy(), f['prior'].numpy(), self.bank, self.budget)

        def physical_score(plans):
            batch, _, _ = inputs(world, f, plans)
            return physical_components(batch, cfg)[0][0].numpy()

        _, trace = whole_configuration_search(initial, physical_score,
            [row.nonzero()[:, 0].numpy() for row in f['valid']], self.budget, self.rounds)
        plans = np.asarray(trace['configurations'], dtype=np.int64)
        batch, _, _ = inputs(world, f, plans)
        batch = {k: v.to(self.device) for k, v in batch.items()}
        output = self.model(batch)
        logits = output['logits'][0]
        distribution = torch.distributions.Categorical(logits=logits)
        if self.stochastic:
            choice = int(distribution.sample())
        else:
            values = logits.cpu().tolist()
            choice = min(range(len(plans)), key=lambda k: (-values[k], tuple(plans[k])))
        indices = plans[choice]
        gates = f['grid'][torch.arange(len(indices)), torch.as_tensor(indices)]
        force = (750 * (gates * f['learned'] + (1 - gates) * f['rules']) + 250).numpy()
        physical_choice = min(range(len(plans)), key=lambda k: (trace['scores'][k], tuple(plans[k])))
        self.transition = dict(features=output['features'][0].cpu().half(),
            baseline=output['baseline'][0].cpu(), action=choice,
            log_probability=float(distribution.log_prob(torch.as_tensor(choice, device=logits.device))),
            changed=int(choice != physical_choice), entropy=float(distribution.entropy()))
        trace.update(cost=float(-logits[choice]), scores=(-logits).cpu().tolist(),
            gates=gates.tolist(), selected_indices=indices.tolist(), physical_thrust=force.tolist(),
            physical_selected_indices=plans[physical_choice].tolist(),
            source_center_indices=centre.tolist(), source_center_scored=centre.tolist() in trace['configurations'],
            initial_configurations=initial.tolist(), selection='categorical learned action on common physical-search bank',
            compute_seconds=time.perf_counter() - started)
        return force, trace


_worker = {}


def initialize_worker(project, source, config, encoder_state):
    torch.set_num_threads(1)
    model = ConfigurationPolicy(PolicyConfig(**config))
    model.encoder.load_state_dict(encoder_state)
    model.eval()
    _worker.update(project=project, source=Source(project, source, 'cpu'), model=model)


def rollout(job):
    model = _worker['model']
    model.actor.load_state_dict(job['actor'])
    torch.manual_seed(job['policy_seed'])
    task = dict(job['task'])
    protocol = replace(Protocol(**job['protocol']), risk_weight=model.cfg.objective_risk_weight)
    world = World(_worker['project'], protocol, task['defenders'])
    world.reset(task['scene_seed'], task['future_seed'], task['agility'])
    controller = PolicyController(_worker['source'], model, stochastic=job['training'])
    transitions, changed, entropy, steps = [], 0, 0., 0
    while not world.done:
        if steps % 25 == 0 and (time.time() >= job['deadline'] or Path(job['stop_file']).exists()):
            return dict(completed=False, task=task, steps=steps, policy_seed=job['policy_seed'])
        force, _ = controller.act(world)
        result = world.step(force)
        transition = controller.transition
        changed += transition['changed']; entropy += transition['entropy']; steps += 1
        if job['training']:
            # NumPy payloads avoid exporting one shared-memory file descriptor
            # for every per-step tensor through the process result channel.
            transitions.append(dict(transition, features=transition['features'].numpy(),
                                    baseline=transition['baseline'].numpy()))
    reward = (1. if result['success'] else -1.) - .25 * result['collision']
    reward += .02 * result['capture'] - .0005 * result['duration']
    record = dict(completed=True, task=task, policy_seed=job['policy_seed'], steps=steps,
        reward=reward, changed=changed, mean_entropy=entropy / steps, **result)
    return dict(record=record, transitions=transitions)


def collate_transitions(results, advantages):
    transitions, values = [], []
    for episode, advantage in zip(results, advantages):
        transitions.extend(episode['transitions']); values.extend([advantage] * len(episode['transitions']))
    count = len(transitions)
    candidates = max(len(t['baseline']) for t in transitions)
    width = transitions[0]['features'].shape[-1]
    features = torch.zeros(count, candidates, width, dtype=torch.float16)
    baseline = torch.zeros(count, candidates)
    valid = torch.zeros(count, candidates, dtype=torch.bool)
    for row, transition in enumerate(transitions):
        n = len(transition['baseline'])
        features[row, :n] = torch.from_numpy(transition['features'])
        baseline[row, :n] = torch.from_numpy(transition['baseline']); valid[row, :n] = True
    return dict(features=features, baseline=baseline, valid=valid,
        action=torch.tensor([t['action'] for t in transitions]),
        old_logp=torch.tensor([t['log_probability'] for t in transitions]),
        advantage=torch.tensor(values, dtype=torch.float32))


def train_main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'source', 'pretrained', 'train-bank', 'validation-bank', 'out', 'deadline'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=160)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--groups', type=int, default=2)
    parser.add_argument('--group-size', type=int, default=4)
    parser.add_argument('--validation-every', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=192)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--entropy', type=float, default=.002)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_bank = json.loads(Path(args.train_bank).read_text())
    validation = json.loads(Path(args.validation_bank).read_text())
    if train_bank['role'] != 'train' or validation['role'] != 'validation':
        raise ValueError('Policy training and validation task roles differ')
    if {x['geometry_family_id'] for x in train_bank['tasks']} & {x['geometry_family_id'] for x in validation['tasks']}:
        raise ValueError('Training and validation layouts overlap')
    pretrained = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    config = dict(pretrained['config'], revision=PolicyConfig().revision)
    cfg = PolicyConfig(**config)
    model = ConfigurationPolicy(cfg).to(args.device)
    model.encoder.load_state_dict(pretrained['state'])
    model.eval()
    optimizer = torch.optim.AdamW(model.actor.parameters(), lr=args.lr, weight_decay=1e-4)
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    stop_file = str(out.parent / 'STOP_TRAINING')
    protocol = dict(pretrained['protocol'], config=asdict(cfg), arguments=vars(args),
        pretrained_geometry_checkpoint=args.pretrained, pretrained_geometry_sha256=file_hash(args.pretrained),
        objective='ordinary complete-task policy gradient; no value or outcome prediction head',
        source_unchanged=True, encoder_frozen=True, training_task_role='train',
        selection='greedy training-validation successes, fewer collisions, shorter capped capture time',
        runtime_code=canonical_hash(dict(core=file_hash(geometry.__file__), configuration_policy=file_hash(__file__))))
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    best, update, training_episodes, training_steps = None, 0, 0, 0

    def actor_state():
        return {k: v.detach().cpu().clone() for k, v in model.actor.state_dict().items()}

    def job(task, state, training, policy_seed):
        return dict(task=task, actor=state, training=training, policy_seed=int(policy_seed),
                    protocol=train_bank['protocol'], deadline=deadline, stop_file=stop_file)

    validated_update = None

    def checkpoint(path, selected=None):
        atomic_save(dict(state=model.state_dict(), config=asdict(cfg), protocol=protocol,
            selected=selected, update=update,
            completed=update == args.iterations and validated_update == update,
            validated_update=validated_update,
            optimizer=optimizer.state_dict(), sampler_rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(), training_episodes=training_episodes,
            training_steps=training_steps), path)

    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
            initializer=initialize_worker,
            initargs=(args.project, args.source, asdict(cfg), pretrained['state'])) as pool:
        def validate():
            nonlocal best, validated_update
            state = actor_state()
            results = list(pool.map(rollout, [job(task, state, False, args.seed + i)
                                           for i, task in enumerate(validation['tasks'])]))
            if any('record' not in result for result in results):
                return False
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
            checkpoint(out / 'last.pt', best)
            print(json.dumps(dict(validation=row | {'tasks': None}, best=best)), flush=True)
            return True

        proceed = validate()
        while proceed and update < args.iterations and time.time() < deadline and not Path(stop_file).exists():
            state = actor_state(); jobs = []
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
            data = collate_transitions(results, advantages)
            training_episodes += len(results); training_steps += len(data['action'])
            count = len(data['action']); losses, entropies, kls = [], [], []
            for epoch in range(args.epochs):
                indices = rng.permutation(count)
                for start in range(0, count, args.batch_size):
                    ids = indices[start:start + args.batch_size]
                    batch = {k: v[ids].to(args.device) for k, v in data.items()}
                    logits = model.action_logits(batch['features'], batch['baseline'], batch['valid'])
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(batch['action']); logratio = logp - batch['old_logp']
                    ratio = logratio.exp()
                    loss = -torch.minimum(ratio * batch['advantage'], ratio.clamp(.8, 1.2) * batch['advantage']).mean()
                    entropy = dist.entropy().mean(); loss = loss - args.entropy * entropy
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    nn.utils.clip_grad_norm_(model.actor.parameters(), 1.); optimizer.step()
                    losses.append(float(loss)); entropies.append(float(entropy))
                    kls.append(float(((ratio - 1) - logratio).mean()))
                if np.mean(kls[-max(1, math.ceil(count / args.batch_size)):]) > .08: break
            update += 1
            row = dict(update=update, episodes=training_episodes, environment_steps=training_steps,
                success=sum(r['record']['success'] for r in results), N=len(results),
                collision=sum(r['record']['collision'] for r in results),
                reward_mean=float(rewards.mean()), advantage_abs_mean=float(np.abs(advantages).mean()),
                policy_loss=float(np.mean(losses)), entropy=float(np.mean(entropies)), kl=float(np.mean(kls)))
            with (out / 'learning.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            if update % args.validation_every == 0 or update == args.iterations: proceed = validate()
            else: checkpoint(out / 'last.pt', best)
        checkpoint(out / 'last.pt', best)
    (out / 'status.json').write_text(json.dumps(dict(
        completed=update == args.iterations and validated_update == update,
        validated_update=validated_update,
        update=update, total_updates=args.iterations, best=best,
        training_episodes=training_episodes, environment_steps=training_steps), indent=2))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1); train_main(); return
    import citf.collect as collection
    import citf.evaluate as evaluation
    collection.Config, collection.Predictor = PolicyConfig, ConfigurationPolicy
    evaluation.Controller = PolicyController
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        result['method'] = 'CITF-CurrentConfigurationPolicy-development'
        result['controller'].update(action_head_role=PolicyConfig().action_head_role,
            encoder_frozen=True, selection='greedy learned logit on common physical-search bank')
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'], configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    evaluation.main()


if __name__ == '__main__':
    main()
