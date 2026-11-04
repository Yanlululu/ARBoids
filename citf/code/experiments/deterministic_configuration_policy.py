"""Refine greedy configuration decisions with antithetic parameter exploration.

The learned candidate-interaction encoder and hidden action readout are held
fixed. Complete task returns update the final current-action readout, using
the same layouts and future random seeds for positive and negative parameter
perturbations. Every rollout executes greedy current configuration choices;
there is no value model, terminal prediction, or future configuration plan.
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

from citf.contracts import canonical_hash, file_hash
from citf.train import atomic_save
from experiments import joint_configuration_policy as joint


@dataclass(frozen=True)
class DeterministicPolicyConfig(joint.JointPolicyConfig):
    revision: str = 'CITF-deterministic-readout-configuration-policy-v1'
    encoder_trainable: bool = False
    training_signal: str = 'ordinary complete-task returns with antithetic final-action-readout perturbations and greedy execution'


_original_worker = joint.initialize_worker


def initialize_worker(*args):
    joint.JointPolicyConfig = DeterministicPolicyConfig
    _original_worker(*args)


def rollout(job):
    job = dict(job)
    job['actor'] = {k: torch.from_numpy(v) for k, v in job['actor'].items()}
    result = joint.policy.rollout(job)
    if 'record' in result:
        record = result['record']
        duration = record['duration'] if record['capture'] else 80.
        record['reward'] = (1. if record['success'] else -1.) - .25 * record['collision']
        record['reward'] += .02 * record['capture'] - .0005 * duration
        record['reward_duration'] = 'capped_capture'
    return result


def train_main():
    parser = argparse.ArgumentParser()
    for name in ('project', 'source', 'start-policy', 'train-bank', 'validation-bank', 'out', 'deadline'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--iterations', type=int, default=40)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--groups', type=int, default=4)
    parser.add_argument('--directions', type=int, default=8)
    parser.add_argument('--noise', type=float, default=.025)
    parser.add_argument('--step-size', type=float, default=.01)
    parser.add_argument('--validation-every', type=int, default=10)
    parser.add_argument('--stage-every', type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    layout_rng = np.random.default_rng(args.seed + 200000)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    train_bank = json.loads(Path(args.train_bank).read_text())
    validation = json.loads(Path(args.validation_bank).read_text())
    if train_bank['role'] != 'train' or validation['role'] != 'validation':
        raise ValueError('Policy training and validation task roles differ')
    if {x['geometry_family_id'] for x in train_bank['tasks']} & {x['geometry_family_id'] for x in validation['tasks']}:
        raise ValueError('Training and validation layouts overlap')
    start_policy = torch.load(args.start_policy, map_location='cpu', weights_only=False)
    cfg = DeterministicPolicyConfig(**dict(start_policy['config'],
        revision=DeterministicPolicyConfig().revision, encoder_trainable=False,
        training_signal=DeterministicPolicyConfig().training_signal))
    model = joint.JointConfigurationPolicy(cfg)
    model.load_state_dict(start_policy['state']); model.requires_grad_(False); model.eval()
    if model.proposal is not None:
        raise ValueError('This paired readout experiment uses the common physical-search bank')
    actor_template = {k: v.detach().cpu().numpy().copy() for k, v in model.actor.state_dict().items()}
    weight_key, bias_key = '3.weight', '3.bias'
    weight_shape = actor_template[weight_key].shape
    weight_count = actor_template[weight_key].size
    theta = np.concatenate((actor_template[weight_key].reshape(-1), actor_template[bias_key].reshape(-1)))
    initial_theta = theta.copy()
    first_moment = np.zeros_like(theta); second_moment = np.zeros_like(theta)
    layout_order = layout_rng.permutation(len(train_bank['tasks']))
    layout_position = 0
    layout_exposure = np.zeros(len(train_bank['tasks']), dtype=np.int64)
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    stop_file = str(out.parent / 'STOP_TRAINING')
    protocol = dict(start_policy['protocol'], config=asdict(cfg), arguments=vars(args),
        starting_policy_checkpoint=args.start_policy, starting_policy_sha256=file_hash(args.start_policy),
        starting_policy_selected=start_policy.get('selected'), source_unchanged=True,
        encoder_frozen=True, hidden_action_readout_frozen=True,
        trained_parameters=[f'actor.{weight_key}', f'actor.{bias_key}'], parameter_count=len(theta),
        objective='ordinary complete-task reward; antithetic finite differences, normalized by paired task-return spread; Adam ascent',
        exploration='Gaussian parameter perturbations; every trajectory uses the same greedy rule as deployment',
        paired_control='all directions and both signs use identical training layouts and future seeds within each update',
        reward_duration='capped capture time: observed duration for capture, 80seconds otherwise; failed trajectories gain no reward for failing earlier',
        layout_sampling='shuffled complete training-layout cycles; no development-evaluation feedback in updates',
        selection='greedy training-validation successes, fewer collisions, shorter capped capture time',
        runtime_code=canonical_hash(dict(base=file_hash(joint.policy.__file__),
            joint_policy=file_hash(joint.__file__), deterministic_policy=file_hash(__file__))))
    (out / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    best, update, episodes, steps = None, 0, 0, 0

    def actor_state(vector):
        state = dict(actor_template)
        state[weight_key] = vector[:weight_count].reshape(weight_shape).astype(np.float32).copy()
        state[bias_key] = vector[weight_count:].astype(np.float32).copy()
        return state

    def set_parameters():
        state = {k: torch.from_numpy(v) for k, v in actor_state(theta).items()}
        model.actor.load_state_dict(state)

    def job(task, state, policy_seed):
        return dict(task=task, actor=state, training=False, policy_seed=int(policy_seed),
            protocol=train_bank['protocol'], deadline=deadline, stop_file=stop_file)

    validated_update = None

    def checkpoint(path, selected=None):
        set_parameters()
        atomic_save(dict(state=model.state_dict(), config=asdict(cfg), protocol=protocol,
            selected=selected, update=update,
            completed=update == args.iterations and validated_update == update,
            validated_update=validated_update,
            optimizer=dict(kind='Adam finite-difference ascent', first_moment=first_moment,
                second_moment=second_moment, steps=update),
            parameter_vector=theta.copy(), initial_parameter_vector=initial_theta,
            sampler_rng=rng.bit_generator.state, layout_rng=layout_rng.bit_generator.state,
            layout_order=layout_order.tolist(), layout_position=layout_position,
            layout_exposure=layout_exposure.tolist(), training_episodes=episodes,
            training_steps=steps, torch_rng=torch.get_rng_state()), path)

    context = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, mp_context=context,
            initializer=initialize_worker,
            initargs=(args.project, args.source, asdict(cfg), model.encoder.state_dict(), None)) as pool:
        def validate():
            nonlocal best, validated_update
            state = actor_state(theta)
            results = list(pool.map(rollout, [job(task, state, args.seed + i)
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
            if update == 0: checkpoint(out / 'initial.pt', best)
            checkpoint(out / 'last.pt', best)
            if update % args.stage_every == 0: checkpoint(out / f'stage_{update:06d}.pt', best)
            print(json.dumps(dict(validation=row | {'tasks': None}, best=best)), flush=True)
            return True

        proceed = validate()
        while proceed and update < args.iterations and time.time() < deadline and not Path(stop_file).exists():
            tasks = []
            for _ in range(args.groups):
                if layout_position == len(layout_order):
                    layout_order = layout_rng.permutation(len(train_bank['tasks'])); layout_position = 0
                index = int(layout_order[layout_position]); layout_position += 1; layout_exposure[index] += 1
                task = dict(train_bank['tasks'][index])
                task['future_seed'] = int(rng.integers(100000000, 2000000000))
                tasks.append(task)
            directions = rng.standard_normal((args.directions, len(theta))).astype(np.float32)
            jobs, identities = [], []
            for direction, perturbation in enumerate(directions):
                for sign in (1, -1):
                    state = actor_state(theta + sign * args.noise * perturbation)
                    for group, task in enumerate(tasks):
                        jobs.append(job(task, state, args.seed + update * args.groups + group))
                        identities.append(dict(direction=direction, sign=sign, group=group))
            results = list(pool.map(rollout, jobs))
            with (out / 'episodes.jsonl').open('a') as stream:
                for identity, result in zip(identities, results):
                    stream.write(json.dumps(dict(update=update + 1, **identity, **result.get('record', result))) + '\n')
            if any('record' not in result for result in results): break
            rewards = np.asarray([r['record']['reward'] for r in results]).reshape(args.directions, 2, args.groups)
            # Center within each identical task before estimating the scale;
            # easy-versus-hard layout variation must not dilute action signal.
            centered = rewards - rewards.mean((0, 1), keepdims=True)
            reward_scale = max(float(centered.std()), 1e-3)
            paired_difference = (rewards[:, 0] - rewards[:, 1]).mean(-1)
            gradient = (paired_difference[:, None] * directions).mean(0) / (2 * args.noise * reward_scale)
            first_moment = .9 * first_moment + .1 * gradient
            second_moment = .999 * second_moment + .001 * np.square(gradient)
            update += 1
            corrected_first = first_moment / (1 - .9 ** update)
            corrected_second = second_moment / (1 - .999 ** update)
            theta = (theta + args.step_size * corrected_first / (np.sqrt(corrected_second) + 1e-8)).astype(np.float32)
            episodes += len(results); steps += sum(r['record']['steps'] for r in results)
            row = dict(update=update, episodes=episodes, environment_steps=steps,
                covered_layouts=int((layout_exposure > 0).sum()), minimum_layout_visits=int(layout_exposure.min()),
                success=sum(r['record']['success'] for r in results), N=len(results),
                collision=sum(r['record']['collision'] for r in results),
                perturbed_policy_reward_mean=float(rewards.mean()), within_task_reward_scale=reward_scale,
                paired_return_difference_abs_mean=float(np.abs(paired_difference).mean()),
                gradient_norm=float(np.linalg.norm(gradient)),
                final_readout_change_l2=float(np.linalg.norm(theta - initial_theta)))
            with (out / 'learning.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            if update % args.validation_every == 0 or update == args.iterations: proceed = validate()
            else: checkpoint(out / 'last.pt', best)
        checkpoint(out / 'last.pt', best)
    (out / 'status.json').write_text(json.dumps(dict(
        completed=update == args.iterations and validated_update == update,
        validated_update=validated_update,
        update=update, total_updates=args.iterations, best=best, training_episodes=episodes,
        environment_steps=steps, covered_layouts=int((layout_exposure > 0).sum()),
        minimum_layout_visits=int(layout_exposure.min()), maximum_layout_visits=int(layout_exposure.max())), indent=2))


def main():
    joint.JointPolicyConfig = DeterministicPolicyConfig
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1); train_main(); return
    import citf.evaluate as evaluation
    previous = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        result['controller']['policy_training'] = DeterministicPolicyConfig().training_signal
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            deterministic_configuration_policy=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    joint.main()


if __name__ == '__main__':
    main()
