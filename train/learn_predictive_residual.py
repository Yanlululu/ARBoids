"""Fit real-return corrections to a predictive conditional composition value."""
import study_runtime
import argparse
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

from distill_role_value import initialize, rollout
from envs.TADgame import TADEnv
from envs.snapshot import preserved_random_state, seed_random
from feedback_joint_control import observe, nominal_attacker_command
from interaction_rollout import compact_snapshot, execute, public_packet
from policy.role_value import compositions
from role_prediction import RolePrediction
from train_interaction import atomic_json, outcome_row


def prediction_features(controller, measurement, predictions):
    remaining = measurement.total_time-measurement.time
    raw = np.array([p['estimated_capture_time'] for p in predictions])
    bounded = np.minimum(raw, remaining)/60.
    count = compositions(controller.defenders).sum(1)/2.
    outcome = np.array([p['continuation_outcome'] for p in predictions])
    agility, uncertainty = controller.attacker_agility/8., controller.observer.uncertainty/8.
    margin = np.array([p['guard_margin'] for p in predictions])/30.
    separation = np.array([p['minimum_distance'] for p in predictions])/10.
    features = np.column_stack((np.ones(len(raw)), bounded, bounded**2,
        outcome == 3, outcome == 1, outcome == 2, outcome == 4,
        margin, separation, count, bounded*count, bounded*agility,
        bounded*uncertainty, count*agility, margin*agility, margin*count,
        bounded*remaining/60.))
    # The learned part corrects task time; nominal collision/breach priorities
    # remain exactly those of the strong predictive controller.
    return features, -np.minimum(raw, remaining)/20.


def neighboring_choices(predictions, n):
    best = min(range(len(predictions)), key=lambda i: predictions[i]['score'])
    masks = compositions(n)
    neighbors = np.where(abs(masks-masks[best]).sum(1) == 1)[0]
    alternative = min(neighbors, key=lambda i: predictions[i]['score'])
    return np.array([best, alternative], dtype=int)


class CorrectedPrediction(RolePrediction):
    def __init__(self, tail_steps, coefficients=None, strength=1., local_improvement=False):
        super().__init__(6, tail_steps=tail_steps, tail_policy='candidate')
        self.coefficients, self.strength = coefficients, strength
        self.local_improvement = local_improvement
        self.predictions, self.changed = [], 0

    def predict(self, measurement, template):
        if template == 'baseline':
            self.predictions = []
        predicted = super().predict(measurement, template)
        self.predictions.append(dict(predicted))
        return predicted

    def nominal(self, measurement, template):
        # Called inside each forecast and once for the actual feedback. All
        # candidate forecasts have finished only at the actual decision call.
        if self.remaining == self.block_steps and len(self.predictions) == len(self.options):
            original = template
            if self.coefficients is not None:
                x, _ = prediction_features(self, measurement, self.predictions)
                correction = self.strength*(x @ self.coefficients)*20.
                choices = (neighboring_choices(self.predictions, self.defenders) if self.local_improvement
                           else range(len(self.predictions)))
                choice = min(choices, key=lambda i:
                    (*self.predictions[i]['score'][:2], self.predictions[i]['score'][2]-correction[i]))
                self.plan = self.predictions[choice]
                self.template = template = self.plan['template']
                self.changed += int(template != original)
            self.predictions_at_decision = self.predictions
            self.predictions = []
        return super().nominal(measurement, template)


def predictive_continuation(branch, controller, mask):
    """Intervene for one block, then use the same frozen predictive policy."""
    future = RolePrediction(6, tail_steps=controller.tail_steps, tail_policy='candidate')
    future.observer = deepcopy(controller.observer)
    future.attacker_agility = controller.attacker_agility
    packet, ended, step = public_packet(branch), 0, 0
    while not ended:
        measurement = observe(branch, packet['obs'])
        if step < controller.block_steps:
            if step:
                future.attacker_agility = future.observer.update(measurement)
            action = controller.residual.compose(packet, mask)
            packet, _, ended, force, _, _ = execute(branch, packet, action)
            future.previous_attacker_thrust = nominal_attacker_command(measurement, future.attacker_agility)
            future.previous_thrust = force.copy()
            future.tick += 1
        else:
            force, _ = future.control(measurement)
            obs, _, ended, _ = branch.step(np.zeros((6, 3)), 'AdaRes', defender_thrust=force)
            packet = public_packet(branch, obs)
        step += 1
    return outcome_row(branch, ended)


def collect(task):
    seed, agility, tail_steps, repetitions, predictive_future = task[:5]
    previous = {int(r['step']): r for r in task[5]} if len(task) > 5 else None
    seed_random(seed)
    env = TADEnv(6, protocol='paper-parameters-v1')
    obs, _ = env.reset(agility, noisy_agility=False)
    controller = CorrectedPrediction(tail_steps)
    np.random.seed(seed+1000000)
    records, done, step = [], 0, 0
    while not done:
        measurement = observe(env, obs)
        force, _ = controller.control(measurement)
        if step in (0, 20, 50, 90, 130):
            snapshot = compact_snapshot(env)
            predictions = controller.predictions_at_decision
            x, q_model = prediction_features(controller, measurement, predictions)
            choices = (neighboring_choices(predictions, 6) if predictive_future else np.arange(len(predictions)))
            labels = []
            if previous is not None:
                old = previous.pop(step)
                for name, value in (('features', x[choices]), ('q_model', q_model[choices]), ('choices', choices)):
                    np.testing.assert_array_equal(old[name], value)
                if float(old['time']) != measurement.time:
                    raise ValueError('Augmented training state changed.')
                labels = list(old['outcomes'])
                if len(labels) >= repetitions:
                    raise ValueError('Augmentation must add independent futures.')
            with preserved_random_state():
                for repetition in range(len(labels), repetitions):
                    branch_rows = []
                    future_seed = (seed*37+step*1009+repetition*100000007) % 2**32
                    for mask in compositions(6)[choices]:
                        branch = snapshot.restore(future_seed=future_seed)
                        if predictive_future:
                            row = predictive_continuation(branch, controller, mask)
                        else:
                            packet, ended = public_packet(branch), 0
                            while not ended:
                                action = controller.residual.compose(packet, mask)
                                packet, _, ended, _, _, _ = execute(branch, packet, action)
                            row = outcome_row(branch, ended)
                        branch_rows.append([row[k] for k in ('capture_time', 'capture', 'success', 'collision')])
                    labels.append(branch_rows)
            records.append(dict(scene_seed=seed, agility=agility, step=step, time=measurement.time,
                features=x[choices], q_model=q_model[choices], choices=choices, outcomes=np.array(labels)))
        obs, _, done, _ = env.step(np.zeros((6, 3)), 'AdaRes', defender_thrust=force)
        step += 1
    if previous:
        raise ValueError('Some original training states were not reproduced.')
    return records


def fit(data, arm):
    x, model = data['features'], data['q_model']
    y = -(data['outcomes'][..., 0].mean(1)-data['time'][:, None])/20.
    residual = y-model
    if arm == 'conditional':
        if x.shape[1] == 2:
            hi, lo = [1], [0]
        else:
            masks = compositions(6)
            delta = masks[:, None]-masks[None]
            hi, lo = np.where((abs(delta).sum(-1) == 1) & (delta.sum(-1) == 1))
        x, residual = x[:, hi]-x[:, lo], residual[:, hi]-residual[:, lo]
    x, residual = x.reshape(-1, x.shape[-1]), residual.reshape(-1)
    coefficients = np.linalg.solve(x.T@x/len(x)+.01*np.eye(x.shape[1]), x.T@residual/len(x))
    return coefficients, float(np.mean((x@coefficients-residual)**2))


def evaluate(task):
    arm, coefficients, strength, tail, seed, agility, local_improvement = task
    row, controller = rollout(seed, agility, lambda: CorrectedPrediction(tail, coefficients, strength, local_improvement))
    return dict(row, arm=arm, strength=strength, scene_seed=seed, agility=agility,
                decisions=controller.plan_count, changed=controller.changed)


def summarize(rows):
    result = {}
    for arm, strength, agility in sorted({(r['arm'], r['strength'], r['agility']) for r in rows}):
        group = [r for r in rows if (r['arm'], r['strength'], r['agility']) == (arm, strength, agility)]
        result[f'{arm}-{strength:g}-a{agility:g}'] = dict(episodes=len(group), **{
            k: float(np.mean([r[k] for r in group])) for k in
            ('capture_time', 'capture', 'success', 'collision', 'changed')})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--count', type=int, default=12)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--tail-steps', type=int, default=100, choices=[100, 200])
    parser.add_argument('--reuse-labels', type=Path)
    parser.add_argument('--predictive-future', action='store_true')
    parser.add_argument('--evaluation-scene-base', type=int, default=1070000000)
    parser.add_argument('--evaluation-count', type=int, default=32)
    parser.add_argument('--strengths', type=float, nargs='+', default=[.25, 1.])
    parser.add_argument('--evaluate-checkpoints', type=Path)
    parser.add_argument('--omit-prior', action='store_true')
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--augment-labels', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Existing experiment evidence must be retained.')
    if args.evaluate_checkpoints is not None and args.reuse_labels is not None:
        parser.error('Choose frozen-checkpoint evaluation or label reuse, not both.')
    if args.augment_labels is not None and (args.evaluate_checkpoints is not None or args.reuse_labels is not None):
        parser.error('Augmentation requires its own generation run.')
    checkpoint_inputs = {}
    if args.evaluate_checkpoints is not None:
        original = json.loads((args.evaluate_checkpoints/'protocol.json').read_text())
        if (original['training_seed'] != args.seed or original['tail_steps'] != args.tail_steps or
                original['predictive_future'] != args.predictive_future):
            parser.error('Frozen checkpoint configuration does not match this evaluation.')
        checkpoint_inputs = {str(args.evaluate_checkpoints/(arm+'.json')):
            hashlib.sha256((args.evaluate_checkpoints/(arm+'.json')).read_bytes()).hexdigest()
            for arm in ('conditional', 'absolute')}
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(range(24)))
    initialize()
    paths = [Path(__file__), Path('train/role_prediction.py'), Path('train/policy/role_residual.py'),
             Path('train/policy/role_value.py'), Path('train/envs/TADgame.py'), Path('train/jit_rollout.py')]
    protocol = dict(training_bank=(1180000000 if args.predictive_future else 1140000000)+args.seed*10000,
        validation_bank=args.evaluation_scene_base,
        training_scenes_per_cell=0 if args.evaluate_checkpoints else args.count,
        validation_scenes_per_cell=args.evaluation_count, training_seed=args.seed,
        tail_steps=args.tail_steps, independent_futures_per_state=args.repetitions,
        objective='real sustained-composition return minus model return, conditional single-vessel differences',
        ridge=.01, strengths=args.strengths, formal_evidence=False,
        frozen_checkpoint_inputs=checkpoint_inputs, omit_prior=args.omit_prior,
        reused_labels=None if args.reuse_labels is None else str(args.reuse_labels),
        reused_labels_sha256=None if args.reuse_labels is None else hashlib.sha256(args.reuse_labels.read_bytes()).hexdigest(),
        augmented_labels=None if args.augment_labels is None else str(args.augment_labels),
        augmented_labels_sha256=None if args.augment_labels is None else hashlib.sha256(args.augment_labels.read_bytes()).hexdigest(),
        predictive_future=args.predictive_future,
        branch_policy=('one-block local composition intervention, then the frozen predictive policy'
                       if args.predictive_future else 'sustained composition until termination'),
        source={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), text=p.read_text()) for p in paths})
    atomic_json(args.output/'protocol.json', protocol)
    started, records = time.perf_counter(), []
    with ProcessPoolExecutor(args.workers, initializer=initialize, mp_context=mp.get_context('spawn')) as pool:
        if args.evaluate_checkpoints is not None:
            data = None
        elif args.reuse_labels is None:
            tasks = [(protocol['training_bank']+c*1000000+i, a, args.tail_steps, args.repetitions, args.predictive_future)
                     for c, a in enumerate((4., 6.)) for i in range(args.count)]
            if args.augment_labels is not None:
                old = np.load(args.augment_labels)
                if set(old['scene_seed']) != {t[0] for t in tasks}:
                    raise ValueError('Augmentation must retain exactly the original training scenarios.')
                tasks = [(*t, [{k: old[k][i] for k in old.files}
                               for i in np.where(old['scene_seed'] == t[0])[0]]) for t in tasks]
            for future in as_completed([pool.submit(collect, t) for t in tasks]):
                records.extend(future.result())
                print(dict(stage='labels', states=len(records)), flush=True)
            records.sort(key=lambda r: (r['scene_seed'], r['time']))
            data = {k: np.array([r[k] for r in records]) for k in records[0]}
            np.savez_compressed(args.output/'real-return-labels.npz', **data)
        else:
            data = np.load(args.reuse_labels)
        arms = [] if args.omit_prior else [('predictive', None, 0.)]
        for arm in ('conditional', 'absolute'):
            if args.evaluate_checkpoints is None:
                coef, loss = fit(data, arm)
                atomic_json(args.output/(arm+'.json'), dict(coefficients=coef.tolist(), training_loss=loss))
            else:
                saved = json.loads((args.evaluate_checkpoints/(arm+'.json')).read_text())
                coef, loss = np.array(saved['coefficients']), saved['training_loss']
            arms.extend((arm, coef, strength) for strength in protocol['strengths'])
            print(dict(stage='loaded' if args.evaluate_checkpoints else 'trained', arm=arm, loss=loss), flush=True)
        tasks = [(arm, coef, strength, args.tail_steps, protocol['validation_bank']+c*1000000+i, a, args.predictive_future)
                 for c, a in enumerate((4., 6.)) for i in range(args.evaluation_count) for arm, coef, strength in arms]
        rows = []
        for future in as_completed([pool.submit(evaluate, t) for t in tasks]):
            rows.append(future.result())
            if len(rows)%32 == 0:
                atomic_json(args.output/'results.json', dict(complete=False, episodes=rows, summary=summarize(rows)))
                print(dict(stage='tasks', completed=len(rows), total=len(tasks)), flush=True)
    for name, frozen in protocol['source'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != frozen['sha256']:
            raise RuntimeError('Frozen source changed: '+name)
    for name, digest in checkpoint_inputs.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Frozen checkpoint changed: '+name)
    result = dict(complete=True, episodes=rows, summary=summarize(rows),
                  wall_seconds=time.perf_counter()-started, formal_evidence=False)
    atomic_json(args.output/'results.json', result)
    print(dict(complete=True, summary=result['summary'], wall_seconds=result['wall_seconds']), flush=True)


if __name__ == '__main__':
    main()
