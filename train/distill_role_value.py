"""Learn the causal predictive conditional values of residual feedback compositions."""
import study_runtime
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import time
import warnings

import numpy as np
import torch
from torch import nn

from envs.TADgame import TADEnv
from envs.snapshot import preserved_random_state, seed_random
from feedback_joint_control import observe
from policy.role_value import ConditionalRoleValue, compositions, features
from role_prediction import RolePrediction
from train_interaction import atomic_json, atomic_torch, outcome_row


def measured_features(controller, measurement):
    packet = dict(obs=np.asarray(measurement.observations, dtype=np.float32), motion=measurement.defenders)
    base = features(packet, controller.residual.candidate_controls(packet))
    target = measurement.attacker
    radial = np.arctan2(target[1], target[0])
    extra = np.array([np.cos(target[2]-radial), np.sin(target[2]-radial), target[5],
                      controller.attacker_agility/8., controller.observer.uncertainty/8.], dtype=np.float32)
    return np.column_stack((base, np.broadcast_to(extra, (len(base), 5))))


class PredictiveValue(ConditionalRoleValue):
    def __init__(self, interaction=True, team_context=False):
        super().__init__(interaction)
        self.encoder[0] = nn.Linear(21, 32)
        self.team_context = team_context
        if team_context:
            self.pair[0] = nn.Linear(96, 32)

    def forward(self, x, masks):
        if not self.team_context:
            return super().forward(x, masks)
        h = self.encoder(x)
        pooled = h.mean(-2)
        unary = self.unary(torch.cat((h, pooled[:, None].expand_as(h)), -1)).squeeze(-1)
        value = self.baseline(pooled)+torch.einsum('bn,kn->bk', unary, masks)
        if self.interaction:
            i, j = torch.triu_indices(x.shape[1], x.shape[1], 1, device=x.device)
            pair = self.pair(torch.cat((h[:, i]+h[:, j], torch.abs(h[:, i]-h[:, j]),
                                       pooled[:, None].expand(-1, len(i), -1)), -1)).squeeze(-1)
            value = value+torch.einsum('bp,kp->bk', pair, masks[:, i]*masks[:, j])
        return value


class OrderedPredictiveValue(nn.Module):
    """Preserve public vessel identities used by the released opponent model."""
    def __init__(self, interaction=True):
        super().__init__()
        self.interaction = interaction
        self.encoder = nn.Sequential(nn.Linear(6*21, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.baseline = nn.Linear(128, 1)
        self.unary = nn.Linear(128, 6)
        self.composed = nn.Sequential(nn.Linear(128+6, 128), nn.SiLU(), nn.Linear(128, 64),
                                      nn.SiLU(), nn.Linear(64, 1))

    def forward(self, x, masks):
        if x.shape[1:] != (6, 21):
            raise ValueError('The ordered pilot is specialized to six publicly identified vessels.')
        h = self.encoder(x.flatten(1))
        value = self.baseline(h)+self.unary(h)@masks.T
        if self.interaction:
            batch, choices = len(x), len(masks)
            state = h[:, None].expand(-1, choices, -1)
            options = masks[None].expand(batch, -1, -1)
            value = value+self.composed(torch.cat((state, options), -1)).squeeze(-1)
        return value


def value_model(arm, team_context=False, ordered=False):
    return (OrderedPredictiveValue(interaction=arm != 'additive') if ordered else
            PredictiveValue(interaction=arm != 'additive', team_context=team_context))


class TreeValue(nn.Module):
    """Small-data conditional-value comparator; same public inputs and labels."""
    def __init__(self, estimator, additive=False):
        super().__init__()
        self.estimator, self.additive = estimator, additive

    def forward(self, x, masks):
        value = torch.as_tensor(self.estimator.predict(x.flatten(1).numpy()), dtype=x.dtype)
        return value @ masks.T if self.additive else value


def fit_tree(records, arm, seed):
    from sklearn.ensemble import ExtraTreesRegressor
    x = np.stack([r['features'] for r in records]).reshape(len(records), -1)
    costs = np.stack([r['costs'] for r in records])
    costs = np.minimum(costs, 60.-np.array([r['time'] for r in records])[:, None])
    y = -costs/20.
    target = y if arm == 'absolute' else y-y[:, :1]
    if arm == 'additive':
        target = target @ np.linalg.pinv(compositions(6)).T
    estimator = ExtraTreesRegressor(n_estimators=128, min_samples_leaf=4,
                                   max_features=.8, random_state=seed, n_jobs=1)
    estimator.fit(x, target)
    return TreeValue(estimator, arm == 'additive'), float(np.mean((estimator.predict(x)-target)**2))


class RecordingTeacher(RolePrediction):
    def __init__(self, defenders):
        super().__init__(defenders, tail_policy='candidate')
        self.pending, self.records = [], []

    def predict(self, measurement, template):
        if template == 'baseline':
            self.pending = []
            self.current_features = measured_features(self, measurement)
        prediction = super().predict(measurement, template)
        c, b, cost = prediction['score']
        self.pending.append(float(360.*c+180.*b+cost))
        return prediction

    def control(self, measurement):
        thrust, info = super().control(measurement)
        if info['planned']:
            self.records.append(dict(features=self.current_features.copy(),
                costs=np.array(self.pending, dtype=np.float32), time=measurement.time))
        return thrust, info


class DistilledController(RolePrediction):
    def __init__(self, defenders, model, minimum_gain=0.):
        super().__init__(defenders, tail_policy='candidate')
        self.model = model.eval().requires_grad_(False)
        self.minimum_gain = minimum_gain
        self.interventions = 0

    @torch.no_grad()
    def control(self, measurement):
        self.attacker_agility = self.observer.update(measurement)
        planned = self.remaining == 0
        if planned:
            x = torch.from_numpy(measured_features(self, measurement))[None]
            masks = torch.from_numpy(compositions(self.defenders))
            value = self.model(x, masks)[0].numpy()
            choice = int(np.argmax(value))
            if (value[choice]-value[0])*20. < self.minimum_gain:
                choice = 0
            self.template = list(self.options)[choice]
            self.remaining = self.block_steps
            self.plan_count += 1
            self.interventions += int(choice != 0)
        action = self.nominal(measurement, self.template)
        _, thrust, info = self.safety.control(measurement.defenders, action, measurement.boids)
        self.remaining -= 1
        self.tick += 1
        info.update(planned=planned, template=self.template)
        return thrust, info


def initialize():
    torch.set_num_threads(1)
    warnings.filterwarnings('ignore', message='.*Lgh is zero.*')


def rollout(seed, agility, factory):
    with preserved_random_state():
        seed_random(seed)
        env = TADEnv(6, protocol='paper-parameters-v1')
        obs, _ = env.reset(agility, noisy_agility=False)
        controller = factory()
        np.random.seed(seed+1000000)
        done = 0
        while not done:
            thrust, _ = controller.control(observe(env, obs))
            obs, _, done, _ = env.step(np.zeros((6, 3)), 'AdaRes', defender_thrust=thrust)
        return outcome_row(env, done), controller


def collect(task):
    seed, agility = task
    row, controller = rollout(seed, agility, lambda: RecordingTeacher(6))
    return dict(row, scene_seed=seed, agility=agility), [dict(r, scene_seed=seed, agility=agility)
                                                     for r in controller.records]


def fit(records, arm, seed, updates, bounded_targets=False, team_context=False, ordered=False,
        preference_weight=0.):
    seed_random(seed)
    model = value_model(arm, team_context, ordered)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    x = torch.tensor(np.stack([r['features'] for r in records]))
    costs = np.stack([r['costs'] for r in records])
    if bounded_targets:
        costs = np.minimum(costs, 60.-np.array([r['time'] for r in records])[:, None])
    target = -torch.tensor(costs, dtype=torch.float32)/20.
    masks = torch.from_numpy(compositions(6))
    diff = masks[:, None]-masks[None]
    hi, lo = torch.where((diff.abs().sum(-1) == 1) & (diff.sum(-1) == 1))
    rng = np.random.default_rng(seed+1703)
    for _ in range(updates):
        indices = rng.choice(len(x), min(32, len(x)), replace=False)
        value, y = model(x[indices], masks), target[indices]
        if arm == 'absolute':
            loss = nn.functional.mse_loss(value, y)
        else:
            loss = nn.functional.mse_loss(value[:, hi]-value[:, lo], y[:, hi]-y[:, lo])
            loss = loss+.05*nn.functional.mse_loss(value[:, 0], y[:, 0])
        if preference_weight:
            preference = torch.softmax(y*10., dim=-1)
            loss = loss-preference_weight*(preference*torch.log_softmax(value*10., dim=-1)).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
    return model, float(loss.detach())


def evaluate(task):
    path, threshold, seed, agility = task
    if path.suffix == '.joblib':
        import joblib
        saved = joblib.load(path)
        model = TreeValue(saved['estimator'], saved['arm'] == 'additive')
    else:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        model = value_model(saved['arm'], saved.get('team_context', False), saved.get('ordered', False))
        model.load_state_dict(saved['model'])
    row, controller = rollout(seed, agility, lambda: DistilledController(6, model, threshold))
    return dict(row, arm=saved['arm'], training_seed=saved['seed'], minimum_gain=threshold,
        scene_seed=seed, agility=agility, decisions=controller.plan_count, interventions=controller.interventions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--count', type=int, default=16)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--updates', type=int, default=4000)
    parser.add_argument('--reuse-labels', type=Path)
    parser.add_argument('--bounded-targets', action='store_true')
    parser.add_argument('--team-context', action='store_true')
    parser.add_argument('--ordered', action='store_true')
    parser.add_argument('--preference-weight', type=float, default=0.)
    parser.add_argument('--tree', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Existing experiment evidence must be retained.')
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, set(range(24)))
    initialize()
    files = [Path(__file__), Path('train/role_prediction.py'), Path('train/policy/role_value.py'),
             Path('train/policy/role_residual.py'), Path('train/envs/TADgame.py')]
    protocol = dict(seed=args.seed, training_scenes_per_cell=args.count, validation_scenes_per_cell=32,
        training_bank=1120000000+args.seed*10000, validation_bank=1070000000,
        updates=args.updates, arms=['full', 'absolute', 'additive'], thresholds_seconds=[0., 1., 3.],
        objective='conditional differences of sustained causal predictive feedback values',
        teacher_public_state_only=True, observer_bounds=[1., 8.], formal_evidence=False,
        bounded_targets=args.bounded_targets, team_context=args.team_context, ordered=args.ordered,
        preference_weight=args.preference_weight,
        model_kind='extra-trees-128-leaf4' if args.tree else 'neural',
        reused_labels=None if args.reuse_labels is None else str(args.reuse_labels),
        reused_label_sha256=None if args.reuse_labels is None else hashlib.sha256(args.reuse_labels.read_bytes()).hexdigest(),
        source={str(p):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), text=p.read_text()) for p in files})
    atomic_json(args.output/'protocol.json', protocol)
    started, records, teacher_rows = time.perf_counter(), [], []
    with ProcessPoolExecutor(args.workers, initializer=initialize, mp_context=mp.get_context('spawn')) as pool:
        tasks = [] if args.reuse_labels is not None else [(protocol['training_bank']+c*1000000+i, a)
                 for c, a in enumerate((4., 6.)) for i in range(args.count)]
        for future in as_completed([pool.submit(collect, task) for task in tasks]):
            row, states = future.result()
            teacher_rows.append(row)
            records.extend(states)
            print(dict(stage='teacher', episodes=len(teacher_rows), states=len(records)), flush=True)
        records.sort(key=lambda r: (r['scene_seed'], r['time']))
        if args.reuse_labels is not None:
            data = np.load(args.reuse_labels)
            records = [dict(features=data['features'][i], costs=data['costs'][i],
                scene_seed=int(data['scene_seed'][i]), time=float(data['time'][i]), agility=float(data['agility'][i]))
                for i in range(len(data['features']))]
        else:
            np.savez_compressed(args.output/'predictive-labels.npz',
                features=np.stack([r['features'] for r in records]), costs=np.stack([r['costs'] for r in records]),
                scene_seed=[r['scene_seed'] for r in records], time=[r['time'] for r in records],
                agility=[r['agility'] for r in records], masks=compositions(6))
            atomic_json(args.output/'teacher-episodes.json', teacher_rows)
        paths = []
        for arm in protocol['arms']:
            model, loss = (fit_tree(records, arm, args.seed) if args.tree else
                          fit(records, arm, args.seed, args.updates, args.bounded_targets,
                              args.team_context, args.ordered, args.preference_weight))
            path = args.output/(arm+('.joblib' if args.tree else '.pth'))
            saved = dict(format='predictive-role-value-v1', arm=arm, seed=args.seed,
                                   model=model.state_dict(), training_loss=loss,
                                   team_context=args.team_context, bounded_targets=args.bounded_targets,
                                   ordered=args.ordered, preference_weight=args.preference_weight)
            if args.tree:
                import joblib
                saved['estimator'] = model.estimator
                joblib.dump(saved, path)
            else:
                atomic_torch(path, saved)
            paths.append(path)
            print(dict(stage='trained', arm=arm, loss=loss), flush=True)
        tasks = [(p, threshold, protocol['validation_bank']+c*1000000+i, a)
                 for c, a in enumerate((4., 6.)) for i in range(32)
                 for p in paths for threshold in protocol['thresholds_seconds']]
        rows = []
        from train_role_residual import summary
        for future in as_completed([pool.submit(evaluate, task) for task in tasks]):
            rows.append(future.result())
            if len(rows)%64 == 0:
                atomic_json(args.output/'results.json', dict(complete=False, episodes=rows, summary=summary(rows)))
                print(dict(stage='tasks', completed=len(rows), total=len(tasks)), flush=True)
    for name, saved in protocol['source'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != saved['sha256']:
            raise RuntimeError('Frozen source changed: '+name)
    result = dict(complete=True, states=len(records), episodes=rows, summary=summary(rows),
                  wall_seconds=time.perf_counter()-started, formal_evidence=False)
    atomic_json(args.output/'results.json', result)
    print(dict(complete=True, summary=result['summary'], wall_seconds=result['wall_seconds']), flush=True)


if __name__ == '__main__':
    main()
