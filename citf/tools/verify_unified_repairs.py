"""Focused deterministic acceptance of the two adopted review recommendations.

No research training, parameter search, scenario benchmark or result selection.
Run: python -X utf8 citf/tools/verify_unified_repairs.py
"""
import ast
import copy
from collections import deque
from dataclasses import asdict, replace
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import numpy as np
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'citf'/'code'))
from unified import IMPLEMENTATION_VERSION
from unified.baselines import PublicBaseline, BaselineLearner, author_networks
from unified.environment import FleetEnv, TaskConfig, execute, boids, fusion
from unified.networks import CITF, ModelConfig
from unified.learner import Learner, Replay
from unified.train import (arguments, policy_schedule, rollout_count, write_json, prepare_output,
                           append_evaluation, resume_arguments)
from unified.protocol import (baseline_updates, build_registry, load_registry, registered_arguments,
    run_identity, bind_run, canonical_arguments, model_definition, learning_counts, checkpoint_identity,
    validate_checkpoint, evaluation_identity, attach_identity, validate_result, outcome_metrics,
    phase_identity, phase_counts, ATTACKER_RECIPE, collection_updates, digest, policy_digest,
    validate_learning_curve, validate_phase_checkpoint, chain_evaluation_identity, vrx_rows, validate_optimizer_counts)


def recipe():
    with patch.object(sys, 'argv', ['train', '--out', 'unused', '--feedback', '.01', '--batch', '512']):
        args = arguments()
    return dict(arguments=vars(args), task=asdict(TaskConfig()), implementation_version=IMPLEMENTATION_VERSION)


def rows(count=100, seed=74001):
    return [dict(scene_seed=seed, scene_index=i, outcome=i % 4+1, seconds=60. if i % 4 == 3 else 10.+i % 4,
                 success=int(i % 4 >= 2), breach=int(i % 4 == 0), collision=int(i % 4 == 1),
                 capture=int(i % 4 == 2), timeout=int(i % 4 == 3)) for i in range(count)]


class Repairs(unittest.TestCase):
    def assertClose(self, a, b, atol=2e-6, rtol=2e-5):
        torch.testing.assert_close(torch.as_tensor(a), torch.as_tensor(b), atol=atol, rtol=rtol)

    def test_author_learner_fixed_batch(self):
        source = ROOT/'third_party/arboids_release/train/policy/SAC.py'
        networks = author_networks()
        spec = importlib.util.spec_from_file_location('review_author_sac', source)
        original = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'policy': ModuleType('policy'), 'policy.networks': networks}):
            spec.loader.exec_module(original)
        x = FleetEnv(4, device='cpu', seed=173).observation()
        for method in ('arboids', 'rp', 'sac'):
            with self.subTest(method=method):
                torch.manual_seed(71)
                model = PublicBaseline(method, 512)
                port = BaselineLearner(model)
                cfg = SimpleNamespace(rl=SimpleNamespace(batch_size=12, GAMMA=.99, TAU=.005,
                                                         learning_rate=1e-4, hidden_dim=512))
                ref = original.SAC(cfg, 9, 0 if method == 'sac' else 8, model.action_dim,
                                   adaptive=method == 'arboids')
                ref.actor.load_state_dict(model.actor.state_dict())
                ref.critic.load_state_dict(model.critic.state_dict())
                ref.critic_target.load_state_dict(port.target.state_dict())
                state = model.observe(x)[0].flatten(0, 1)
                latent = torch.rand(12, model.action_dim)*1.2-.6
                if method == 'arboids':
                    latent[:, 2] = .4
                reward = torch.linspace(-2., 2., 12)
                done = torch.tensor([0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0], dtype=torch.bool)
                next_state = state+.02
                buffer = SimpleNamespace(sample=lambda *_: (state, latent, reward[:, None], next_state, done.float()[:, None]))
                original_mse = F.mse_loss
                for iteration in range(2):
                    captured = []

                    def mse(prediction, target, *args, **kwargs):
                        loss = original_mse(prediction, target, *args, **kwargs)
                        captured.append((target.detach().clone(), loss.detach().clone()))
                        return loss

                    ref_actor, ref_q = [], []
                    ha = ref.actor.register_forward_hook(lambda module, args, output: ref_actor.append(tuple(v.detach() for v in output)))
                    hq = ref.critic.register_forward_hook(lambda module, args, output: ref_q.append(tuple(v.detach() for v in output)))
                    alpha_before = ref.alpha.detach().clone()
                    torch.manual_seed(800+iteration)
                    with patch.object(F, 'mse_loss', side_effect=mse):
                        ref.learn(buffer)
                    ha.remove(); hq.remove()
                    expected = captured[:]
                    captured.clear()
                    torch.manual_seed(800+iteration)
                    with patch.object(F, 'mse_loss', side_effect=mse):
                        report = port.update(state, latent, reward, next_state, done)
                    for (target, loss), (target_ref, loss_ref) in zip(captured, expected):
                        self.assertClose(target, target_ref)
                        self.assertClose(loss, loss_ref)
                    self.assertClose(report['q_loss'], sum(v[1] for v in expected))
                    self.assertClose(report['actor_loss'], (alpha_before*ref_actor[-1][1]-torch.minimum(*ref_q[-1])).mean())
                    for actual, reference in ((model.actor, ref.actor), (model.critic, ref.critic), (port.target, ref.critic_target)):
                        for name, parameter in actual.named_parameters():
                            counterpart = dict(reference.named_parameters())[name]
                            self.assertClose(parameter, counterpart)
                            if parameter.grad is not None:
                                self.assertClose(parameter.grad, counterpart.grad)
                    self.assertClose(port.log_alpha.reshape(1), ref.log_alpha)
                    self.assertClose(port.log_alpha.grad.reshape(1), ref.log_alpha.grad)
                    if method == 'arboids':
                        self.assertGreater(float(model.actor.adap_layer.weight.grad.norm()), 0.)
                        self.assertEqual(port.target_entropy, -3.)
                    else:
                        self.assertEqual(port.target_entropy, -2.)
                    validate_optimizer_counts(port.payload(), ('actor', 'critic', 'temperature'),
                                              dict.fromkeys(('actor', 'critic', 'temperature'), iteration+1))
                    # Optimizer/RNG continuation after a real serialization round trip.
                    stream = io.BytesIO()
                    torch.save(port.payload(), stream)
                    stream.seek(0)
                    restored = BaselineLearner(PublicBaseline(method, 512))
                    restored.restore(torch.load(stream, weights_only=False))
                    port, model = restored, restored.model

    def test_baseline_transition_ledger(self):
        self.assertEqual(baseline_updates(49999, 50000), 0)
        self.assertEqual(baseline_updates(50000, 50000), 1)
        self.assertEqual(baseline_updates(1000000, 50000), 950001)
        for envs, interval, total, warmup in ((64, 5000, 1000000, 50000), (17, 41, 253, 50), (64, 13, 155, 50)):
            args = SimpleNamespace(envs=envs, eval_interval=interval, steps=total,
                                   start_learning=warmup, updates_per_joint_transition=1.)
            transitions = updates = 0
            while transitions < total:
                boundary = (transitions//interval+1)*interval
                transitions += rollout_count(args, transitions, boundary)
                due = baseline_updates(transitions, warmup)-updates
                self.assertGreaterEqual(due, 0)
                updates += due
                # Saving/restoring the two counters consumes no extra entitlement.
                transitions, updates = json.loads(json.dumps([transitions, updates]))
                self.assertEqual(baseline_updates(transitions, warmup)-updates, 0)
            self.assertEqual(updates, max(0, total-warmup+1))
        registry = build_registry(recipe())
        for method in ('arboids', 'rp', 'sac'):
            args = registered_arguments(registry, method, 2101)
            self.assertNotIn('updates_per_step', args)
            self.assertEqual(learning_counts(method, 1000000, args, 950001),
                             dict(critic=950001, actor=950001, temperature=950001, target=950001))
            with self.assertRaises(ValueError):
                learning_counts(method, 1000000, args, 59436)

    def test_local_capacity_and_action_independence(self):
        cfg = ModelConfig(samples=4)
        torch.manual_seed(3)
        full = CITF(cfg)
        torch.manual_seed(3)
        local = CITF(replace(cfg, action_interaction=False))
        self.assertEqual(sum(p.numel() for p in full.q1.parameters()), 857044)
        self.assertEqual(sum(p.numel() for p in local.q1.parameters()), 856810)
        for key, value in local.state_dict().items():
            if key in full.state_dict():
                shared = full.state_dict()[key]
                self.assertClose(value[tuple(slice(size) for size in shared.shape)], shared, atol=0, rtol=0)
        self.assertFalse(any('local_control' in key for key in local.state_dict()))
        optimizer = torch.optim.Adam(local.q1.parameters(), lr=1e-3)
        active = {}
        x = FleetEnv(5, device='cpu', seed=139).observation()
        u = torch.rand(5, 3, 2)*500-250
        self.assertClose(local.q1(x, u, individual=True), full.q1(x, u, individual=True))
        for layer in local.q1.layers:
            self.assertEqual(layer.ff[0].out_features, 1072)
            self.assertEqual(float(layer.ff[2].weight[:, 512:].detach().abs().sum()), 0.)
        for iteration in range(3):
            u = torch.rand(5, 3, 2)*500-250
            loss = (local.q1(x, u, individual=True)-torch.randn(5, 3)).square().mean()
            optimizer.zero_grad()
            loss.backward()
            for key, parameter in local.q1.named_parameters():
                if parameter.grad is not None:
                    active[key] = active.get(key, torch.zeros_like(parameter, dtype=torch.bool)) | (parameter.grad != 0)
            optimizer.step()
        for index in range(2):
            self.assertTrue(bool(active[f'layers.{index}.ff.0.weight'][512:].any(-1).all()))
            self.assertTrue(bool(active[f'layers.{index}.ff.0.bias'][512:].all()))
            self.assertTrue(bool(active[f'layers.{index}.ff.2.weight'][:, 512:].any(0).all()))
        for n in (3, 7):
            x = FleetEnv(3, TaskConfig(defenders=n), seed=171, device='cpu').observation()
            u = (torch.rand(3, n, 2)*500-250).requires_grad_()
            for critic in (local.q1, local.q2):
                first = critic(x, u, individual=True)[:, 0]
                derivative = torch.autograd.grad(first.sum(), u)[0]
                self.assertEqual(float(derivative[:, 1:].abs().max()), 0.)
                changed = u.detach().clone(); changed[:, 1:] += 73.
                self.assertClose(first, critic(x, changed, individual=True)[:, 0], atol=0, rtol=0)
            expected = torch.minimum(local.q1(x, u, individual=True), local.q2(x, u, individual=True)).mean(-1)
            self.assertClose(local.value_from_encoded(local.encode_value(x), u), expected)

    def test_feedback_only_changes_explicit_generator_term(self):
        cfg = ModelConfig(width=32, layers=1, samples=4, feedback=.01)
        torch.manual_seed(51)
        full = CITF(cfg)
        no_feedback = copy.deepcopy(full); no_feedback.cfg = replace(cfg, feedback=0.)
        x = FleetEnv(4, device='cpu', seed=91).observation()
        u = torch.rand(4, 3, 2)*400-200
        batch = dict(x=x, u=u, ri=torch.randn(4, 3), next_x=x, done=torch.ones(4, dtype=torch.bool))
        runs = []
        for model in (full, no_feedback):
            measured = Learner(copy.deepcopy(model))
            validate_optimizer_counts(measured.payload(), ('critic', 'endpoint', 'generator'),
                                      dict(critic=0, endpoint=0, generator=0))
            measured.update(batch, temperature=.7)
            measured.update(batch, temperature=.7)
            validate_optimizer_counts(measured.payload(), ('critic', 'endpoint', 'generator'),
                                      dict(critic=2, endpoint=1, generator=1))
            learner = Learner(model); learner.updates = 1
            calls = []
            original = model.generator.log_prob

            def log_prob(*args):
                result = original(*args)
                result.retain_grad()
                calls.append((args, result))
                return result

            torch.manual_seed(512)
            # Retain the common pre-step parameter snapshot to decompose gradients.
            with patch.object(model.generator, 'log_prob', side_effect=log_prob), \
                    patch.object(learner.generator_optimizer, 'step'), patch.object(learner.endpoint_optimizer, 'step'), \
                    patch('torch.nn.utils.clip_grad_norm_', return_value=torch.tensor(0.)):
                report = learner.update(batch, temperature=.7)
            gradient = torch.cat([p.grad.flatten() for p in model.generator.parameters()])
            self.assertGreater(float(gradient.norm()), 0.)
            self.assertTrue(all(p.grad is not None for p in model.endpoint.parameters()))
            parameters = list(model.generator.parameters())
            decomposition = sum((original(*args)*value.grad.detach()).sum() for args, value in calls)
            rebuilt = torch.cat([g.flatten() for g in torch.autograd.grad(decomposition, parameters)])
            self.assertClose(gradient, rebuilt)
            runs.append((report, calls, gradient, [p.grad.clone() for p in model.endpoint.parameters()]))
        self.assertEqual(len(runs[0][1]), 2)
        self.assertEqual(len(runs[1][1]), 1)
        self.assertClose(runs[0][1][0][1], runs[1][1][0][1])
        self.assertClose(runs[0][1][0][1].grad, runs[1][1][0][1].grad)
        self.assertClose(runs[0][0]['base_loss'], runs[1][0]['base_loss'])
        self.assertEqual(float(runs[1][0]['feedback_loss']), 0.)
        self.assertGreater(float((runs[0][2]-runs[1][2]).norm()), 0.)
        for first, second in zip(runs[0][3], runs[1][3]):
            self.assertClose(first, second)

    def test_replay_mixture_rounding_and_ring(self):
        replay = Replay(9, 3, 'cpu', shapes=dict(r=(), event=(), done=()))
        replay.add(r=torch.arange(12.), event=torch.tensor([0, 1, 2, 0, 1, 1, 2, 3, 0, 4, 1, 0]),
                   done=torch.tensor([0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0], dtype=torch.bool))
        self.assertEqual(replay.cursor, 3)
        self.assertEqual(set(replay.data['r'].tolist()), set(range(3, 12)))
        for count in (1, 7, 11):
            batch = replay.sample_value(count, torch.Generator().manual_seed(93))
            if count == 1:
                self.assertNotIn('value_weight', batch)
                continue
            pools = {kind: ((replay.data['event'] == kind) & replay.data['done']).nonzero().flatten()
                     for kind in range(1, 5)}
            pools = {key: value for key, value in pools.items() if len(value)}
            terminal = int(count*.5); uniform = (count-terminal)/count/replay.size
            for row in range(count):
                probability = uniform
                for index, (kind, pool) in enumerate(pools.items()):
                    if batch['event'][row] == kind and batch['done'][row]:
                        probability += (terminal//len(pools)+(index < terminal % len(pools)))/count/len(pool)
                self.assertClose(batch['value_probability'][row], probability)
                self.assertClose(batch['value_weight'][row], (uniform/probability)**.5)
        saved = replay.state_dict()
        restored = Replay(9, 3, 'cpu', shapes=dict(r=(), event=(), done=()))
        restored.load_state_dict(saved)
        for key, value in replay.sample_value(11, torch.Generator().manual_seed(10)).items():
            self.assertClose(value, restored.sample_value(11, torch.Generator().manual_seed(10))[key])
        restored.add(r=torch.arange(9.), event=torch.zeros(9, dtype=torch.long), done=torch.zeros(9, dtype=torch.bool))
        self.assertNotIn('value_weight', restored.sample_value(11))

    def test_executable_endpoints_and_clipped_team_score(self):
        model = CITF(ModelConfig(width=32, layers=1, samples=4))
        x = FleetEnv(3, seed=31, device='cpu').observation()
        x[:, :-1, 6:8] = torch.tensor([[-470., 970.], [250., -100.], [500., 600.]])
        source = boids(x); b = execute(source, x[:, :-1, 6:8], 300.)
        a = model.endpoint(x, source)
        g = torch.rand(3, 4, 3)
        convex = b[:, None]+g[..., None]*(a[:, None]-b[:, None])
        self.assertClose(fusion(x, a, b, g, 300.), convex)
        result = model.select(x)
        self.assertClose(result['action'], fusion(x, result['a'], result['b'], result['g'], 300.))
        q1, q2 = model.q1(x, result['action'], True), model.q2(x, result['action'], True)
        self.assertClose(model.value_from_encoded(model.encode_value(x), result['action']), torch.minimum(q1, q2).mean(-1))
        self.assertTrue(bool(((result['action']-x[:, :-1, 6:8]).abs() <= 300.001).all()))

    def test_registry_checkpoint_cache_and_metrics(self):
        registry = build_registry(recipe())
        args = registered_arguments(registry, 'full', 2101)
        run = run_identity(args, registry['task'], registry)
        updates = collection_updates(1000000, args)
        payload = dict(run_identity=run, arguments=args, task=run['task'], config=run['model'],
                       model={'fixture': torch.tensor([1.])}, updates=updates, transitions=1000000,
                       update_counts=learning_counts('full', 1000000, args, updates),
                       learner_recipe=dict(gamma=.99, tau=args['target_tau'], policy_delay=2),
                       implementation_version=IMPLEMENTATION_VERSION)
        validation_rows = rows(seed=73001)
        payload['evaluation'] = dict(rows=validation_rows, task=run['task'], scene_seed=73001, policy_seed=83001,
            **outcome_metrics(validation_rows, 100, 73001, 60.), role='validation', research_stage='formal',
            run_id=run['id'], policy_id=policy_digest(payload['model']), transitions=1000000, updates=updates,
            update_counts=payload['update_counts'])
        payload['optimizers'] = [dict(state={0: dict(step=payload['update_counts'][key])},
                                     param_groups=[dict(params=[0])]) for key in ('critic', 'endpoint', 'generator')]
        self.assertEqual(registry['critic_capacity']['no_action'], 856810)
        self.assertLess(registry['critic_capacity']['relative_difference'], .0003)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); path = root/'last.pt'
            write_json(root/'registry.json', registry)
            self.assertEqual(load_registry(root/'registry.json'), registry)
            namespace = SimpleNamespace(**args, registry=str(root/'registry.json'), out=temp,
                                        resume=None, init_from=None, device='cpu')
            self.assertEqual(bind_run(namespace, TaskConfig()), run)
            torch.save(payload, path)
            validate_checkpoint(payload, registry, 'full', 2101, 1000000)
            for kind in ('missing', 'count'):
                changed = copy.deepcopy(payload)
                if kind == 'missing':
                    changed['optimizers'].pop()
                else:
                    changed['optimizers'][1]['state'][0]['step'] -= 1
                with self.assertRaises(ValueError):
                    validate_checkpoint(changed, registry, 'full', 2101, 1000000)
            checkpoint = checkpoint_identity(path, payload)
            identity = evaluation_identity(registry, 'full', 2101, checkpoint, run['task'], 74001, 84001, 100)
            data = rows()
            result = dict(rows=data, task=run['task'], method='full', scene_seed=74001, policy_seed=84001,
                          **outcome_metrics(data, 100, 74001, 60.))
            attach_identity(result, identity)
            validate_result(result, identity)
            corruptions = [lambda r: r['rows'].__setitem__(1, r['rows'][0]),
                           lambda r: r['rows'][0].__setitem__('success', 1),
                           lambda r: r['rows'][0].__setitem__('seconds', float('nan')),
                           lambda r: r['rows'][3].__setitem__('seconds', 13.),
                           lambda r: r.__setitem__('capture_count', 0),
                           lambda r: r.__setitem__('method', 'no_feedback'),
                           lambda r: r['identity']['checkpoint'].__setitem__('sha256', 'different')]
            for mutate in corruptions:
                changed = copy.deepcopy(result); mutate(changed)
                with self.assertRaises(ValueError):
                    validate_result(changed, identity)
            payload['model']['fixture'].add_(1)
            with self.assertRaises(ValueError):
                validate_checkpoint(payload, registry, 'full', 2101, 1000000)
            payload['evaluation']['policy_id'] = policy_digest(payload['model'])
            torch.save(payload, path)
            replaced = evaluation_identity(registry, 'full', 2101, checkpoint_identity(path, payload), run['task'], 74001, 84001, 100)
            with self.assertRaises(ValueError):
                validate_result(result, replaced)
            for method, seed in (('no_action', 2101), ('full', 2102)):
                with self.assertRaises(ValueError):
                    validate_checkpoint(payload, registry, method, seed, 1000000)
            changed = copy.deepcopy(registry); changed['source']['id'] = 'edited-source'
            changed['id'] = digest({k: v for k, v in changed.items() if k != 'id'})
            write_json(root/'bad_registry.json', changed)
            with self.assertRaises(ValueError):
                load_registry(root/'bad_registry.json')
            for task in (dict(run['task'], defenders=7), dict(run['task'], agility=2.75)):
                with self.assertRaises(ValueError):
                    evaluation_identity(registry, 'no_action', 2101, checkpoint, task, 74005, 84001, 100)
        uncaptured = [dict(r, outcome=1, success=0, breach=1, capture=0, collision=0, timeout=0) for r in rows()]
        self.assertIsNone(outcome_metrics(uncaptured, 100, 74001, 60.)['mean_capture_seconds'])

    def test_phase_inheritance_and_cumulative_clock(self):
        registry = build_registry(recipe())
        root_args = registered_arguments(registry, 'full', 2101)
        root = run_identity(root_args, registry['task'], registry)
        counts = dict(defender=500000, attacker=0)
        updates = dict(defender=collection_updates(500000, root_args), attacker=0)
        previous = dict(sha256='fixture', transitions=500000, run_id=root['id'])
        for phase in range(2, 6):
            active = 'defender' if phase % 2 else 'attacker'
            inherited = root_args if active == 'defender' else ATTACKER_RECIPE
            args = dict(phase=phase, steps=500000, seed=2101, eval_count=100, eval_interval=50000,
                        eval_seed=73003, test_seed=74003, policy_eval_seed=83003, stage='formal',
                        **{k: inherited[k] for k in ATTACKER_RECIPE})
            identity = phase_identity(args, registry['task'], root, previous, ATTACKER_RECIPE, counts, updates, registry)
            if active == 'defender':
                self.assertEqual(args['batch'], 512)
                changed = dict(args, batch=256)
                with self.assertRaises(ValueError):
                    phase_identity(changed, registry['task'], root, previous, ATTACKER_RECIPE, counts, updates, registry)
            updates[active] += collection_updates(500000, args)
            counts, recorded = phase_counts(identity, 500000, updates['defender'], updates['attacker'])
            self.assertEqual(counts[active], 500000*((phase+1)//2 if active == 'defender' else phase//2))
            previous = dict(sha256=f'phase{phase}', phase=phase, transitions=500000, root_run_id=root['id'])
        custom = dict(root_args, feedback_start=600000, feedback_ramp=400000, exploration_steps=1200000)
        schedule = policy_schedule(custom, 750000)
        self.assertEqual(schedule['feedback_scale'], .375)
        self.assertNotEqual(schedule['temperature'], custom['temperature_end'])
        self.assertEqual(schedule['probability'], .4375)

    def test_checkpoint_curve_recovery(self):
        registry = build_registry(recipe())
        args = registered_arguments(registry, 'full', 2101)
        run = run_identity(args, registry['task'], registry)
        evaluations = []
        for transitions in (5000, 10000):
            data = rows(seed=73001)
            updates = collection_updates(transitions, args)
            evaluations.append(dict(rows=data, task=run['task'], scene_seed=73001, policy_seed=83001,
                **outcome_metrics(data, 100, 73001, 60.), role='validation', research_stage='formal',
                run_id=run['id'], policy_id=policy_digest({'fixture': torch.tensor([float(transitions)])}),
                transitions=transitions, updates=updates, update_counts=learning_counts('full', transitions, args, updates)))
        saved = dict(run_identity=run, arguments=args, transitions=10000, evaluation=evaluations[-1])
        resume = dict(saved, implementation_version=IMPLEMENTATION_VERSION, training_state={})
        with patch.object(sys, 'argv', ['train', '--out', 'unused', '--resume', 'last.pt']):
            restored = arguments()
            resume_arguments(restored, resume)
            self.assertEqual(restored.steps, 1000000)
        with patch.object(sys, 'argv', ['train', '--out', 'unused', '--resume', 'last.pt', '--steps', '2000000']):
            with self.assertRaises(ValueError):
                resume_arguments(arguments(), resume)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            append_evaluation(out, evaluations[0])
            prepare_output(out, saved)
            prepare_output(out, saved)
            self.assertEqual(validate_learning_curve(out/'evaluations.jsonl', run, until=10000), evaluations)
            descriptor = dict(run_id=run['id'], transitions=10000, policy_id=evaluations[-1]['policy_id'],
                              validation_id=digest(evaluations[-1]))
            validate_learning_curve(out/'evaluations.jsonl', run, until=10000, endpoint=descriptor)
            for field in ('policy_id', 'validation_id'):
                changed = dict(descriptor, **{field: '0'*64})
                with self.assertRaises(ValueError):
                    validate_learning_curve(out/'evaluations.jsonl', run, until=10000, endpoint=changed)
            append_evaluation(out, evaluations[-1])
            with self.assertRaises(ValueError):
                prepare_output(out, saved)
            (out/'evaluations.jsonl').write_text(json.dumps(evaluations[-1])+'\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                prepare_output(out, saved)

    def test_system_artifact_identity_and_entry_points(self):
        registry = build_registry(recipe())
        args = registered_arguments(registry, 'full', 2101)
        run = run_identity(args, registry['task'], registry)
        descriptor = dict(run_id=run['id'], transitions=1000000, sha256='fixed-checkpoint')
        task = dict(registry['task'], agility=2.25)
        identity = evaluation_identity(registry, 'full', 2101, descriptor, task, 74002, 84002, 100, 'vrx')
        adapter = {'id': 'frozen-adapter'}
        episodes = [dict(passed=True, identity=identity, adapter_source=adapter, scene_index=i,
                         policy_seed=84002+i, scene_seed=74002, task=task, outcome_code=i % 4+1,
                         simulation_seconds=60. if i % 4 == 3 else 12., success=i % 4 >= 2) for i in range(100)]
        data = vrx_rows(episodes, identity, adapter)
        self.assertEqual(outcome_metrics(data, 100, 74002, 60.)['success'], 50)
        for field, value in (('policy_seed', 84002), ('adapter_source', {'id': 'changed'}), ('passed', False)):
            changed = copy.deepcopy(episodes)
            changed[1][field] = value
            with self.assertRaises(ValueError):
                vrx_rows(changed, identity, adapter)
        swapped = copy.deepcopy(episodes)
        swapped[0], swapped[37] = swapped[37], swapped[0]
        for changed in (swapped, episodes[:-1]):
            with self.assertRaises(ValueError):
                vrx_rows(changed, identity, adapter)
        with self.assertRaises(ValueError):
            evaluation_identity(registry, 'full', 2102, descriptor, task, 74002, 84002, 100, 'vrx')
        for phase, role in ((2, 'retention'), (4, 'retention')):
            with self.assertRaises(ValueError):
                chain_evaluation_identity(registry, phase,
                    dict(root_run_id=run['id'], transitions=500000, phase=phase), registry['task'], role)
        # VRX imports ROS at runtime; parse the changed entry points without starting Gazebo.
        for path in [ROOT/'vrx'/name for name in ('run_batch.py', 'run_experiment.py', 'unified_vrx.py')]+[
                ROOT/'citf/tools/render_unified_results.py']:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        # Exercise the real observed-pose interpolation without importing ROS.
        tree = ast.parse((ROOT/'vrx/run_experiment.py').read_text(encoding='utf-8-sig'))
        trial = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Trial')
        names = {'initialize_terminal_feedback', 'record_terminal_pose', 'terminal_snapshot', 'outcome'}
        functions = ast.Module(body=[node for node in trial.body if isinstance(node, ast.FunctionDef)
                                     and node.name in names], type_ignores=[])
        namespace = dict(np=np, deque=deque)
        exec(compile(functions, 'vrx_observed_feedback', 'exec'), namespace)
        feedback = SimpleNamespace(args=SimpleNamespace(termination_rule='paper'),
                                   total_time=60., target_r=15., collision_r=5., defend_r=5.)
        namespace['initialize_terminal_feedback'](feedback, 4)
        for index, point in enumerate(([16.2, 0.], [0., 30.], [30., 30.], [30., -30.])):
            namespace['record_terminal_pose'](feedback, index, 59.9, point, 0.)
            later = [14.2, 0.] if index == 0 else point
            namespace['record_terminal_pose'](feedback, index, 60.1, later, 0.)
        final_pose = namespace['terminal_snapshot'](feedback, deadline=60.)
        self.assertEqual(final_pose['stamp'], 60.)
        np.testing.assert_allclose(final_pose['positions'][0], [15.2, 0.], atol=1e-12)
        self.assertEqual(namespace['outcome'](feedback, 60., final_pose['positions']), 4)
        from unified import train_baseline, train_alternating, evaluate, run_study
        self.assertTrue(all(callable(module.main) for module in (train_baseline, train_alternating, evaluate, run_study)))
        # Formal phase resume inherits its saved registry before validating the source.
        with tempfile.TemporaryDirectory() as temporary:
            registry_path = Path(temporary)/'registry.json'
            write_json(registry_path, registry)
            saved = dict(implementation_version=IMPLEMENTATION_VERSION, training_state={}, transitions=5000,
                         arguments=dict(stage='formal', seed=2101, phase=2, registry=str(registry_path)))
            class ReachedValidation(Exception):
                pass
            def accept_resume(payload, restored_registry, phase, completed):
                self.assertEqual(restored_registry, registry)
                self.assertEqual(phase, 2)
                self.assertFalse(completed)
                raise ReachedValidation()
            with patch.object(sys, 'argv', ['alternating', '--previous', 'parent.pt', '--phase', '2',
                                           '--out', temporary, '--resume', 'saved.pt']), \
                    patch.object(torch, 'load', return_value=saved), \
                    patch.object(train_alternating, 'validate_phase_checkpoint', side_effect=accept_resume):
                with self.assertRaises(ReachedValidation):
                    train_alternating.main()


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)
