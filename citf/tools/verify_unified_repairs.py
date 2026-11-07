"""Focused deterministic acceptance of the two adopted review recommendations.

No research training, parameter search, scenario benchmark or result selection.
Run: python -X utf8 citf/tools/verify_unified_repairs.py
"""
import ast
import copy
from collections import deque
from concurrent.futures import Future
from dataclasses import asdict, replace
import importlib.util
import io
import json
import math
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import time
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
from unified.networks import CITF, ModelConfig, TeamCritic, relative_geometry, nearest_other
from unified.learner import Learner, Replay
from unified.train import (arguments, policy_schedule, rollout_count, write_json, prepare_output,
                           append_evaluation, resume_arguments)
from unified.protocol import (baseline_updates, build_registry, load_registry, registered_arguments,
    run_identity, bind_run, canonical_arguments, model_definition, learning_counts, checkpoint_identity,
    validate_checkpoint, evaluation_identity, attach_identity, validate_result, outcome_metrics,
    phase_identity, phase_counts, ATTACKER_RECIPE, collection_updates, digest, policy_digest,
    validate_learning_curve, validate_phase_checkpoint, chain_evaluation_identity, vrx_rows, validate_optimizer_counts,
    vrx_source_identity, VRX_ADAPTER_CONTRACT)


def recipe():
    with patch.object(sys, 'argv', ['train', '--out', 'unused', '--feedback', '.01', '--batch', '512']):
        args = arguments()
    return dict(arguments=vars(args), task=asdict(TaskConfig()), implementation_version=IMPLEMENTATION_VERSION)


def rows(count=100, seed=74001):
    return [dict(scene_seed=seed, scene_index=i, outcome=i % 4+1, seconds=60. if i % 4 == 3 else 10.+i % 4,
                 success=int(i % 4 >= 2), breach=int(i % 4 == 0), collision=int(i % 4 == 1),
                 capture=int(i % 4 == 2), timeout=int(i % 4 == 3)) for i in range(count)]


def ros_free_class(file, name, methods, namespace=None):
    tree = ast.parse((ROOT/'vrx'/file).read_text(encoding='utf-8-sig'))
    source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    selected = [copy.deepcopy(node) for node in source.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    scope = dict(np=np, torch=torch, deque=deque, math=math, execute=execute)
    scope.update(namespace or {})
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(file), 'exec'), scope)
    return type(name, (), {key: scope[key] for key in methods})


def entry_probe():
    """Use real CLI entry points; interrupt only after an atomic checkpoint."""
    name, stop = sys.argv[2], int(sys.argv[3])
    sys.argv = [name]+sys.argv[4:]
    module = importlib.import_module(name)
    if name == 'unified.run_study':
        def inspect_launch(command, **kwargs):
            with patch.object(sys, 'argv', [command[3]]+command[4:]):
                actual = vars(arguments())
            registry = load_registry(actual['registry'])
            expected = registered_arguments(registry, 'full', actual['seed'])
            if canonical_arguments(actual) != expected:
                raise ValueError('The real formal launch command differs from its registry.')
            print('Formal entry registered its actual source and produced the exact training command.', flush=True)
            raise SystemExit(86)
        module.subprocess.Popen = inspect_launch
    elif stop:
        original = module.append_evaluation
        def after_checkpoint(out, evaluation):
            original(out, evaluation)
            if evaluation['transitions'] == stop:
                raise SystemExit(85)
        module.append_evaluation = after_checkpoint
    print(f'Actual entry: {module.__file__}; {IMPLEMENTATION_VERSION}', flush=True)
    module.main()


class Repairs(unittest.TestCase):
    def assertClose(self, a, b, atol=2e-6, rtol=2e-5):
        torch.testing.assert_close(torch.as_tensor(a), torch.as_tensor(b), atol=atol, rtol=rtol)

    @unittest.skipUnless(os.environ.get('CITF_ACCEPTANCE_DEVICE') == 'cuda', 'Explicit actual-GPU entry acceptance only.')
    def test_cuda_entry_resume(self):
        self.assertTrue(torch.cuda.is_available())
        largest = [0.]
        def compare(a, b):
            if isinstance(a, torch.Tensor):
                if a.is_floating_point():
                    self.assertTrue(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()))
                    if a.numel():
                        largest[0] = max(largest[0], float((a-b).abs().max()))
                self.assertClose(a, b, atol=0, rtol=0)
            elif isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            elif isinstance(a, dict):
                self.assertEqual(a.keys(), b.keys())
                for key in a:
                    compare(a[key], b[key])
            elif isinstance(a, (list, tuple)):
                self.assertEqual(len(a), len(b))
                for first, second in zip(a, b):
                    compare(first, second)
            else:
                self.assertEqual(a, b)
        with tempfile.TemporaryDirectory(prefix='citf-entry-') as temporary:
            root = Path(temporary)
            def run(name, stop, args, expected_code=0):
                result = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),
                    '--entry-probe', name, str(stop), *args], text=True, capture_output=True,
                    cwd=ROOT/'citf/code', timeout=180)
                self.assertEqual(result.returncode, expected_code, result.stdout[-5000:]+result.stderr[-5000:])
                self.assertIn(str(ROOT/'citf/code/unified'), result.stdout)
            recipe_path = os.environ['CITF_ACCEPTANCE_RECIPE']
            print(' Actual formal registration, then isolated CUDA collection/update/resume.', flush=True)
            run('unified.run_study', 0, ['--recipe', recipe_path, '--out', str(root/'formal-entry')], 86)
            common = ['--steps', '64', '--envs', '8', '--capacity', '128', '--batch', '16',
                      '--updates-per-step', '1', '--start-learning', '16', '--eval-interval', '32',
                      '--eval-count', '1', '--seed', '1197', '--device', 'cuda', '--width', '128',
                      '--layers', '2', '--samples', '4', '--target-tau', '.02',
                      '--feedback', str(1/300), '--feedback-start', '0', '--feedback-ramp', '0',
                      '--temperature-start', '1', '--temperature-end', '.05', '--temperature-steps', '100000',
                      '--endpoint-noise-start', '200', '--endpoint-noise-end', '50',
                      '--policy-lr-end', '.000025', '--policy-lr-decay-start', '500000',
                      '--policy-lr-decay-steps', '500000', '--value-terminal-fraction', '.5']
            reference, resumed = root/'continuous', root/'resumed'
            run('unified.train', 0, ['--out', str(reference), *common])
            print(' Continuous CUDA path completed; restarting the saved odd-update checkpoint.', flush=True)
            run('unified.train', 32, ['--out', str(resumed), *common], 85)
            middle = torch.load(resumed/'last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(middle['updates'] % 2, 1)
            validate_checkpoint(middle)
            run('unified.train', 0, ['--out', str(resumed), '--resume', str(resumed/'last.pt'), '--device', 'cuda'])
            saved = [torch.load(path/'last.pt', map_location='cpu', weights_only=False) for path in (reference, resumed)]
            for payload in saved:
                validate_checkpoint(payload)
                self.assertEqual(payload['arguments']['steps'], 64)
            for key in ('model', 'target', 'optimizers', 'updates', 'update_counts', 'transitions', 'training_state'):
                compare(saved[0][key], saved[1][key])
            compare(saved[0]['evaluation']['rows'], saved[1]['evaluation']['rows'])
            next_batches = []
            for payload in saved:
                replay = Replay(128, 3, 'cuda'); replay.load_state_dict(payload['training_state']['replay'])
                generator = torch.Generator(device='cuda')
                generator.set_state(payload['training_state']['random']['streams']['replay'].cpu())
                next_batches.append(replay.sample_value(16, generator))
            compare(next_batches[0], next_batches[1])
            Runtime = ros_free_class('unified_vrx.py', 'UnifiedTrial', {'compute_defense'}, dict(time=time))
            for device in ('cuda', 'cpu'):
                runtime = Runtime(); runtime.device = device
                runtime.policy = CITF(ModelConfig(**saved[0]['config'])).to(device)
                runtime.policy.load_state_dict(saved[0]['model'])
                env = FleetEnv(1, seed=41, device=device)
                result = runtime.compute_defense(env.observation(), env.order.clone(), {'episode': 'acceptance'})
                self.assertEqual(result[0].device.type, 'cpu')
                self.assertTrue(bool(torch.isfinite(result[0]).all()))
                self.assertTrue(bool((result[0].abs() <= 300.001).all()))
                self.assertGreater(result[1], 0.)
            print(f' CUDA resume maximum state/weight difference={largest[0]:.3g}; '
                  f'next replay batch and RNG identical; {torch.__version__}, CUDA {torch.version.cuda}, '
                  f'{torch.cuda.get_device_name()}', flush=True)

    def test_autoregressive_sampling_and_likelihood(self):
        maximum = 0.
        for n in (2, 3, 7):
            torch.manual_seed(400+n)
            model = CITF(ModelConfig(width=32, layers=2, samples=4))
            generator = model.generator
            x = FleetEnv(2, TaskConfig(defenders=n), seed=391+n, device='cpu').observation()
            source = boids(x); b = execute(source, x[:, :-1, 6:8], 300.)
            a = model.endpoint(x, source).detach()
            recorded = []
            original = generator.parameters_at
            def trace(*args):
                result = original(*args)
                recorded.append(tuple(value.detach().clone() for value in result))
                return result
            with patch.object(generator, 'parameters_at', side_effect=trace):
                sampled = generator.sample(x, a, b, 4)
            self.assertFalse(sampled.requires_grad)
            def conditional(logits, shape, value):
                logs = logits.log_softmax(-1)
                interior = (value > 0) & (value < 1)
                density = torch.distributions.Beta(shape[..., 0], shape[..., 1]).log_prob(
                    torch.where(interior, value, .5))
                return torch.where(value == 0, logs[..., 0],
                                   torch.where(value == 1, logs[..., 1], logs[..., 2]+density))
            sequential = sum(conditional(logits[:, j], shape[:, j], sampled.reshape(-1, n)[:, j])
                             for j, (logits, shape) in enumerate(recorded)).reshape(2, 4)
            parallel = generator.log_prob(x, a, b, sampled)
            maximum = max(maximum, float((parallel-sequential).detach().abs().max()))
            self.assertClose(parallel, sequential, atol=2e-5)
            near = torch.tensor([0., 1., torch.finfo(torch.float32).tiny, 1-torch.finfo(torch.float32).eps/2, .25, .75, .5])[:n]
            optimized = model.select(x)['candidates']
            configurations = torch.cat((torch.zeros(2, 1, n), torch.ones(2, 1, n),
                                         near.expand(2, 1, n), optimized), 1)
            count = configurations.shape[1]
            geometry = relative_geometry(x)
            z = generator.encode(x, a, b, geometry)[:, None].expand(-1, count, -1, -1).reshape(2*count, n, -1)
            pair = geometry[:, None, :n, :n].expand(-1, count, -1, -1, -1).reshape(2*count, n, n, 8)
            u = fusion(x, a, b, configurations, 300.)
            tokens = torch.cat((configurations[..., None], u/1000, torch.ones_like(configurations[..., None])), -1).reshape(2*count, n, 4)
            prefix = torch.zeros_like(tokens); terms = []
            for j in range(n):
                logits, shape = original(z, prefix, pair)
                terms.append(conditional(logits[:, j], shape[:, j], configurations.reshape(-1, n)[:, j]))
                altered = prefix.clone(); altered[:, j:] = torch.randn_like(altered[:, j:])
                changed_logits, changed_shape = original(z, altered, pair)
                self.assertClose(logits[:, :j+1], changed_logits[:, :j+1], atol=0, rtol=0)
                self.assertClose(shape[:, :j+1], changed_shape[:, :j+1], atol=0, rtol=0)
                prefix[:, j] = tokens[:, j]
            manual = torch.stack(terms, -1).sum(-1).reshape(2, count)
            aa, bb, gg = (value.detach().requires_grad_() for value in (a, b, configurations))
            joint = generator.log_prob(x, aa, bb, gg)
            maximum = max(maximum, float((joint-manual).detach().abs().max()))
            self.assertClose(joint, manual, atol=3e-5)
            self.assertTrue(bool(torch.isfinite(joint).all()))
            generator.zero_grad(); joint.sum().backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in generator.parameters()))
            self.assertTrue(all(value.grad is None for value in (aa, bb, gg)))
        print(f' autoregressive maximum log-probability error={maximum:.3g}')

    def test_worst_response_and_search_gradients(self):
        for n in (2, 3, 7):
            torch.manual_seed(615+n)
            critic = TeamCritic(ModelConfig(width=32, layers=1))
            x = FleetEnv(1, TaskConfig(defenders=n), seed=431+n, device='cpu').observation()
            encoded = critic.encode(x); pair = encoded[1][:, :n, :n]
            allowed = nearest_other(pair, 2) if n > 3 else ~torch.eye(n, dtype=torch.bool)[None]
            command = encoded[-1]+torch.tensor([1000., 0.])
            patterns = ([2., 4.], [-2., -4.], [2., -4.], [0., 0.], [-2., -2.])
            for pattern in patterns:
                values = torch.full((1, n, n), 999.)
                for i in range(n):
                    selected = allowed[0, i].nonzero().flatten()
                    values[0, i, selected] = torch.tensor(pattern[:len(selected)])
                values.requires_grad_()
                def zeros(z):
                    return z.new_zeros(*z.shape[:-1], 1)
                with patch.object(critic.value, 'forward', side_effect=zeros), \
                        patch.object(critic.readout, 'forward', side_effect=zeros), \
                        patch.object(critic.interaction_readout, 'forward', side_effect=lambda _: torch.stack((values, torch.zeros_like(values)), -1)):
                    actual = critic.from_encoded(encoded, command, individual=True)
                local = sum(max(v, 0.) for v in pattern[:min(2, n-1)])/min(2, n-1)+min(0., *pattern[:min(2, n-1)])
                expected = local+(min(0., local) if n > 3 else 0.)
                self.assertClose(actual, torch.full((1, n), expected))
                if n == 3 and pattern in ([0., 0.], [-2., -2.]):
                    derivative = torch.autograd.grad(actual[0, 0], values)[0]
                    self.assertClose(derivative[0, 0], torch.tensor([0., .5, .5]), atol=0, rtol=0)
        errors = []
        for n in (2, 3, 7):
            torch.manual_seed(870+n)
            model = CITF(ModelConfig(width=32, layers=2, samples=4)).double()
            with torch.no_grad():
                for q in (model.q1, model.q2):
                    q.interaction_readout[-1].weight.normal_(0, .07)
                    q.interaction_readout[-1].bias.normal_(0, .07)
            x = FleetEnv(1, TaskConfig(defenders=n), seed=81+n, device='cpu').observation().double()
            source = boids(x); b = execute(source, x[:, :-1, 6:8], 300.)
            a = model.endpoint(x, source).detach(); encoded = model.encode_value(x)
            g = torch.linspace(.3, .7, n, dtype=torch.float64)[None].requires_grad_()
            def score(coefficients):
                return model.value_from_encoded(encoded, fusion(x, a, b, coefficients, 300.))
            automatic = torch.autograd.grad(score(g).sum(), g)[0]
            finite = torch.zeros_like(g); h = 1e-5
            for i in range(n):
                delta = torch.zeros_like(g); delta[0, i] = h
                finite[0, i] = (score(g.detach()+delta)-score(g.detach()-delta))/(2*h)
            error = float((automatic-finite).detach().abs().max()); errors.append(error)
            self.assertClose(automatic, finite, atol=2e-7, rtol=2e-4)
            result = model.select(x)
            selected = model.value_from_encoded(model.encode_value(x), result['action'])
            self.assertTrue(bool((selected >= result['initial_pool_scores'].amax(-1)-1e-10).all()))
        print(' search gradient maximum absolute errors='+','.join(f'{v:.3g}' for v in errors))

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
        fixed = dict(root_args, lr=1e-4, policy_lr_end=2.5e-5, policy_lr_decay_start=500000,
                     policy_lr_decay_steps=500000, temperature_start=1., temperature_end=.05,
                     temperature_steps=100000, exploration_steps=100000,
                     endpoint_noise_start=200., endpoint_noise_end=50., feedback=1/300,
                     feedback_start=50000, feedback_ramp=50000)
        for transitions in (499999, 500000, 500001, 999999, 1000000, 1000001, 1499999, 1500000):
            result = policy_schedule(fixed, transitions)
            expected_lr = 1e-4-7.5e-5*min(1., max(0., (transitions-500000)/500000))
            self.assertAlmostEqual(result['policy_lr'], expected_lr, places=15)
            self.assertAlmostEqual(result['temperature'], .05)
            self.assertAlmostEqual(result['endpoint_noise'], 50.)
            self.assertAlmostEqual(result['probability'], .1)
            self.assertEqual(result['feedback_scale'], 1.)
        # Stage-local update entitlement starts at its own new replay warmup,
        # even when the defender's policy schedule has reached 1M transitions.
        self.assertEqual(collection_updates(4999, root_args), 0)
        self.assertEqual(collection_updates(5000, root_args), root_args['updates_per_step'])
        self.assertEqual(policy_schedule(fixed, 1500000)['policy_lr'], policy_schedule(fixed, 1000000)['policy_lr'])

    def test_vrx_event_history_and_async_isolation(self):
        Feedback = ros_free_class('run_experiment.py', 'Trial',
            {'initialize_terminal_feedback', 'start_terminal_monitor', 'record_terminal_pose', 'on_pose',
             'terminal_snapshot', 'next_terminal_snapshot', 'outcome'})
        def monitor():
            trial = Feedback()
            trial.args = SimpleNamespace(termination_rule='paper')
            trial.total_time, trial.target_r, trial.collision_r, trial.defend_r = 60., 15., 5., 5.
            trial.initialize_terminal_feedback(4); trial.start_terminal_monitor(0., .05)
            return trial
        normal = np.array([[60., 0.], [0., 0.], [20., 0.], [0., 20.]])
        def record(trial, stamp, positions):
            for index, position in enumerate(positions):
                trial.record_terminal_pose(index, stamp, position, 0.)
        trial = monitor(); record(trial, 0., normal)
        collision = normal.copy(); collision[2] = [4., 0.]
        record(trial, .05, collision); record(trial, .1, normal)
        breach = normal.copy(); breach[0] = [14., 0.]
        record(trial, .15, breach)
        event = trial.next_terminal_snapshot()
        self.assertEqual((event['outcome_code'], event['elapsed']), (2, .05))
        self.assertIs(trial.next_terminal_snapshot(), event)
        simultaneous = collision.copy(); simultaneous[0] = [14., 0.]
        self.assertEqual(trial.outcome(.05, simultaneous), 1)
        trial = monitor(); trial.terminal_tick = 1199
        for stamp in (59.9, 59.95, 60.):
            record(trial, stamp, normal)
        record(trial, 60.05, breach)
        event = trial.next_terminal_snapshot()
        self.assertEqual((event['outcome_code'], event['elapsed']), (4, 60.))
        trial = monitor(); record(trial, 0., normal); record(trial, .1, normal)
        with self.assertRaises(RuntimeError):
            trial.next_terminal_snapshot()
        for stamp, positions in ((float('nan'), normal), (-.05, normal), (.05, np.full_like(normal, np.nan))):
            with self.assertRaises(ValueError):
                record(monitor(), stamp, positions)
        with self.assertRaises(ValueError):
            trial.outcome(float('nan'), normal)
        for stamp in (float('nan'), -.05):
            trial = monitor(); trial.pose_data = {'wamv1': []}; trial.last_stamp = np.zeros(4)
            message = SimpleNamespace(transforms=[SimpleNamespace(child_frame_id='wamv1',
                header=SimpleNamespace(stamp=SimpleNamespace(sec=stamp, nanosec=0)))])
            with self.assertRaises(ValueError):
                trial.on_pose(message)
        clock = [0.]
        sys.path.insert(0, str(ROOT/'vrx'))
        from mappo_controller import synchronize_kinematics
        Controller = ros_free_class('unified_vrx.py', 'UnifiedTrial',
            {'poll_control', 'publish_commands', 'close_controller', 'control_accounting', 'control'},
            dict(time=SimpleNamespace(monotonic=lambda: clock[0]),
                 Float64=lambda **kwargs: SimpleNamespace(**kwargs), synchronize_kinematics=synchronize_kinematics))
        def controller():
            c = Controller(); c.args = SimpleNamespace(action_period=.2)
            c.episode_id, c.request_id, c.last_control_cycle, c.controller_closed = 'round-1', 1, 0, False
            c.common = FleetEnv(1, seed=51, device='cpu'); c.applied = torch.zeros(1, 4, 2)
            c.agility = 2.; c.pending = None; c.published = []
            c.publishers = [SimpleNamespace(publish=lambda message: c.published.append(message.data)) for _ in range(8)]
            c.rows = [dict(PolicyWallSeconds=0., ControlWallSeconds=0., MissedDeadline=0,
                DiscardedLateOutput=0, DiscardedStaleOutput=0, RequestId=1, ControlCycle=0,
                PolicyStarted=1, DefenderOutputApplied=0, TerminalCancellation=0, SkippedControlCycles=0)]
            return c
        for case in ('valid', 'late', 'old_round', 'wrong_request'):
            c = controller(); clock[0] = .3 if case == 'late' else .05
            identity = dict(episode='round-1', request=1, cycle=0, source_stamp=100., wall_deadline=.2, sim_deadline=.2)
            returned = dict(identity, request=9) if case == 'wrong_request' else identity.copy()
            future = Future(); future.set_result((torch.full((1, 3, 2), 450.), .01, clock[0], returned))
            c.pending = dict(future=future, row=0, started=0., elapsed=0., expired=False, identity=identity)
            if case == 'old_round':
                c.episode_id = 'round-2'
            c.poll_control(.05)
            self.assertEqual(len(c.published), 6 if case == 'valid' else 0)
            self.assertClose(c.applied[:, :-1], torch.full((1, 3, 2), 300. if case == 'valid' else 0.))
        c = controller(); c.rows[0]['DefenderOutputApplied'] = 1
        c.applied[:, :-1] = 170.
        physical = c.common.state[0, [3, 0, 1, 2]].numpy()
        c.curr_pos, c.curr_phi, c.curr_vel, c.yaw_rates = physical[:, :2], physical[:, 2], physical[:, 3:5], physical[:, 5]
        c.last_stamp = np.full(4, 100.4)
        submitted = []
        def submit(_fn, x, order, identity):
            future = Future(); future.set_running_or_notify_cancel()
            submitted.append((x.clone(), identity, future))
            return future
        def shutdown(**kwargs):
            self.assertTrue(c.controller_closed)
            self.assertFalse(submitted[-1][2].cancel())
            submitted[-1][2].set_result((torch.full((1, 3, 2), 450.), .45, .85, submitted[-1][1]))
        c.compute_defense = lambda *args: None
        c.executor = SimpleNamespace(submit=submit, shutdown=shutdown)
        clock[0] = .4; c.control(.4)
        self.assertClose(submitted[-1][0][:, :-1, 6:8], torch.full((1, 3, 2), 170.))
        clock[0] = .6; c.last_stamp[:] = 100.6; c.control(.6)
        before = len(c.published)
        c.close_controller(.95)
        self.assertEqual(len(c.published), before)
        self.assertClose(c.applied[:, :-1], torch.full((1, 3, 2), 170.))
        accounting = c.control_accounting(.95)
        self.assertEqual(accounting['control_steps'], 5)
        self.assertEqual(accounting['skipped_control_cycles'], 2)
        self.assertEqual(accounting['control_deadline_misses'], 4)
        # Completion after the episode ended is a cancellation, not an extra
        # deadline miss if both deadlines were still valid at termination.
        c = controller(); clock[0] = .05
        future = Future(); future.set_running_or_notify_cancel()
        identity = dict(episode='round-1', request=1, cycle=0, source_stamp=100., wall_deadline=.2, sim_deadline=.2)
        c.pending = dict(future=future, row=0, started=0., elapsed=0., expired=False, identity=identity)
        def finish_after_terminal(**kwargs):
            self.assertTrue(c.controller_closed)
            self.assertFalse(future.cancel())
            future.set_result((torch.zeros(1, 3, 2), .8, .8, identity))
        c.executor = SimpleNamespace(shutdown=finish_after_terminal)
        c.close_controller(.05)
        self.assertEqual(c.rows[0]['TerminalCancellation'], 1)
        self.assertEqual(c.rows[0]['MissedDeadline'], 0)
        self.assertEqual(c.published, [])

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
        self.assertEqual(registry['vrx_adapter_contract'], VRX_ADAPTER_CONTRACT)
        args = registered_arguments(registry, 'full', 2101)
        run = run_identity(args, registry['task'], registry)
        descriptor = dict(run_id=run['id'], transitions=1000000, sha256='fixed-checkpoint')
        task = dict(registry['task'], agility=2.25)
        identity = evaluation_identity(registry, 'full', 2101, descriptor, task, 74002, 84002, 100, 'vrx')
        adapter = vrx_source_identity(ROOT/'vrx')
        episodes = [dict(passed=True, identity=identity, adapter_source=adapter, scene_index=i,
                         adapter_contract=VRX_ADAPTER_CONTRACT,
                         policy_seed=84002+i, scene_seed=74002, task=task, outcome_code=i % 4+1,
                         simulation_seconds=60. if i % 4 == 3 else 12., success=i % 4 >= 2) for i in range(100)]
        data = vrx_rows(episodes, identity, adapter)
        self.assertEqual(outcome_metrics(data, 100, 74002, 60.)['success'], 50)
        for field, value in (('policy_seed', 84002), ('adapter_source', {'id': 'changed'}),
                             ('adapter_contract', 'legacy'), ('passed', False)):
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
        result = attach_identity(dict(outcome_metrics(data, 100, 74002, 60.), rows=data,
            task=task, scene_seed=74002, policy_seed=84002, adapter_source=adapter), identity)
        validate_result(result, identity)
        with self.assertRaises(ValueError):
            validate_result(dict(result, adapter_source=dict(adapter, contract='legacy')), identity)
        with tempfile.TemporaryDirectory(prefix='citf-release-routing-') as temporary:
            directory = Path(temporary)
            registry_path = directory/'registry.json'
            old = {k: v for k, v in registry.items() if k not in ('id', 'vrx_adapter_contract')}
            old['id'] = digest(old); write_json(registry_path, old)
            with self.assertRaises(ValueError):
                load_registry(registry_path)
            adapter_directory = directory/'vrx'; adapter_directory.mkdir()
            for source in (ROOT/'vrx').glob('*.py'):
                (adapter_directory/source.name).write_bytes(source.read_bytes())
            self.assertEqual(vrx_source_identity(adapter_directory), adapter)
            source = adapter_directory/'unified_vrx.py'
            source.write_text(source.read_text(encoding='utf-8').replace(
                f"ADAPTER_CONTRACT = '{VRX_ADAPTER_CONTRACT}'", "ADAPTER_CONTRACT = 'legacy'"), encoding='utf-8')
            with self.assertRaises(ValueError):
                vrx_source_identity(adapter_directory)
            legacy_directory = os.environ.get('CITF_LEGACY_VRX_SOURCE')
            if legacy_directory:
                with self.assertRaises(ValueError):
                    vrx_source_identity(legacy_directory)
            # Exercise actual formal registration and its real launch command,
            # but intercept Popen before any research training can start.
            selected = directory/'recipe.json'
            if os.environ.get('CITF_ACCEPTANCE_RECIPE'):
                selected.write_bytes(Path(os.environ['CITF_ACCEPTANCE_RECIPE']).read_bytes())
            else:
                write_json(selected, recipe())
            launch = subprocess.run([sys.executable, '-X', 'utf8', str(Path(__file__).resolve()),
                '--entry-probe', 'unified.run_study', '0', '--recipe', str(selected),
                '--out', str(directory/'study')], cwd=ROOT/'citf/code', text=True, capture_output=True, timeout=60)
            self.assertEqual(launch.returncode, 86, launch.stdout[-3000:]+launch.stderr[-3000:])
            self.assertEqual(load_registry(directory/'study/registry.json')['vrx_adapter_contract'], VRX_ADAPTER_CONTRACT)
            self.assertFalse(list((directory/'study').rglob('last.pt')))
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
    if len(sys.argv) > 1 and sys.argv[1] == '--entry-probe':
        entry_probe()
    else:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        unittest.main(verbosity=2)
