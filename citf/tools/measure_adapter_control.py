"""Development experiment: actual control from initial and optimized joint pools.

Uses fixed trained models and their actual replay. This does not add a formal
ablation, generate learning targets, or change an active training process.
"""
import argparse
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ.get('CITF_CODE_ROOT', str(Path(__file__).resolve().parents[1]/'code')))
from unified.environment import TaskConfig
from unified.networks import CITF, ModelConfig
from unified.protocol import require_standard_controller
from unified.train import evaluate, seed_policy, write_json, policy_schedule


def learning_signal(model, states, seed, device, temperature=.05, feedback_scale=1.):
    rng = torch.Generator(device='cpu').manual_seed(951000+seed)
    index = torch.randint(len(states), (1024,), generator=rng)
    parameters = list(model.generator.parameters())
    observations = []
    for start in range(0, len(index), 256):
        x = states[index[start:start+256]].to(device)
        result = model.select(x)
        g, scores = result['candidates'], result['scores']
        count = g.shape[1]
        duplicate = (g[:, :, None] == g[:, None, :]).all(-1)
        earlier = torch.ones(count, count, dtype=torch.bool, device=device).tril(-1)
        keep = ~(duplicate & earlier).any(-1)
        weights = (scores/temperature).masked_fill(~keep, -torch.inf).softmax(-1).detach()
        initial = result['initial_scores']/100.
        advantage = (initial-(initial.sum(-1, keepdim=True)-initial)/(model.cfg.samples-1)).detach()
        base = -(advantage*model.generator.log_prob(x, result['a'], result['b'], result['initial'])).mean()
        feedback = -(weights*model.generator.log_prob(x, result['a'], result['b'], g)).sum(-1).mean()
        first = torch.autograd.grad(base, parameters)
        second = torch.autograd.grad(model.cfg.feedback*feedback_scale*feedback, parameters)
        square1 = sum(a.square().sum() for a in first)
        square2 = sum(b.square().sum() for b in second)
        dot = sum((a*b).sum() for a, b in zip(first, second))
        gain = scores.amax(-1)-result['initial_pool_scores'].amax(-1)
        sample_best = result['initial_scores'].amax(-1)
        anchor_best = result['initial_pool_scores'][:, model.cfg.samples:].amax(-1)
        observations.append(dict(
            mean_value_gain=float(gain.mean()),
            improved_pool_fraction=float((gain > 1e-6).float().mean()),
            generated_beats_anchors=float((sample_best > anchor_best).float().mean()),
            base_gradient_norm=float(square1.sqrt()),
            weighted_feedback_gradient_norm=float(square2.sqrt()),
            gradient_cosine=float(dot/(square1*square2).sqrt().clamp_min(1e-12)),
            combined_norm_ratio=float((square1+square2+2*dot).clamp_min(0).sqrt() /
                                      (square1.sqrt()+square2.sqrt()).clamp_min(1e-12))))
    return dict(states=1024, temperature=temperature, feedback_scale=feedback_scale, batches=observations,
                means={key: sum(row[key] for row in observations)/len(observations)
                       for key in observations[0]})


def paired(initial, optimized):
    if (initial['task'] != optimized['task'] or initial['scene_seed'] != optimized['scene_seed']
            or initial['policy_seed'] != optimized['policy_seed'] or initial['tasks'] != optimized['tasks']
            or len(initial['rows']) != initial['tasks'] or len(optimized['rows']) != optimized['tasks']
            or [r['scene_index'] for r in initial['rows']] != [r['scene_index'] for r in optimized['rows']]):
        raise ValueError('Paired control comparisons require the same complete ordered scene bank and task.')
    pairs = list(zip(initial['rows'], optimized['rows']))
    return dict(
        optimized_minus_initial=optimized['success']-initial['success'],
        rescued=sum(a['success'] == 0 and b['success'] == 1 for a, b in pairs),
        harmed=sum(a['success'] == 1 and b['success'] == 0 for a, b in pairs),
        collision_change=optimized['collision']-initial['collision'],
        breach_change=optimized['breach']-initial['breach'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    parser.add_argument('--transitions', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    write_json(args.out/'configuration.json', dict(
        sources=[str(p) for p in args.checkpoint], transitions=args.transitions,
        research_stage='development', variants=['initial_pool', 'optimized_pool'],
        conditions=[dict(defenders=3, scene_seed=73001), dict(defenders=7, scene_seed=73005)],
        episodes_per_condition=100, policy_seed=83001, agility=2.,
        replay_states_per_source=1024, additional_training_transitions=0,
        purpose='Choose between improving action interaction/search and improving generator absorption.',
        reuse='The stored original three-vessel validation is reused; only new control conditions are simulated.'))
    summaries = []
    for source in args.checkpoint:
        payload = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
        require_standard_controller(payload)
        if payload['transitions'] != args.transitions:
            raise ValueError('Use the prospectively selected training node.')
        seed = payload['arguments']['seed']
        seed_policy(952000+seed, args.device)
        model = CITF(ModelConfig(**payload['config'])).to(args.device)
        model.load_state_dict(payload['model'])
        model.eval()
        original = model.cfg
        schedule = policy_schedule(payload['arguments'], args.transitions)
        signal = learning_signal(model, payload['training_state']['replay']['data']['x'], seed, args.device,
                                 schedule['temperature'], schedule['feedback_scale'])
        write_json(args.out/f'{seed}_learning_signal.json', signal)
        comparisons = []
        for defenders, scene_seed in ((3, 73001), (7, 73005)):
            task = replace(TaskConfig(**payload['task']), defenders=defenders, agility=2.)
            model.cfg = replace(original, search_steps=0)
            initial = evaluate(model, task, 100, scene_seed, 83001, args.device)
            initial.update(source=str(source), source_transitions=args.transitions,
                           variant='initial_pool', training_seed=seed)
            write_json(args.out/f'{seed}_initial_pool_team{defenders}.json', initial)
            model.cfg = original
            if defenders == 3:
                optimized = payload['evaluation']
                if (optimized['scene_seed'] != scene_seed or optimized['policy_seed'] != 83001
                        or optimized['tasks'] != 100 or optimized['task'] != asdict(task)
                        or optimized['transitions'] != args.transitions):
                    raise ValueError('The saved validation must use the same registered development condition.')
            else:
                optimized = evaluate(model, task, 100, scene_seed, 83001, args.device)
                optimized.update(source=str(source), source_transitions=args.transitions,
                                 variant='optimized_pool', training_seed=seed)
                write_json(args.out/f'{seed}_optimized_pool_team{defenders}.json', optimized)
            row = dict(seed=seed, defenders=defenders,
                       initial={k: initial[k] for k in ('success', 'collision', 'breach', 'mean_capture_seconds')},
                       optimized={k: optimized[k] for k in ('success', 'collision', 'breach', 'mean_capture_seconds')},
                       paired=paired(initial, optimized))
            comparisons.append(row)
            print(json.dumps(row), flush=True)
        summaries.append(dict(seed=seed, source=str(source), control=comparisons,
                              learning_signal=signal['means']))
        write_json(args.out/'results.json', dict(status='running', sources=summaries))
        del model, payload
        gc.collect()
        torch.cuda.empty_cache()
    write_json(args.out/'results.json', dict(status='completed', sources=summaries))
    write_json(args.out/'status.json', dict(status='completed', sources=len(summaries), new_evaluations=3*len(summaries), reused_evaluations=len(summaries)))


if __name__ == '__main__':
    main()
