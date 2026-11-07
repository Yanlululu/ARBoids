"""Matched actual-replay value learning with relative-action geometry.

The candidate exposes radial/transverse relative input acceleration to the
existing action-interaction layers. No future rollout, auxiliary target or
extra multi-vessel training data is used. Current training sources stay fixed.
"""
import argparse
import copy
from dataclasses import asdict, replace
import gc
import json
import os
from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, os.environ.get('CITF_CODE_ROOT', str(Path(__file__).resolve().parents[1]/'code')))
from unified.environment import TaskConfig
from unified.learner import Learner, Replay
from unified.networks import CITF, ModelConfig
from unified.protocol import require_standard_controller
from unified.train import evaluate, seed_policy, write_json, save


class RelativeActionLinear(nn.Linear):
    def forward(self, edge):
        direction = edge[..., :2]/edge[..., 4:5].clamp_min(1e-6)
        acceleration = edge[..., 8:10]
        radial = (direction*acceleration).sum(-1, keepdim=True)
        transverse = (direction[..., 0]*acceleration[..., 1]-direction[..., 1]*acceleration[..., 0])[..., None]
        return super().forward(torch.cat((edge, radial, transverse), -1))


def augment(model):
    def expanded(original):
        layer = RelativeActionLinear(original.in_features+2, original.out_features, bias=original.bias is not None)
        layer = layer.to(device=original.weight.device, dtype=original.weight.dtype)
        with torch.no_grad():
            layer.weight.zero_()
            layer.weight[:, :original.in_features].copy_(original.weight)
            if original.bias is not None:
                layer.bias.copy_(original.bias)
        return layer
    for critic in (model.q1, model.q2):
        for layer in critic.interaction_layers:
            layer.edge_bias[0] = expanded(layer.edge_bias[0])
            layer.edge_value = expanded(layer.edge_value)
        critic.interaction_pair[0] = expanded(critic.interaction_pair[0])


def expanded_payload(source, learner):
    result = {key: source[key] for key in ('config', 'updates', 'learner_recipe')}
    for key, module in (('model', learner.model), ('target', learner.target)):
        state = module.state_dict()
        for name, value in source[key].items():
            if state[name].shape == value.shape:
                state[name] = value
            else:
                expanded = torch.zeros_like(state[name], device='cpu')
                expanded[tuple(slice(0, size) for size in value.shape)] = value
                state[name] = expanded
        result[key] = state
    result['optimizers'] = copy.deepcopy(source['optimizers'])
    q_state = result['optimizers'][0]
    ids = [index for group in q_state['param_groups'] for index in group['params']]
    for index, parameter in zip(ids, learner.q_parameters):
        for key, value in q_state['state'][index].items():
            if torch.is_tensor(value) and value.ndim and value.shape != parameter.shape:
                expanded = value.new_zeros(parameter.shape)
                expanded[tuple(slice(0, size) for size in value.shape)] = value
                q_state['state'][index][key] = expanded
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    write_json(args.out/'configuration.json', dict(
        research_stage='development', sources=[str(p) for p in args.checkpoint],
        variants=['ordinary_refit', 'relative_action_refit'], critic_updates=args.updates,
        replay='same actual three-vessel replay, terminal mixture and importance weights',
        initialization='existing parameters and optimizer moments retained; added input columns and moments start at zero',
        frozen='learning endpoints and joint generator', target_update='original tau every two critic updates',
        conditions=[dict(defenders=3, agility=2., scene=73001), dict(defenders=7, agility=2., scene=73005),
                    dict(defenders=3, agility=3., scene=73004)], episodes=100, policy_seed=83001,
        additional_training_transitions=0))
    results = []
    for path in args.checkpoint:
        source = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        require_standard_controller(source)
        recipe, seed = source['arguments'], source['arguments']['seed']
        for variant in ('ordinary_refit', 'relative_action_refit'):
            seed_policy(953000+seed, args.device)
            model = CITF(ModelConfig(**source['config'])).to(args.device)
            if variant == 'relative_action_refit':
                augment(model)
            learner = Learner(model, lr=recipe['lr'], tau=recipe['target_tau'], value_huber_delta=recipe['value_huber_delta'])
            learner.restore(expanded_payload(source, learner) if variant == 'relative_action_refit' else source)
            learner.model.endpoint.requires_grad_(False)
            learner.model.generator.requires_grad_(False)
            learner.policy_delay = 2147483647
            replay = Replay(recipe['capacity'], source['task']['defenders'], args.device)
            replay.load_state_dict(source['training_state']['replay'])
            sampler = torch.Generator(device=args.device).manual_seed(954000+seed)
            seed_policy(955000+seed, args.device)
            for step in range(args.updates):
                batch = replay.sample_value(recipe['batch'], sampler, recipe['value_terminal_fraction'], recipe['value_importance_exponent'])
                metrics = learner.update(batch, robust_value=True)
                if learner.updates % 2 == 0:
                    learner.soft_update()
                if (step+1) % 100 == 0:
                    status = dict(status='fitting', seed=seed, variant=variant, updates=step+1,
                                  metrics={key: float(value) for key, value in metrics.items()})
                    write_json(args.out/'status.json', status)
                    print(json.dumps(status), flush=True)
            model.eval()
            records = []
            for defenders, agility, scene in ((3, 2., 73001), (7, 2., 73005), (3, 3., 73004)):
                task = replace(TaskConfig(**source['task']), defenders=defenders, agility=agility)
                result = evaluate(model, task, 100, scene, 83001, args.device)
                result.update(source=str(path), source_transitions=source['transitions'], seed=seed,
                              variant=variant, extra_critic_updates=args.updates, additional_training_transitions=0)
                write_json(args.out/f'{seed}_{variant}_team{defenders}_agility{agility:.1f}.json', result)
                row = {key: value for key, value in result.items() if key != 'rows'}
                records.append(row)
                print(json.dumps(row), flush=True)
            save(args.out/f'{seed}_{variant}.pt', dict(model=model.state_dict(), config=asdict(model.cfg), task=source['task'],
                 source=str(path), source_transitions=source['transitions'], extra_critic_updates=args.updates,
                 representation=variant, additional_training_transitions=0))
            results.append(dict(seed=seed, variant=variant, evaluations=records))
            write_json(args.out/'results.json', dict(status='running', records=results))
            del model, learner, replay, batch, metrics
            gc.collect()
            torch.cuda.empty_cache()
        del source
        gc.collect()
    write_json(args.out/'results.json', dict(status='completed', records=results))
    write_json(args.out/'status.json', dict(status='completed', refinements=len(results)))


if __name__ == '__main__':
    main()
