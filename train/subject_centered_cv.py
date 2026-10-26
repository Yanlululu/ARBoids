"""Share conditional intervention features across the selected vessel."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import fixed_pool_value_selection as selection

fixed = selection.fixed
e = fixed.e
OUT = fixed.OUT
native_initialize = fixed.initialize
native_load_actor = selection.load_actor
training = False


def ordered_state(context, controls, subjects, identity_conditioned):
    """Keep all observations, using subject-relative peer positions/velocities."""
    n = controls.shape[-2]
    rows = context[..., :8*n].reshape(-1, n, 8)
    batch = torch.arange(len(rows), device=rows.device)
    reference = rows[batch, subjects]
    distance = (rows[..., :2] - reference[:, None, :2]).square().sum(-1)
    distance[batch, subjects] = -1.
    order = distance.argsort(dim=-1, stable=True)
    ordered = rows.gather(1, order[..., None].expand(-1, -1, 8))
    relative = torch.cat((ordered[:, :1, :4],
                          ordered[:, 1:, :4] - ordered[:, :1, :4]), 1)
    ordered = torch.cat((relative, ordered[..., 4:]), -1).flatten(1)
    if identity_conditioned:
        identity = context[:, 8*n:9*n].gather(1, order)
        ordered = torch.cat((ordered, identity), -1)
    ordered = torch.cat((ordered, context[:, -5:]), -1)
    control = controls.gather(1, order[..., None].expand(-1, -1, 6)).flatten(1)
    return torch.cat((ordered, control), -1), order


def centered_features(network, features):
    n = network.defenders
    shape = features.shape
    flat = features.reshape(-1, shape[-1])
    context_size = (8 + network.identity_conditioned)*n + 5
    context = flat[:, :context_size]
    controls = flat[:, context_size:network.state_size].reshape(-1, n, 6)
    actions = flat[:, network.state_size:network.state_size+10*n].reshape(-1, n, 10)
    active = actions[..., 1:].sum(-1) > .5
    subjects = active.long().argmax(-1)
    state, order = ordered_state(context, controls, subjects, network.identity_conditioned)
    actions = actions.gather(1, order[..., None].expand(-1, -1, 10)).flatten(1)
    centered = torch.cat((state, actions, flat[:, -2:]), -1)
    return torch.where(active.any(-1, keepdim=True), centered, flat).reshape(shape)


def install():
    plain = sys.modules['plain_return']
    if hasattr(plain, '_subject_centered_originals'):
        return
    network_class = plain.PlainReturnNetwork
    native_features = network_class.forward_features
    native_catalog = network_class.forward_catalog
    plain._subject_centered_originals = (native_features, native_catalog)

    def enabled(network):
        return training or getattr(network, 'subject_centered_features', False)

    def forward_features(network, features):
        if enabled(network):
            assert network.action_encoding == 'categorical' and network.representation == 'dense'
            features = centered_features(network, features)
        return native_features(network, features)

    def forward_catalog(network, context, controls, programs, timing, chunk_size=1024):
        if not enabled(network):
            return native_catalog(network, context, controls, programs, timing, chunk_size)
        assert network.action_encoding == 'categorical' and network.representation == 'dense'
        assert not network.state_baseline
        n = network.defenders
        state = torch.cat((context, controls.flatten(1)), -1)
        variants = [state]
        for subject in range(n):
            subjects = torch.full((len(context),), subject, device=context.device, dtype=torch.long)
            variants.append(ordered_state(context, controls, subjects,
                                          network.identity_conditioned)[0])
        first = network.layers[0]
        hidden = F.linear(torch.stack(variants, 1),
                          first.weight[:, :network.state_size], first.bias)
        active = programs != 0
        assert bool((active.sum(-1) <= 1).all())
        subjects = active.long().argmax(-1)
        indices = torch.where(active.any(-1), subjects + 1, torch.zeros_like(subjects))
        centered_programs = torch.zeros_like(programs)
        centered_programs[:, 0] = programs.gather(1, subjects[:, None]).flatten()
        actions = network.action_features(centered_programs, timing)
        value = hidden[:, indices] + F.linear(actions, first.weight[:, network.state_size:])[None]
        for layer in list(network.layers)[1:]:
            value = layer(value)
        return value.squeeze(-1)

    network_class.forward_features = forward_features
    network_class.forward_catalog = forward_catalog


def initialize():
    state = native_initialize()
    install()
    return state


fixed.initialize = initialize


def load_actor(path, development):
    actor = native_load_actor(path, development)
    settings = e.read(Path(path)).get('subject_centered_value')
    if settings is not None:
        assert settings['source_sha256'] == e.digest(Path(__file__))
        decision = actor.successor_task_value
        for member in (decision.network, *decision.ensemble):
            member.subject_centered_features = True
    return actor


selection.load_actor = load_actor


def train(args):
    global training
    training = True
    try:
        fixed.train(args)
    finally:
        training = False
    arms = ('conditional', 'ordinary') if args.matched_ordinary else ('conditional',)
    for arm in arms:
        path = OUT / f'{args.name}-{arm}.pth'
        model = e.read(path)
        model['subject_centered_value'] = dict(source_sha256=e.digest(Path(__file__)),
            representation='Intervened vessel first; peers ordered by distance to that vessel.',
            relative_features='Peer positions and velocities relative to the intervened vessel.',
            preserved_observations='All fleet state, physical identity, public target bearing and commands.',
            factual_representation_unchanged=True, candidate_execution_unchanged=True,
            training_observations_unchanged=True, added_trainable_parameters=0)
        e.save(model, path)
    print(json.dumps(dict(phase='subject_centered_training_complete', seed=args.seed,
                         arms=arms)), flush=True)


def evaluate(args):
    selection.evaluate(args)
    specification = e.read(Path(args.design) if args.design else fixed.DESIGN)
    tasks = specification['tasks'] if args.design else specification['validation_tasks']
    count = len(range(args.shard_index, len(tasks), args.shards))
    path = OUT / f'{args.name}-part{args.shard_index}-n{count}-evaluation.pth'
    result = e.read(path)
    result['subject_centered_driver_sha256'] = e.digest(Path(__file__))
    e.save(result, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('train', 'evaluate', 'design'))
    parser.add_argument('--name', required=True)
    parser.add_argument('--policy')
    parser.add_argument('--policies', nargs='+', default=[])
    parser.add_argument('--selected-policy')
    parser.add_argument('--design')
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--batch', type=int, default=250)
    parser.add_argument('--updates', type=int, default=27000)
    parser.add_argument('--gain-weight', type=float, default=8.)
    parser.add_argument('--harm-weight', type=float, default=4.)
    parser.add_argument('--rank-weight', type=float, default=4.)
    parser.add_argument('--joint-weight', type=float, default=1.28)
    parser.add_argument('--matched-ordinary', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.operation == 'design':
        selection.design(args)
    else:
        globals()[args.operation](args)


if __name__ == '__main__':
    main()
