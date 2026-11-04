"""Equal mean of all registered contact-policy repetitions at event times."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

from citf.train import atomic_save
from experiments.contact_event_policy import (
    ContactEventConfig, ContactEventController, ContactEventModel)


class ContactEventEnsemble(nn.Module):
    def __init__(self, members):
        super().__init__()
        self.cfg = ContactEventConfig(**members[0]['config'])
        self.members = nn.ModuleList()
        for item in members:
            if item['config'] != members[0]['config']:
                raise ValueError('Equal-mean members must share the contact controller definition')
            model = ContactEventModel(ContactEventConfig(**item['config']))
            model.load_state_dict(item['state'])
            self.members.append(model)

    def forward(self, batch):
        logits = torch.stack([model(batch)['logits'] for model in self.members])
        return dict(logits=logits.mean(0))


def build():
    parser = argparse.ArgumentParser()
    parser.add_argument('--members', nargs=3, required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise ValueError('Refuse ensemble checkpoint overwrite')
    values = [torch.load(path, map_location='cpu', weights_only=False) for path in args.members]
    first = values[0]
    for value in values:
        if value['config'] != first['config']:
            raise ValueError('Ensemble member controller definitions differ')
        for key in ['source_sha256', 'environment_sha256']:
            if value['protocol'][key] != first['protocol'][key]:
                raise ValueError('Ensemble source or environment differs')
    members = [dict(config=value['config'], state=value['state'],
                    checkpoint=str(path), update=value['update'], seed=value['seed'])
               for value, path in zip(values, args.members)]
    protocol = dict(first['protocol'], ensemble=dict(
        reduction='equal arithmetic mean of all three policy logits',
        members=[{k: v for k, v in item.items() if k != 'state'} for item in members],
        selection='each completed80update repetition selected by original64validation; no subset or mean weight selection',
        scope='shared frozen teacher and physical dataset; not independent teachers'))
    atomic_save(dict(config=first['config'], protocol=protocol, members=members), args.out)
    print(json.dumps(dict(out=args.out, members=[x['checkpoint'] for x in members])), flush=True)


def load_model(path, source, world, device):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value['protocol']['source_sha256'] != source.sha256 or value['protocol']['environment_sha256'] != world.environment_sha256:
        raise ValueError('Ensemble source/environment mismatch')
    return ContactEventEnsemble(value['members']).to(device).eval()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'build':
        sys.argv.pop(1)
        return build()
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    from experiments import contact_event_policy
    evaluation.load_behavior = load_model
    evaluation.Controller = ContactEventController
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        checkpoint = args[2] if len(args) > 2 else kwargs['checkpoint']
        stored = torch.load(checkpoint, map_location='cpu', weights_only=False)
        value['method'] = 'CITF-retained-contact-event-equal-mean-development'
        value = contact_event_policy.contact_runtime_identity(value, stored)
        value['controller']['ensemble'] = stored['protocol']['ensemble']
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'],
            ensemble=file_hash(__file__)))
        return value

    evaluation.runtime_identity = identity
    evaluation.main()


if __name__ == '__main__':
    main()
