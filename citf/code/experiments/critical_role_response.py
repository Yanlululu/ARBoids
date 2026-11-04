"""Read candidate responses through the boats relevant to defense geometry.

The reference is fixed before candidate scoring. Radial coverage, possible
capture, and close defender pairs identify smooth, permutation-equivariant
roles. Their candidate-conditioned representations supplement the global
pooled features. The new readout starts at zero, retaining the strong physical
controller, and learns ordinary current-configuration actions from full tasks.
"""
import argparse
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import sys

import torch
from torch import nn

from citf.train import atomic_save
from experiments import candidate_response as response
from experiments import response_policy as policy


@dataclass(frozen=True)
class CriticalResponseConfig(response.ResponseConfig):
    revision: str = 'CITF-fixed-reference-critical-role-response-v1'
    critical_role_temperature: float = 2.
    role_condition: str = 'fixed-reference radial defender, capture defender, conflict pair, attacker'


def role_weights(batch, temperature):
    """[B,4,E] reference-only role weights, independent of the candidate pool."""
    nominal = batch['reference_nominal']
    roots, entities, horizon = nominal.shape[:3]
    defenders = entities - 1
    valid = batch['entity_valid'][:, :-1]
    time_valid = batch['known_time_valid']
    radius = nominal.norm(dim=-1)
    radial = radius[:, :-1].masked_fill(~valid[..., None], torch.inf).amin(1) - radius[:, -1]
    worst_time = radial.masked_fill(~time_valid, -torch.inf).argmax(-1)
    inner_radius = radius[:, :-1].gather(-1, worst_time[:, None, None].expand(roots, defenders, 1)).squeeze(-1)
    inner = (-inner_radius / temperature).masked_fill(~valid, -torch.inf).softmax(-1)

    distance = (nominal[:, :-1] - nominal[:, -1:]).norm(dim=-1)
    nearest = distance.masked_fill(~time_valid[:, None], torch.inf).amin(-1)
    capture = (-nearest / temperature).masked_fill(~valid, -torch.inf).softmax(-1)

    pair_distance = (nominal[:, :-1, None] - nominal[:, None, :-1]).norm(dim=-1)
    minimum = pair_distance.masked_fill(~time_valid[:, None, None], torch.inf).amin(-1)
    diagonal = torch.eye(defenders, device=nominal.device, dtype=torch.bool)[None]
    pairs = valid[:, :, None] & valid[:, None, :] & ~diagonal
    pair_logits = (-minimum / temperature).masked_fill(~pairs, -torch.inf)
    pair_mass = pair_logits.flatten(1).softmax(-1).reshape(roots, defenders, defenders)
    conflict = .5 * (pair_mass.sum(-1) + pair_mass.sum(-2))

    weights = nominal.new_zeros(roots, 4, entities)
    weights[:, 0, :-1] = inner
    weights[:, 1, :-1] = capture
    weights[:, 2, :-1] = conflict
    weights[:, 3, -1] = 1.
    return weights


class RoleReadout(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.context = None
        self.net = nn.Sequential(nn.LayerNorm(12 * width + 16),
            nn.Linear(12 * width + 16, width), nn.GELU(), nn.Linear(width, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, global_features):
        if self.context is None:
            raise RuntimeError('Role-conditioned readout requires the current reference and candidate encodings')
        return self.net(torch.cat((global_features, self.context), -1))


class CriticalResponseModel(response.CandidateResponseModel):
    def __init__(self, cfg=CriticalResponseConfig()):
        super().__init__(cfg)
        self.actor = RoleReadout(cfg.width)
        self._reference_encoding = None

    def encode_candidates(self, batch, nominal, candidates, history):
        encoded = super().encode_candidates(batch, nominal, candidates, history)
        if self._reference_encoding is None:
            self._reference_encoding = encoded
        else:
            weights = role_weights(batch, self.cfg.critical_role_temperature)
            roles = torch.einsum('bre,bcew->bcrw', weights, encoded).flatten(-2)
            reference = torch.einsum('bre,bcew->bcrw', weights, self._reference_encoding).flatten(-2)
            self.actor.context = torch.cat((roles,
                (roles - reference) / self.cfg.policy_reference_feature_scale), -1)
        return encoded

    def forward(self, batch, history_cache=None):
        self._reference_encoding = None
        self.actor.context = None
        try:
            return super().forward(batch, history_cache)
        finally:
            self._reference_encoding = None
            self.actor.context = None


_base_initialize = policy.initialize


def bind_policy():
    policy.ResponseConfig = CriticalResponseConfig
    policy.CandidateResponseModel = CriticalResponseModel
    policy.initialize = initialize


def initialize(project, source, config):
    bind_policy()
    _base_initialize(project, source, config)


def initialize_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    source = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    config = CriticalResponseConfig(**dict(source['config'],
        revision=CriticalResponseConfig().revision, deployment='policy',
        policy_prior_mode='reference_saturated'))
    model = CriticalResponseModel(config)
    inherited = {key: value for key, value in source['state'].items() if not key.startswith('actor.')}
    model.load_state_dict(inherited, strict=False)
    protocol = dict(source['protocol'], response_pretrained=args.pretrained,
        critical_role_initialization=dict(seed=args.seed,
            inherited='all response/history/interaction parameters; new zero-output critical-role actor',
            geometry='same fixed physical reference for every candidate; no candidate-pool statistics',
            objective='ordinary task policy gradient; identical response replay',
            status='initialized; no task training or performance claim'))
    target = Path(args.out)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(config=asdict(config), state=model.state_dict(), protocol=protocol,
        seed=args.seed, step=source.get('step'), update=0, training_episodes=0), target)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode in ('initialize', 'train'):
        sys.argv.pop(1)
        if mode == 'initialize':
            initialize_main()
        else:
            bind_policy()
            policy.main()
        return
    # The core evaluator owns task execution and records the complete config.
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    original_identity = evaluation.runtime_identity

    def identity(*args, **kwargs):
        result = original_identity(*args, **kwargs)
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            critical_role_response=file_hash(__file__)))
        return result

    evaluation.runtime_identity = identity
    response.ResponseConfig = CriticalResponseConfig
    response.CandidateResponseModel = CriticalResponseModel
    response.main()


if __name__ == '__main__':
    main()
