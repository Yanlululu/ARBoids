"""Score a joint correction through its actual changes in current thrust.

The retained proposal has exactly zero learned residual. Each defender's
two-channel thrust difference multiplies a shared learned sensitivity that
reads its candidate-conditioned fleet context, retained context, and observed
relative motion. Complete returns still train only the action-ranking loss;
there is no return, success, or continuation-value prediction head.
"""
import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import sys

import torch
from torch import nn

from citf.train import atomic_save
from experiments import contact_event_policy as contact
from experiments import contact_score_anchor_policy as anchor
from experiments.candidate_response import CandidateResponseModel

common = anchor.common
training = anchor.training


@dataclass(frozen=True)
class ActuationConfig(anchor.ScoreAnchorConfig):
    revision: str = 'CITF-contact-retained-relative-actuation-readout-v1'
    action_readout: str = 'sum of current thrust differences times candidate-conditioned fleet sensitivities'
    readout_reference: str = 'retained teacher proposal for task decisions; physical reference for short replay'


class ActuationReadout(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(3 * width + 24),
            nn.Linear(3 * width + 24, width), nn.GELU(), nn.Linear(width, 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.context = None
        self.last_score = None

    def forward(self, unused_global_features):
        if self.context is None:
            raise RuntimeError('Actuation readout requires the current candidate encodings')
        features, difference, valid = self.context
        sensitivity = self.net(features)
        # candidate[..., 6:8] is the actual normalized mixed source command.
        # Identical current forces have zero difference even when gates differ.
        self.last_score = (sensitivity * difference * valid).sum((-2, -1))
        return self.last_score[..., None]


class ActuationModel(contact.ContactEventModel):
    def __init__(self, cfg=ActuationConfig()):
        super().__init__(cfg)
        self.actor = ActuationReadout(cfg.width)
        # Event information enters the per-defender readout directly.
        del self.phase_embed
        self._encoding_calls = 0

    def encode_candidates(self, batch, nominal, candidates, history):
        encoded = super().encode_candidates(batch, nominal, candidates, history)
        self._encoding_calls += 1
        if self._encoding_calls != 2:
            return encoded
        roots, count, entities, width = encoded.shape
        defenders = entities - 1
        if 'teacher_logits' in batch:
            retained = batch['teacher_logits'].argmax(-1)
        else:
            retained = torch.zeros(roots, dtype=torch.long, device=encoded.device)
        rows = torch.arange(roots, device=encoded.device)
        reference = encoded[rows, retained][:, None, :-1].expand(roots, count, defenders, width)
        current = encoded[:, :, :-1]
        commands = batch['candidate'][:, :, :-1, 6:8]
        retained_commands = commands[rows, retained][:, None].expand_as(commands)
        difference = commands - retained_commands

        position = batch['positions']
        last = batch['history'][:, :, -1]
        relative_position = (position[:, :-1] - position[:, -1:]) / 20.
        relative_velocity = last[:, :-1, 2:4] - last[:, -1:, 2:4]
        radial = (position[:, :-1].norm(dim=-1) - position[:, -1:].norm(dim=-1)) / 20.
        geometry = torch.cat((relative_position, relative_velocity,
            last[:, :-1, 4:6], last[:, :-1, 6:7], radial[..., None]), -1)
        geometry = geometry[:, None].expand(roots, count, defenders, 8)
        phase = batch.get('event_features', encoded.new_zeros(roots, 10))
        phase = phase[:, None, None].expand(roots, count, defenders, 10)
        features = torch.cat((current, reference,
            (current - reference) / self.cfg.policy_reference_feature_scale,
            geometry, phase, commands, retained_commands, difference), -1)
        valid = batch['entity_valid'][:, None, :-1, None]
        self.actor.context = features, difference, valid
        return encoded

    def forward(self, batch, history_cache=None):
        self._encoding_calls = 0
        self.actor.context = None
        self.actor.last_score = None
        try:
            # Reuse the same physical-response network and loss outputs, while
            # replacing the pooled global actor and its phase embedding.
            output = CandidateResponseModel.forward(self, batch, history_cache)
            if 'teacher_logits' in batch:
                logits = batch['teacher_logits'] + self.cfg.policy_logit_scale * self.actor.last_score
                output['logits'] = logits.masked_fill(~batch['candidate_valid'], -torch.inf)
            return output
        finally:
            self.actor.context = None
            self.actor.last_score = None


def initialize(project, source, config):
    contact.ContactEventModel = ActuationModel
    training.ModelConfig = ActuationConfig
    common._initialize_training_worker(project, source, config)


def initialize_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seed', type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    value = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    cfg = ActuationConfig(**dict(value['config'], revision=ActuationConfig().revision,
        decision_domain=ActuationConfig().decision_domain,
        training_action_sampling=ActuationConfig().training_action_sampling))
    model = ActuationModel(cfg)
    inherited = {key: tensor for key, tensor in value['state'].items()
                 if not key.startswith(('actor.', 'phase_embed.'))}
    loaded = model.load_state_dict(inherited, strict=False)
    if loaded.unexpected_keys or any(not key.startswith('actor.') for key in loaded.missing_keys):
        raise ValueError('The source does not contain the complete shared response encoder')
    protocol = dict(value['protocol'], actuation_readout_initialization=dict(
        source=args.pretrained, seed=args.seed,
        inherited='all history, candidate-interaction and physical-response parameters',
        replaced='pooled global actor and phase embedding; new zero-output per-defender actuation readout',
        reference='same observed retained proposal for every candidate in the current bank',
        score='sum_i delta_normalized_thrust_i dot sensitivity_i(candidate fleet context,retained context,relative motion,event)',
        training='no additional physical pretraining; current-action comparisons remain real task continuations',
        status='initialized; not trained or evaluated'))
    target = Path(args.out)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(dict(config=asdict(cfg), state=model.state_dict(), protocol=protocol,
        seed=args.seed, update=0, training_episodes=0), target)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ''
    if mode == 'initialize':
        sys.argv.pop(1)
        return initialize_main()
    contact.ContactEventModel = ActuationModel
    if mode == 'train':
        sys.argv.pop(1)
        anchor.optimization_diagnostics.clear()
        training.ModelConfig = ActuationConfig
        training.COMPARISON_PROPOSALS = ('teacher', 'leading_alternative', 'sampled_alternative')
        training.initialize = initialize
        training.rollout = anchor.rollout
        training.event_examples = anchor.event_examples
        training.ACTOR_OBJECTIVE = anchor.anchored_objective
        training.PROTOCOL_OVERRIDES = dict(
            objective='unchanged three-action mean-return-weighted ranking and0.25unobserved promotion penalty',
            action_readout=ActuationConfig().action_readout,
            retained_score='learned residual is structurally zero for the retained actual thrust',
            input='candidate and retained per-entity Transformer encodings,observed relative motion,event phase and actual mixed-thrust differences',
            context='candidate-conditioned fleet interaction remains in the shared response encoder; no candidate-pool normalization',
            policy_change='only current-action readout; no critic,terminal prediction,CV,action restriction or new deployment threshold',
            unchanged='original all-candidate bank,teacher,1.5deployment margin,three proposals by two common futures,physical replay,80updates,original64selection',
            budget='same320root schedule,1920realepisodes and actual477optimizer/replay schedule; new readout changes parameter count and per-candidate compute',
            comparison='single main experiment first; compare complete common768 with score-anchor and unregularized tournament,then decide whether controls are informative',
            hypothesis='explicitly represent which current thrust changes in its fleet context instead of inferring every correction from a pooled global feature; benefit is unverified')
        return training.train_main()
    common.CommonFutureConfig = ActuationConfig
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'],
            actuation_readout=file_hash(__file__)))
        return value

    evaluation.runtime_identity = identity
    return common.main()


if __name__ == '__main__':
    main()
