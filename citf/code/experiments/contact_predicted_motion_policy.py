"""Connect predicted near-contact motion to each defender's current actuation.

Candidate-conditioned short trajectories supply per-defender capture, velocity,
radial and collision geometry. Their relation to the retained action is read
alongside the existing interaction encoding and actual thrust difference.
Complete returns construct only the detached policy-improvement target.
"""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from citf.train import atomic_save
from experiments import contact_actuation_policy as actuation
from experiments import contact_policy_projection as projection

contact = projection.contact
common = projection.common
training = projection.training


@dataclass(frozen=True)
class PredictedMotionConfig(projection.ProjectionConfig):
    revision: str = 'CITF-contact-predicted-motion-actuation-projection-v1'
    predictive_motion_readout: str = 'per-defender predicted capture, crossing, radial and collision geometry with retained-relative actual thrust'


def motion_features(batch, paths, cfg):
    """Return [root, candidate, defender, 20] from predicted paths only."""
    relative = paths[:, :, :, :-1] - paths[:, :, :, -1:]
    known = batch['known_time_valid'][:, None, None, None]
    distance = relative.norm(dim=-1)
    closest = distance.masked_fill(~known, torch.inf).argmin(-1)
    index = closest[..., None, None].expand(*closest.shape, 1, 2)
    nearest = relative.gather(-2, index).squeeze(-2)
    current = (batch['positions'][:, :-1] - batch['positions'][:, -1:])[:, None, None, :, None]
    velocity = torch.cat((relative[..., :1, :] - current,
        relative[..., 1:, :] - relative[..., :-1, :]), -2) / cfg.dt
    nearest_velocity = velocity.gather(-2, index).squeeze(-2)
    sine, cosine = batch['history'][:, :-1, -1, 4], batch['history'][:, :-1, -1, 5]
    forward = torch.stack((cosine, sine), -1)[:, None, None]
    lateral = torch.stack((-sine, cosine), -1)[:, None, None]
    radius = paths.norm(dim=-1)
    radial = radius[:, :, :, :-1] - radius[:, :, :, -1:]
    closest_radial = radial.gather(-1, closest[..., None]).squeeze(-1)
    worst_radial = radial.masked_fill(~known, -torch.inf).amax(-1)

    defenders = paths[:, :, :, :-1]
    pair_distance = (defenders.unsqueeze(4) - defenders.unsqueeze(3)).norm(dim=-1)
    valid = batch['entity_valid'][:, :-1]
    count = valid.shape[-1]
    pair_valid = (valid[:, None, None, :, None, None]
        & valid[:, None, None, None, :, None]
        & ~torch.eye(count, dtype=torch.bool, device=paths.device)[None, None, None, :, :, None]
        & batch['known_time_valid'][:, None, None, None, None])
    pair_minimum = pair_distance.masked_fill(~pair_valid, torch.inf).amin((-1, -2))
    pair_minimum = torch.where(torch.isfinite(pair_minimum), pair_minimum,
                               torch.full_like(pair_minimum, 30.))
    initial_relative = current.squeeze(-2)
    initial_closing = -(initial_relative * velocity[..., 0, :]).sum(-1) / initial_relative.norm(dim=-1).clamp_min(.1)
    per_mode = torch.stack(((nearest.norm(dim=-1) - 5.) / 5.,
        (closest + 1).to(paths.dtype) / cfg.horizon,
        (nearest * forward).sum(-1) / 10., (nearest * lateral).sum(-1) / 10.,
        (nearest_velocity * forward).sum(-1) / 5.,
        (nearest_velocity * lateral).sum(-1) / 5.,
        closest_radial / 20., (pair_minimum - 5.) / 5.,
        worst_radial / 20., initial_closing / 5.), -1)
    per_mode = per_mode.masked_fill(~valid[:, None, None, :, None], 0.)
    # Mode spread is an input descriptor, not a calibrated confidence estimate.
    return torch.cat((per_mode.mean(2), per_mode.std(2, unbiased=False)), -1)


class PredictedMotionModel(actuation.ActuationModel):
    def __init__(self, cfg=PredictedMotionConfig()):
        super().__init__(cfg)
        self.actor.motion_net = nn.Sequential(nn.LayerNorm(3 * cfg.width + 64),
            nn.Linear(3 * cfg.width + 64, cfg.width), nn.GELU(), nn.Linear(cfg.width, 2))
        nn.init.zeros_(self.actor.motion_net[-1].weight)
        nn.init.zeros_(self.actor.motion_net[-1].bias)
        self._motion_context = None

    def encode_candidates(self, batch, nominal, candidates, history):
        encoded = super().encode_candidates(batch, nominal, candidates, history)
        if self.actor.context is not None:
            self._motion_context = self.actor.context
        return encoded

    def forward(self, batch, history_cache=None):
        self._motion_context = None
        try:
            output = super().forward(batch, history_cache)
            if 'teacher_logits' not in batch:
                return output
            geometry = motion_features(batch, output['xy'], self.cfg)
            retained = batch['teacher_logits'].argmax(-1)
            rows = torch.arange(geometry.shape[0], device=geometry.device)
            reference = geometry[rows, retained][:, None].expand_as(geometry)
            features, difference, valid = self._motion_context
            features = torch.cat((features, geometry,
                (geometry - reference) / self.cfg.policy_reference_feature_scale), -1)
            sensitivity = self.actor.motion_net(features)
            correction = (sensitivity * difference * valid).sum((-2, -1))
            output['logits'] = output['logits'] + self.cfg.policy_logit_scale * correction
            output['predicted_motion_action_correction'] = correction
            return output
        finally:
            self._motion_context = None


def initialize(project, source, config):
    contact.ContactEventModel = PredictedMotionModel
    training.ModelConfig = PredictedMotionConfig
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
    cfg = PredictedMotionConfig(**dict(value['config'], revision=PredictedMotionConfig().revision,
        unobserved_promotion_weight=0.,
        policy_projection_temperature=PredictedMotionConfig().policy_projection_temperature,
        policy_update=PredictedMotionConfig().policy_update,
        predictive_motion_readout=PredictedMotionConfig().predictive_motion_readout))
    model = PredictedMotionModel(cfg)
    loaded = model.load_state_dict(value['state'], strict=False)
    if loaded.unexpected_keys or any(not k.startswith('actor.motion_net.') for k in loaded.missing_keys):
        raise ValueError('Source must contain the complete inherited actuation model')
    protocol = dict(value['protocol'], predicted_motion_initialization=dict(
        source=args.pretrained, seed=args.seed, additional_pretraining=0,
        inherited='all existing response and actuation parameters',
        added='zero-output per-defender readout of predicted motion geometry and retained-relative differences',
        geometry='closest distance/time, body-frame relative position/velocity, radial margins, closest teammate distance and initial closing speed; mode mean/spread',
        information='predicted short trajectories, current observations and known remaining time only',
        policy='unchanged finite observed-action probability target and full-bank KL projection',
        scope='representation plus added readout capacity; no isolated attribution before matched evidence',
        parameter_count=sum(p.numel() for p in model.parameters())))
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
    contact.ContactEventModel = PredictedMotionModel
    if mode == 'train':
        sys.argv.pop(1)
        out = Path(sys.argv[sys.argv.index('--out') + 1])
        projection.projection_diagnostics.clear()
        training.ModelConfig = PredictedMotionConfig
        training.COMPARISON_PROPOSALS = ('teacher', 'leading_alternative', 'sampled_alternative')
        training.initialize = initialize
        training.rollout = projection.rollout
        training.event_examples = projection.event_examples
        training.ACTOR_OBJECTIVE = projection.projection_objective
        training.PROTOCOL_OVERRIDES = dict(
            objective='same finite observed-action policy improvement and full-bank KL projection',
            policy_target='q_S=m*softmax(log pi_old_S+mean_actual_return/0.5); q_unqueried=pi_old_unqueried; KL(q||pi_theta)/m',
            motion_readout=PredictedMotionConfig().predictive_motion_readout,
            geometry_scope='short predicted trajectories and current observations; mode spread is not calibrated confidence',
            zero_reference='both inherited and new readouts multiply the actual thrust difference from the retained proposal',
            unchanged='response initialization,teacher,all deployment candidates,1.5margin,three actions and two common futures,physical replay,original64selection',
            budget='same320root schedule,80updates,1920recorded complete episodes and477optimizer batches; extra readout increases parameters and compute',
            attribution='representation and added readout capacity change together; no independent mechanism conclusion from a single main run',
            claim_boundary='no CV,critic,terminal outcome head,future observation input or guarantee of policy improvement')
        result = training.train_main()
        if projection.projection_diagnostics:
            records = projection.projection_diagnostics
            path = out / 'protocol.json'
            protocol = json.loads(path.read_text())
            protocol['policy_projection_optimization'] = dict(
                calls=len(records), mean_loss=float(np.mean([x['loss'] for x in records])),
                mean_target_l1=float(np.mean([x['target_l1_mean'] for x in records])),
                target_greedy_changes=sum(x['target_greedy_changes'] for x in records),
                repeated_root_count=sum(x['roots'] for x in records),
                scope='optimizer-call diagnostics in this process segment; reused roots are not independent')
            path.write_text(json.dumps(protocol, indent=2) + '\n')
        return result
    common.CommonFutureConfig = PredictedMotionConfig
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    original = evaluation.runtime_identity

    def identity(*args, **kwargs):
        value = original(*args, **kwargs)
        value['runtime_code'] = canonical_hash(dict(core=value['runtime_code'],
            predicted_motion=file_hash(__file__), actuation=file_hash(actuation.__file__)))
        return value

    evaluation.runtime_identity = identity
    return common.main()


if __name__ == '__main__':
    main()
