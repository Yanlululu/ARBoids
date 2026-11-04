"""Candidate-interaction response learning under common short physical recourse.

The physical reference is explicit and independent of the candidate set.  Each
candidate is processed separately; attention is over the team, never over the
candidate bank.  Joint future modes are matched once per root and reused for
all candidate differences.  Complete-task returns are not prediction targets.
"""
import argparse
from dataclasses import asdict, dataclass
import datetime
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from citf.bridge import Controller, inputs
from citf.model import Attention
from citf.risk import bounded_mean, signed_log1p
from citf.train import atomic_save
from experiments import synchronized_configuration_policy as synchronized
from experiments.response_data import candidate_bank


@dataclass(frozen=True)
class ResponseConfig:
    revision: str = 'CITF-fixed-reference-recourse-response-v1'
    width: int = 64
    heads: int = 4
    history: int = 8
    horizon: int = 16
    modes: int = 4
    dt: float = .2
    clearance: float = 7.
    objective_risk_weight: float = 64.
    guard_weight: float = 16.
    guard_margin: float = 6.
    candidate_context: bool = True
    supervision: str = 'response'
    response_hold: int = 1
    response_reference: str = 'strong synchronized physical choice then common physical recourse'
    label_semantics: str = 'joint positions through first event; absorbing terminal endpoint'
    policy_prior_scale: float = 2.
    policy_logit_scale: float = 2.
    policy_reference_feature_scale: float = .1
    policy_prior_mode: str = 'signed_physical'
    deployment: str = 'geometry'
    geometry_blend: float = .5
    geometry_kind: str = 'guard'
    candidate_domain: str = 'physical_bank_plus_joint_pairs'


def add_reference(batch):
    return dict(batch, reference_candidate=batch['candidate'][:, 0],
                reference_nominal=batch['nominal'][:, 0], reference_unary=batch['unary'][:, 0])


def geometric_curves(paths, entity_valid=None):
    """[B,C,K,E,T,2] -> synchronized radial/capture/pair trajectories."""
    radius = paths.norm(dim=-1)
    radial = radius[..., :-1, :] - radius[..., -1:, :]
    capture = (paths[..., :-1, :, :] - paths[..., -1:, :, :]).norm(dim=-1)
    if entity_valid is not None:
        defenders = entity_valid[:, None, None, :-1, None]
        radial = radial.masked_fill(~defenders, torch.inf)
        capture = capture.masked_fill(~defenders, torch.inf)
    radial, capture = radial.amin(-2), capture.amin(-2)
    i, j = torch.triu_indices(paths.shape[-3] - 1, paths.shape[-3] - 1, 1, device=paths.device)
    pairs = (paths[..., i, :, :] - paths[..., j, :, :]).norm(dim=-1)
    return radial, capture, pairs, radius[..., -1, :]


def trajectory_scores(paths, known, unary, cfg, kind=None, entity_valid=None):
    radial, capture, pairs, attacker = geometric_curves(paths, entity_valid)
    if entity_valid is not None:
        i, j = torch.triu_indices(paths.shape[-3] - 1, paths.shape[-3] - 1, 1, device=paths.device)
        pairs = pairs.masked_fill(~(entity_valid[:, i] & entity_valid[:, j])[:, None, None, :, None], torch.inf)
    length = paths.shape[-2]
    ticks = torch.arange(length, device=paths.device)
    valid = known[:, None, None]
    loss = ((radial > 0) | (pairs.amin(-2) < 5) | (attacker < 15)) & valid
    caught = (capture < 5) & valid
    first_loss = torch.where(loss, ticks, length + 1).amin(-1)
    first_capture = torch.where(caught, ticks, length + 1).amin(-1)
    observed = valid & (ticks <= torch.minimum(first_loss, first_capture)[..., None])
    gap = radial.masked_fill(~observed, -torch.inf).amax(-1)
    closest = capture.masked_fill(~observed, torch.inf).amin(-1)
    minimum = pairs.masked_fill(~observed[..., None, :], torch.inf).amin(-1)
    collision = (cfg.clearance - minimum).clamp_min(0).square().sum(-1)
    guard = unary[:, :, None] + cfg.objective_risk_weight * collision + cfg.guard_weight * (gap + cfg.guard_margin).clamp_min(0).square()
    lose = ((first_loss <= first_capture) & (first_loss <= length)).to(paths.dtype)
    win = ((first_capture < first_loss) & (first_capture <= length)).to(paths.dtype)
    event = 1000 * (lose - win)
    if (kind or cfg.geometry_kind) == 'ordered':
        cost = event + .01 * torch.tanh(guard / 100)
    elif (kind or cfg.geometry_kind) == 'guard':
        cost = guard
    else:
        raise ValueError('Unknown short physical score')
    return dict(cost=cost, guard=guard, gap=gap, closest=closest, collision=collision,
                short_loss=lose, short_capture=win)


def stopped_positions(batch):
    actual = batch['real_positions']
    count = batch['time_valid'].sum(-1).clamp_min(1)
    last = (count - 1)[..., None, None, None].expand(*actual.shape[:3], actual.shape[3], 1, 2)
    endpoint = actual.gather(-2, last)
    return torch.where(batch['time_valid'][..., None, :, None], actual, endpoint)


class CandidateResponseModel(nn.Module):
    def __init__(self, cfg=ResponseConfig()):
        super().__init__(); self.cfg = cfg
        width = cfg.width
        self.history_embed = nn.Linear(8, width)
        self.temporal = nn.GRU(width, width, batch_first=True)
        self.history_context = Attention(width, cfg.heads)
        self.time_embed = nn.Sequential(nn.Linear(2, width), nn.GELU(), nn.Linear(width, width))
        self.condition = nn.Sequential(nn.Linear(24 + 4 * cfg.horizon, width), nn.GELU(), nn.Linear(width, width))
        self.interaction = nn.ModuleList([Attention(width, cfg.heads, True), Attention(width, cfg.heads, True)])
        self.absolute = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, cfg.modes * cfg.horizon * 2))
        self.response = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, cfg.modes * cfg.horizon * 2))
        self.actor = nn.Sequential(nn.LayerNorm(4 * width + 16), nn.Linear(4 * width + 16, width), nn.GELU(), nn.Linear(width, 1))
        for head in [self.absolute[-1], self.response[-1], self.actor[-1]]:
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
        # Distinct initial future directions break mode symmetry without entity IDs.
        with torch.no_grad():
            offsets = torch.linspace(-.02, .02, cfg.modes)[:, None, None].expand(cfg.modes, cfg.horizon, 2)
            self.absolute[-1].bias.copy_(offsets.flatten())

    def encode_history(self, batch):
        history = batch['history']; roots, entities, length = history.shape[:3]
        mask = batch['history_valid']
        encoded = self.history_embed(history) * mask[..., None]
        encoded = self.temporal(encoded.reshape(roots * entities, length, -1))[0][:, -1]
        encoded = encoded.reshape(roots, entities, -1)
        return self.history_context(encoded, batch['entity_valid']) + self.time_embed(batch['time_context'])[:, None]

    def encode_candidates(self, batch, nominal, candidates, history):
        roots, count, entities, horizon = nominal.shape[:4]
        reference_c = batch['reference_candidate'][:, None].expand(roots, count, entities, 8)
        reference_n = batch['reference_nominal'][:, None].expand_as(nominal)
        known = batch['known_time_valid'][:, None, None, :, None]
        displacement = (nominal - batch['positions'][:, None, :, None]) / 10 * known
        reference_d = (reference_n - batch['positions'][:, None, :, None]) / 10 * known
        feature = torch.cat((candidates, reference_c, candidates - reference_c,
                             displacement.flatten(-2), reference_d.flatten(-2)), -1)
        z = (history[:, None] + self.condition(feature)).reshape(roots * count, entities, -1)
        valid = batch['entity_valid'][:, None].expand(roots, count, entities).reshape(roots * count, entities)
        if self.cfg.candidate_context:
            paths = nominal.reshape(roots * count, entities, horizon, 2)
            pos = batch['positions'][:, None].expand(roots, count, entities, 2).reshape(roots * count, entities, 2)
            distance = (paths[:, :, None] - paths[:, None]).norm(dim=-1)
            times = batch['known_time_valid'][:, None].expand(roots, count, horizon).reshape(roots * count, horizon)
            minimum, when = distance.masked_fill(~times[:, None, None], torch.inf).min(-1)
            last = (times.sum(-1) - 1)[:, None, None, None].expand(roots * count, entities, entities, 1)
            ending = distance.gather(-1, last).squeeze(-1)
            delta = pos[:, None] - pos[:, :, None]
            edge = torch.cat((delta / 20, minimum[..., None] / 10,
                              ((ending - delta.norm(dim=-1)) / 10)[..., None],
                              (when / horizon)[..., None]), -1)
            for layer in self.interaction:
                z = layer(z, valid, edge)
        return z.reshape(roots, count, entities, -1)

    def forward(self, batch, history_cache=None):
        cfg = self.cfg
        nominal = batch['nominal']; roots, candidates, entities, horizon = nominal.shape[:4]
        history = self.encode_history(batch) if history_cache is None else history_cache
        ref_z = self.encode_candidates(batch, batch['reference_nominal'][:, None], batch['reference_candidate'][:, None], history)
        z = self.encode_candidates(batch, nominal, batch['candidate'], history)
        shape = (roots, candidates, entities, cfg.modes, horizon, 2)
        raw = self.response(z).reshape(shape)
        ref_raw = self.response(ref_z).reshape(roots, 1, entities, cfg.modes, horizon, 2)
        absolute = self.absolute(ref_z).reshape(roots, 1, entities, cfg.modes, horizon, 2)
        ticks = torch.arange(1, horizon + 1, device=nominal.device, dtype=nominal.dtype)
        times = ticks * cfg.dt
        envelope = times.square()[None, None, None, None, :, None].expand(roots, candidates, entities, cfg.modes, horizon, 2).clone()
        # The attacker acts before defender movement; its first transition is
        # identical across candidates under common noise. Absolute first-step
        # motion remains learned, including unobserved currents for all boats.
        envelope[:, :, -1, :, 0] = 0
        abs_envelope = times[None, None, None, None, :, None]
        reference = batch['reference_nominal'][:, None, :, None] + absolute * abs_envelope
        first_difference = nominal[..., :1, :] - batch['reference_nominal'][:, None, :, :1]
        impulse = first_difference[:, :, :, None] * (2 * ticks - 1)[None, None, None, None, :, None]
        paths = (reference + impulse + (raw - ref_raw) * envelope).permute(0, 1, 3, 2, 4, 5)
        geometry = trajectory_scores(paths, batch['known_time_valid'], batch['unary'], cfg,
                                     entity_valid=batch['entity_valid'])
        physical, collision, gap = synchronized.physical_components(batch, cfg)
        ref_batch = dict(batch, nominal=batch['reference_nominal'][:, None], unary=batch['reference_unary'][:, None])
        physical_reference = synchronized.physical_components(ref_batch, cfg)[0]
        ref_paths = reference.permute(0, 1, 3, 2, 4, 5)
        reference_geometry = trajectory_scores(ref_paths, batch['known_time_valid'], batch['reference_unary'][:, None], cfg,
                                               entity_valid=batch['entity_valid'])
        reference_cost = reference_geometry['cost'].mean(-1)
        expected = geometry['cost'].mean(-1)
        corrected = physical + cfg.geometry_blend * (expected - reference_cost - physical + physical_reference)
        mask = batch['entity_valid'][:, None, :, None]
        pooled_mean = (z * mask).sum(-2) / mask.sum(-2).clamp_min(1)
        pooled_max = z.masked_fill(~mask, -torch.inf).amax(-2)
        slog = signed_log1p
        scalars = torch.stack((batch['unary'] / 10, torch.log1p(collision), slog(gap / 10),
            slog(geometry['gap'].mean(-1) / 10), torch.log1p(geometry['collision'].mean(-1)),
            geometry['closest'].mean(-1) / 20, geometry['gap'].std(-1, unbiased=False) / 5,
            batch['time_context'][:, :1].expand(roots, candidates)), -1)
        features = torch.cat((pooled_mean, pooled_max, scalars), -1)
        ref_mean = (ref_z * mask).sum(-2) / mask.sum(-2).clamp_min(1)
        ref_max = ref_z.masked_fill(~mask, -torch.inf).amax(-2)
        _, ref_collision, ref_gap = synchronized.physical_components(ref_batch, cfg)
        ref_scalars = torch.stack((batch['reference_unary'][:, None] / 10,
            torch.log1p(ref_collision), slog(ref_gap / 10),
            slog(reference_geometry['gap'].mean(-1) / 10), torch.log1p(reference_geometry['collision'].mean(-1)),
            reference_geometry['closest'].mean(-1) / 20, reference_geometry['gap'].std(-1, unbiased=False) / 5,
            batch['time_context'][:, :1]), -1)
        reference_features = torch.cat((ref_mean, ref_max, ref_scalars), -1)
        action_features = torch.cat((features, (features - reference_features) / cfg.policy_reference_feature_scale), -1)
        physical_difference = physical - physical_reference
        if cfg.policy_prior_mode == 'reference_saturated':
            # The fixed strong controller remains the zero-actor decision. A
            # lower nominal score from an expanded proposal alone is not
            # evidence of a better complete-task action. Learning can still
            # prefer any proposal; there is no bound on the learned residual.
            physical_difference = physical_difference.clamp_min(0)
        elif cfg.policy_prior_mode != 'signed_physical':
            raise ValueError('Unknown fixed-reference policy prior')
        baseline = -torch.asinh(physical_difference / cfg.policy_prior_scale)
        logits = baseline + cfg.policy_logit_scale * self.actor(action_features).squeeze(-1)
        return dict(xy=paths, geometry=geometry, physical=physical.masked_fill(~batch['candidate_valid'], torch.inf),
                    physical_reference=physical_reference, score=corrected.masked_fill(~batch['candidate_valid'], torch.inf),
                    features=action_features, baseline=baseline,
                    logits=logits.masked_fill(~batch['candidate_valid'], -torch.inf))


def response_objective(output, batch, cfg):
    actual = stopped_positions(batch)[:, :, :cfg.modes]
    prediction = output['xy']
    modes = prediction.shape[2]
    permutations = torch.tensor(list(itertools.permutations(range(modes))), device=prediction.device)
    # A single matching per root based on the reference trajectory. Candidate
    # alternatives cannot select a different future to hide response error.
    with torch.no_grad():
        ref_pred = prediction[:, 0]
        ref_actual = actual[:, 0]
        error = F.smooth_l1_loss(ref_pred[:, :, None].expand(-1, -1, modes, -1, -1, -1),
                                ref_actual[:, None].expand(-1, modes, -1, -1, -1, -1), reduction='none')
        match_valid = (batch['entity_valid'][:, None, None, :, None, None]
                       & batch['known_time_valid'][:, None, None, None, :, None]).expand_as(error)
        cost = error.masked_fill(~match_valid, 0).sum((-1, -2, -3)) / match_valid.sum((-1, -2, -3)).clamp_min(1)
        perm_cost = cost[:, torch.arange(modes, device=cost.device)[None], permutations].sum(-1)
        matching = permutations[perm_cost.argmin(-1)]
    index = matching[:, None, :, None, None, None].expand_as(actual)
    truth = actual.gather(2, index)
    valid = (batch['candidate_valid'][:, :, None, None, None, None]
             & batch['entity_valid'][:, None, None, :, None, None]
             & batch['known_time_valid'][:, None, None, None, :, None])
    absolute = F.smooth_l1_loss((prediction - truth) / 3, torch.zeros_like(prediction), reduction='none')
    absolute_loss = bounded_mean(absolute, valid)
    response = (prediction - prediction[:, :1]) - (truth - truth[:, :1])
    response_loss = F.smooth_l1_loss(response / .25, torch.zeros_like(response), reduction='none')
    response_loss = bounded_mean(response_loss, valid)
    pr, pc, pp, pa = geometric_curves(prediction, batch['entity_valid'])
    tr, tc, tp, ta = geometric_curves(truth, batch['entity_valid'])
    i, j = torch.triu_indices(prediction.shape[-3] - 1, prediction.shape[-3] - 1, 1, device=prediction.device)
    pair_valid = batch['entity_valid'][:, i] & batch['entity_valid'][:, j]
    def curve_loss(a, b):
        weight = batch['candidate_valid'][:, :, None, None] & batch['known_time_valid'][:, None, None, :]
        if a.ndim == 5:
            weight = weight[..., None, :] & pair_valid[:, None, None, :, None]
        error = F.smooth_l1_loss(a, b, reduction='none')
        return bounded_mean(error, weight)
    geometric_loss = curve_loss(pr / 2, tr / 2) + curve_loss(pc / 2, tc / 2) + curve_loss(pp / 2, tp / 2)
    difference_loss = curve_loss((pr - pr[:, :1]) / .25, (tr - tr[:, :1]) / .25)
    difference_loss += curve_loss((pc - pc[:, :1]) / .25, (tc - tc[:, :1]) / .25)
    # Soft first physical events supply gradients in their relevant neighborhood.
    def soft_events(radial, capture, pair, attacker):
        pair = pair.masked_fill(~pair_valid[:, None, None, :, None], torch.inf)
        fail = 1 - ((1 - torch.sigmoid(radial / .5))
                    * (1 - torch.sigmoid((5 - pair.amin(-2)) / .5))
                    * (1 - torch.sigmoid((15 - attacker) / .5)))
        caught = torch.sigmoid((5 - capture) / .5)
        time_mask = batch['known_time_valid'][:, None, None]
        fail = fail * time_mask; caught = caught * time_mask
        survive = (1 - fail) * (1 - caught)
        prefix = torch.cat((torch.ones_like(survive[..., :1]), survive[..., :-1]), -1).cumprod(-1)
        return torch.stack(((prefix * fail).sum(-1), (prefix * (1 - fail) * caught).sum(-1)), -1)
    event_error = F.mse_loss(soft_events(pr, pc, pp, pa), soft_events(tr, tc, tp, ta), reduction='none')
    event_loss = bounded_mean(event_error, batch['candidate_valid'][:, :, None, None])
    total = absolute_loss
    if cfg.supervision == 'response':
        total = total + response_loss + .2 * geometric_loss + .5 * difference_loss + event_loss
    elif cfg.supervision != 'ordinary':
        raise ValueError('Unknown response learning control')
    return dict(total=total, absolute=absolute_loss, response=response_loss,
                geometry=geometric_loss, difference=difference_loss, event=event_loss,
                response_mae=bounded_mean(response.abs(), valid), position_mae=bounded_mean((prediction - truth).abs(), valid))


def load_response_data(path, role):
    path = Path(path)
    status = json.loads((path / 'status.json').read_text())
    if not status['completed']:
        raise ValueError('Use only a completed response collection')
    items, roots = [], []
    for file in sorted(path.glob('*.pt')):
        value = torch.load(file, map_location='cpu', weights_only=False)
        if value['metadata']['split'] != role:
            raise ValueError('Response split mismatch')
        items.append(value['tensors']); roots.append(value['metadata'])
    names = ['history', 'positions', 'history_valid', 'entity_valid', 'nominal', 'candidate',
             'unary', 'candidate_valid', 'time_context', 'known_time_valid', 'real_positions', 'time_valid']
    data = {name: torch.stack([x[name] for x in items]) for name in names}
    return add_reference(data), roots


@torch.no_grad()
def response_validation(model, data, device, batch_size=8):
    sums = {}; count = len(data['history']); regret = 0.; physical_regret = 0.; changed = 0
    for start in range(0, count, batch_size):
        b = {k: v[start:start + batch_size].to(device) for k, v in data.items()}
        out = model(b); loss = response_objective(out, b, model.cfg)
        size = len(b['history'])
        for key, value in loss.items():
            sums[key] = sums.get(key, 0.) + float(value) * size
        actual = stopped_positions(b)
        actual_cost = trajectory_scores(actual, b['known_time_valid'], b['unary'], model.cfg,
                                        entity_valid=b['entity_valid'])['cost'].mean(-1)
        actual_cost = actual_cost.masked_fill(~b['candidate_valid'], torch.inf)
        selected = out['score'].argmin(-1); best = actual_cost.amin(-1)
        regret += float((actual_cost.gather(1, selected[:, None]).squeeze(-1) - best).sum())
        physical = out['physical'].argmin(-1)
        physical_regret += float((actual_cost.gather(1, physical[:, None]).squeeze(-1) - best).sum())
        changed += int((selected != physical).sum())
    return dict(N=count, **{k: v / count for k, v in sums.items()},
                candidate_regret=regret / count, physical_candidate_regret=physical_regret / count, changed=changed)


def train_main():
    parser = argparse.ArgumentParser()
    for key in ['train', 'validation', 'out', 'deadline']:
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--seed', type=int, default=8101)
    parser.add_argument('--steps', type=int, default=2400)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--validation-every', type=int, default=400)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--no-context', action='store_true')
    parser.add_argument('--ordinary', action='store_true')
    parser.add_argument('--deferred-validation', action='store_true')
    args = parser.parse_args(); torch.set_num_threads(1)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    train, train_roots = load_response_data(args.train, 'train')
    validation, validation_roots = None, []
    def load_validation():
        nonlocal validation, validation_roots
        state_path = Path(args.validation) / 'status.json'
        if not state_path.exists() or not json.loads(state_path.read_text())['completed']:
            return False
        validation, validation_roots = load_response_data(args.validation, 'validation')
        if {x['geometry_family_id'] for x in train_roots} & {x['geometry_family_id'] for x in validation_roots}:
            raise ValueError('Train and validation physical layouts overlap')
        return True
    if not args.deferred_validation and not load_validation():
        raise ValueError('Validation data are incomplete; use the registered deferred selection procedure')
    cfg = ResponseConfig(candidate_context=not args.no_context, supervision='ordinary' if args.ordinary else 'response')
    model = CandidateResponseModel(cfg).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    deadline = datetime.datetime.fromisoformat(args.deadline).timestamp()
    first = train_roots[0]
    protocol = dict(arguments=vars(args), source_sha256=first['source_sha256'],
        source_implementation_sha256=first['source_implementation_sha256'], environment_sha256=first['environment_sha256'],
        training_roots=len(train_roots), validation_roots=len(validation_roots) or None,
        model_selection='validation absolute loss for ordinary control; response loss for full/no-context; closed loop validation remains required',
        complete_task_returns_used=False)
    best = None; step = 0
    def status(phase, completed=False):
        (out / 'status.json').write_text(json.dumps(dict(completed=completed, step=step,
            total_steps=args.steps, phase=phase, best=best,
            updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2))
    def save(path):
        atomic_save(dict(config=asdict(cfg), state={k: v.detach().cpu() for k, v in model.state_dict().items()},
                         protocol=protocol, step=step, best=best, seed=args.seed), path)
    def validate():
        nonlocal best
        stage_path = out / f'stage_{step:06d}.pt'
        if validation is None:
            if not stage_path.exists():
                save(stage_path)
            save(out / 'last.pt')
            status('training; complete validation collection pending')
            return
        model.eval(); metrics = response_validation(model, validation, args.device, args.batch_size)
        key = metrics['absolute'] if args.ordinary else metrics['response'] + .5 * metrics['difference']
        if best is None or key < best['selection_loss']:
            best = dict(step=step, selection_loss=key, metrics=metrics); save(out / 'best.pt')
        if not stage_path.exists():
            save(stage_path)
        save(out / 'last.pt')
        with (out / 'learning.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(step=step, validation=metrics, best_step=best['step'])) + '\n')
        print(json.dumps(dict(step=step, validation=metrics, best_step=best['step'])), flush=True)
        status('training with complete validation')
        model.train()
    validate()
    recent = []
    while step < args.steps and time.time() < deadline:
        indices = rng.integers(0, len(train['history']), args.batch_size)
        batch = {k: v[indices].to(args.device) for k, v in train.items()}
        loss = response_objective(model(batch), batch, cfg)
        optimizer.zero_grad(set_to_none=True); loss['total'].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.); optimizer.step()
        step += 1; recent.append(float(loss['total'].detach()))
        if step % 100 == 0:
            print(json.dumps(dict(step=step, training_loss=float(np.mean(recent)))), flush=True); recent.clear()
            status('training')
        if step % args.validation_every == 0 or step == args.steps:
            validate()
    save(out / 'last.pt')
    training_step = step
    final_stage = out / f'stage_{step:06d}.pt'
    if not final_stage.exists():
        save(final_stage)
    if validation is None:
        status('waiting for the complete registered validation collection')
        while time.time() < deadline and not load_validation():
            time.sleep(10)
        if validation is not None:
            protocol['validation_roots'] = len(validation_roots)
            for path in sorted(out.glob('stage_*.pt')):
                value = torch.load(path, map_location='cpu', weights_only=False)
                step = value['step']; model.load_state_dict(value['state']); validate()
    step = training_step
    status('finished' if best is not None else 'validation incomplete', step == args.steps and best is not None)


class ResponseController(Controller):
    @torch.no_grad()
    def act(self, world):
        started = time.perf_counter()
        if self.model is None:
            raise ValueError('A registered response checkpoint is required')
        cfg = self.model.cfg
        f, plans, detail = candidate_bank(world, self.source, cfg)
        if cfg.candidate_domain == 'original_physical_bank':
            plans = plans[:detail['original_count']]
        b, gates, _ = inputs(world, f, plans)
        b = add_reference(b); b = {k: v.to(self.device) for k, v in b.items()}
        output = self.model(b)
        values = (-output['logits'][0] if cfg.deployment == 'policy' else output['score'][0]).cpu().tolist()
        choice = min(range(len(plans)), key=lambda k: (values[k],
            int(k != 0) if cfg.deployment == 'policy' and cfg.policy_prior_mode == 'reference_saturated' else 0,
            tuple(plans[k])))
        gate = gates[choice]
        force = (750 * (gate * f['learned'] + (1 - gate) * f['rules']) + 250).numpy()
        trace = dict(evaluated_unique=len(plans), configurations=plans.tolist(), scores=values,
            selected_indices=plans[choice].tolist(),
            fixed_reference=True, candidate_domain=cfg.candidate_domain, current_action_only=True,
            recourse_condition=cfg.response_reference, gates=gate.tolist(), physical_thrust=force.tolist(),
            physical_scores=output['physical'][0].cpu().tolist(),
            compute_seconds=time.perf_counter() - started, **detail)
        return force, trace


def load_model(path, source, world, device):
    value = torch.load(path, map_location='cpu', weights_only=False)
    if value['protocol']['source_sha256'] != source.sha256 or value['protocol']['environment_sha256'] != world.environment_sha256:
        raise ValueError('Response checkpoint source/environment mismatch')
    model = CandidateResponseModel(ResponseConfig(**value['config']))
    model.load_state_dict(value['state']); return model.to(device).eval()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'train':
        sys.argv.pop(1); train_main(); return
    import citf.evaluate as evaluation
    from citf.contracts import canonical_hash, file_hash
    evaluation.load_behavior = load_model
    evaluation.Controller = ResponseController
    previous = evaluation.runtime_identity
    def identity(*args, **kwargs):
        result = previous(*args, **kwargs)
        checkpoint = args[2] if len(args) > 2 else kwargs['checkpoint']
        config = torch.load(checkpoint, map_location='cpu', weights_only=False)['config']
        result['method'] = 'CITF-fixed-reference-recourse-response-development'
        result['controller'].update(config)
        result['runtime_code'] = canonical_hash(dict(core=result['runtime_code'],
            candidate_response=file_hash(__file__), candidate_bank=file_hash(Path(__file__).with_name('response_data.py'))))
        return result
    evaluation.runtime_identity = identity
    evaluation.main()


if __name__ == '__main__':
    main()
