"""Select saved stages using admissible short-geometry regret on fixed labels.

Only the already declared validation roots are read. Selection neither reads
closed-loop outcomes nor changes the data or parameters of a checkpoint.
"""
import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch

from citf.train import load_shards
from experiments.guard_prediction import physical_pair_cost
from experiments.synchronized_guard import actual_team, nominal_team
from experiments.synchronized_delta import PreciseTeamConfig, PreciseTeamPredictor


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models-dir', required=True)
    parser.add_argument('--validation', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[7101, 7102, 7103])
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--nominal-slack', type=float, default=1.)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise ValueError('Refuse to replace a declared checkpoint selection')
    torch.set_num_threads(1)
    data, _, _ = load_shards(args.validation, 'validation')
    physical = data['unary'] + 64. * physical_pair_cost(data, 7.)
    physical += 16. * (nominal_team(data)[1][:, :, 0] + 6.).clamp_min(0).square()
    physical = physical.masked_fill(~data['candidate_valid'], torch.inf)
    feasible = data['candidate_valid'] & (physical <= physical.amin(1, keepdim=True) + args.nominal_slack + 1e-10)
    radius = data['positions'].norm(dim=-1)
    current_gap = (radius[:, :-1] - radius[:, -1:]).masked_fill(~data['entity_valid'][:, :-1], torch.inf).amin(-1)
    active = (current_gap >= -16.) & (feasible.sum(1) > 1)
    truth = actual_team(data)[:, :, :, 0].mean(2)
    optimum = truth.masked_fill(~feasible, torch.inf).amin(1)
    report = dict(role='fixed-source development label validation', arguments=vars(args),
        criterion='minimum mean stopped-team-gap regret among physically admissible candidates on active roots; earliest update on ties',
        activation='observed minimum radial gap >= -16m and more than one feasible candidate',
        fixed_stages=[0, 4000, 8000, 16000, 'label-selected best'],
        total_roots=len(active), active_roots=int(active.sum()),
        used_closed_loop_results=False, selected_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), models=[])
    if not active.any():
        raise ValueError('No active fixed validation roots for this selection rule')
    indices = torch.where(active)[0]
    for seed in args.seeds:
        directory = Path(args.models_dir) / ('precise_boundary128_width48_nll1_s%d' % seed)
        best_update = json.loads((directory / 'status.json').read_text())['best']['update']
        candidates = [(update, directory / 'snapshots' / ('update_%06d.pt' % update)) for update in [0, 4000, 8000, 16000]]
        if best_update not in [0, 4000, 8000, 16000]:
            candidates.append((best_update, directory / 'best.pt'))
        rows = []
        for update, path in candidates:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            model = PreciseTeamPredictor(PreciseTeamConfig(**checkpoint['config'])).to(args.device).eval()
            model.load_state_dict(checkpoint['state'], strict=True)
            values = []
            for start in range(0, len(indices), 8):
                subset = indices[start:start + 8]
                batch = {key: value[subset].to(args.device) for key, value in data.items()}
                means = model(batch)['guard_mu'][:, :, 0].cpu()
                for local, root_index in enumerate(subset.tolist()):
                    eligible = torch.where(feasible[root_index])[0].tolist()
                    choice = min(eligible, key=lambda candidate: (float(means[local, candidate]),
                        float(physical[root_index, candidate]),
                        tuple(data['candidate'][root_index, candidate, :-1, :2].flatten().tolist())))
                    values.append(float(truth[root_index, choice] - optimum[root_index]))
            row = dict(checkpoint=str(path), update=update, regret_m=float(np.mean(values)),
                       maximum_regret_m=float(max(values)), active_roots=len(values))
            rows.append(row)
            print(json.dumps(dict(seed=seed, **row)), flush=True)
            del model
        selected = min(rows, key=lambda row: (row['regret_m'], row['update']))
        report['models'].append(dict(seed=seed, selected=selected, candidates=rows,
                                    original_label_best_update=best_update))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
