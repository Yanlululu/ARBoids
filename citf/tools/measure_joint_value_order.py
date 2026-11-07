"""Relate joint-candidate value ordering to actual closed-loop continuations.

The first four recorded collisions and captures define a small developmental
case set. Candidate continuations share future random streams. These returns
are measurements for choosing the next mechanism change, not learning targets.
"""
import argparse
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import sys

import torch

sys.path.insert(0, os.environ.get('CITF_CODE_ROOT', str(Path(__file__).resolve().parents[1]/'code')))
from unified.adversary import Attacker
from unified import IMPLEMENTATION_VERSION
from unified.environment import FleetEnv, TaskConfig, fusion
from unified.networks import CITF, ModelConfig
from unified.protocol import require_standard_controller
from unified.train import seed_policy, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--opponent', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--defenders', type=int, required=True)
    parser.add_argument('--scene-seed', type=int, required=True)
    parser.add_argument('--policy-seed', type=int, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('high')
    text = args.reference.read_text()
    reference = json.loads(text.splitlines()[-1] if args.reference.suffix == '.jsonl' else text)
    evaluation_count = reference['tasks']
    if (reference['scene_seed'] != args.scene_seed or reference['policy_seed'] != args.policy_seed
            or len(reference['rows']) != evaluation_count
            or sorted(r['scene_index'] for r in reference['rows']) != list(range(evaluation_count))):
        raise ValueError('The reference must contain the complete matching scene and policy stream.')
    reference_rows = {r['scene_index']: r for r in reference['rows']}
    ordered_rows = [reference_rows[i] for i in range(evaluation_count)]
    selected = [r['scene_index'] for r in ordered_rows if r['collision']][:4]
    selected += [r['scene_index'] for r in ordered_rows if r['capture']][:4]
    if not selected:
        raise ValueError('The reference contains no collision or capture cases for this diagnosis.')
    futures = [956101, 956102]
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    require_standard_controller(payload)
    model = CITF(ModelConfig(**payload['config'])).to(args.device)
    model.load_state_dict(payload['model']); model.requires_grad_(False); model.eval()
    task = TaskConfig(**reference['task'])
    if task.defenders != args.defenders or asdict(task) != asdict(replace(
            TaskConfig(**payload['task']), defenders=args.defenders, agility=task.agility)):
        raise ValueError('The reference may vary fleet size and agility, not the shared task definition.')
    initial_count = model.cfg.samples+2
    gamma = payload.get('learner_recipe', {}).get('gamma', .99)
    write_json(args.out/'configuration.json', dict(research_stage='development', checkpoint=str(args.checkpoint),
        reference=str(args.reference), opponent=None if args.opponent is None else str(args.opponent),
        task=asdict(task), source_transitions=payload.get('transitions'),
        implementation_version=IMPLEMENTATION_VERSION, code_root=str(Path(sys.modules[CITF.__module__].__file__).resolve().parent),
        evaluation_count=evaluation_count, initial_candidate_count=initial_count,
        retained_candidate_count=initial_count, intervention_control_steps=1, gamma=gamma,
        selected_scenes=selected, selection='first four collisions and first four captures in recorded scene order',
        state_times=['initial', 'five control decisions before termination'], future_policy_seeds=futures,
        candidates='all initial samples and two anchors, followed by all chain-best configurations; duplicates retained',
        continuation='one candidate action followed by the original frozen complete controller',
        use='control-relevant development diagnosis; no target fitting or new formal ablation'))
    attacker = None
    if args.opponent:
        opponent = torch.load(args.opponent, map_location='cpu', weights_only=False, mmap=True)['attacker']
        attacker = Attacker(opponent['defenders'], opponent['width'], opponent['delta']).to(args.device)
        attacker.load_state_dict(opponent['model']); attacker.requires_grad_(False)
    seed_policy(args.policy_seed, args.device)
    env = FleetEnv(evaluation_count, task, args.scene_seed, args.device)
    index = torch.tensor(selected, device=args.device)
    traces, terminal_steps = [], {}
    step = 0
    while not bool((env.outcome != 0).all()):
        with torch.no_grad():
            x = env.observation()
            result = model.select(x)
            initial = torch.cat((result['initial'], torch.stack((torch.zeros_like(result['g']), torch.ones_like(result['g'])), 1)), 1)
            trace = {key: getattr(env, key)[index].detach().cpu().clone()
                     for key in ('state', 'previous', 'order', 'agility', 'tick', 'elapsed', 'outcome')}
            trace.update(x=x[index].cpu(), a=result['a'][index].cpu(), b=result['b'][index].cpu(),
                         candidates=torch.cat((initial, result['candidates']), 1)[index].cpu(),
                         scores=torch.cat((result['initial_pool_scores'], result['scores']), 1)[index].cpu(),
                         chosen=result['g'][index].cpu())
            traces.append(trace)
            attack = None if attacker is None else attacker.select(x, env.agility)['action']
            env.step(result['action'], attack)
            for scene in selected:
                if scene not in terminal_steps and int(env.outcome[scene]):
                    terminal_steps[scene] = step
        step += 1
    write_json(args.out/'traced_outcomes.json', dict(rows=env.records()))
    if any(int(env.outcome[i]) != reference_rows[i]['outcome'] for i in selected):
        raise ValueError('The frozen controller/opponent did not reproduce the selected reference outcomes.')
    rows, origins = [], []
    for position, scene in enumerate(selected):
        for label, tick in (('initial', 0), ('preterminal', max(0, terminal_steps[scene]-4))):
            rows.append({key: value[position] for key, value in traces[tick].items()})
            origins.append(dict(scene_index=scene, state=label, control_step=tick,
                                recorded_outcome=reference_rows[scene]['outcome']))
    count = len(rows)
    stacked = {key: torch.stack([row[key] for row in rows]).to(args.device) for key in rows[0]}
    saved = env.state_dict()
    saved.update(count=count, episodes=[env.episodes[row['scene_index']] for row in origins])
    for key in ('state', 'previous', 'order', 'agility', 'tick', 'elapsed', 'outcome'):
        saved[key] = stacked[key].cpu()
    saved['noise'] = env.noise[torch.tensor([row['scene_index'] for row in origins], device=args.device)].cpu()
    candidates = stacked['candidates']
    choices = (candidates == stacked['chosen'][:, None]).all(-1).long().argmax(-1)
    initial_best = stacked['scores'][:, :initial_count].argmax(-1)
    continuation_results = []
    for future_seed in futures:
        returns, outcomes = [], []
        for candidate in range(candidates.shape[1]):
            seed_policy(future_seed, args.device)
            branch = FleetEnv(count, task, args.scene_seed, args.device)
            branch.load_state_dict(saved)
            total = torch.zeros(count, device=args.device)
            discount, iteration = 1., 0
            while not bool((branch.outcome != 0).all()):
                with torch.no_grad():
                    x = branch.observation()
                    action = (fusion(x, stacked['a'], stacked['b'], candidates[:, candidate], model.cfg.thrust_delta)
                              if iteration == 0 else model.select(x)['action'])
                    attack = None if attacker is None else attacker.select(x, branch.agility)['action']
                    _, reward, _, _ = branch.step(action, attack)
                    total += discount*reward
                discount *= gamma
                iteration += 1
            returns.append(total.cpu())
            outcomes.append(branch.outcome.cpu())
        continuation_results.append(dict(seed=future_seed, returns=torch.stack(returns, 1), outcomes=torch.stack(outcomes, 1)))
        write_json(args.out/'status.json', dict(status='continuations', completed_futures=len(continuation_results), total_futures=len(futures)))
        print(json.dumps(dict(future_seed=future_seed, states=count, candidates=candidates.shape[1])), flush=True)
    records = []
    for i, origin in enumerate(origins):
        chosen, initial = int(choices[i]), int(initial_best[i])
        record = dict(**origin, predicted_values=stacked['scores'][i].tolist(), coefficients=candidates[i].tolist(),
                      unique_configurations=len(torch.unique(candidates[i], dim=0)),
                      chosen_index=chosen, best_initial_index=initial, futures=[])
        for future in continuation_results:
            values, events = future['returns'][i], future['outcomes'][i]
            success = (events == 3) | (events == 4)
            record['futures'].append(dict(seed=future['seed'], returns=values.tolist(), outcomes=events.tolist(),
                selected_success=bool(success[chosen]), any_candidate_success=bool(success.any()),
                selected_return=float(values[chosen]), best_pool_return=float(values.max()),
                optimized_minus_initial_return=float(values[chosen]-values[initial])))
        records.append(record)
    cells = [future for record in records for future in record['futures']]
    summary = dict(states=len(records), continuations=len(cells)*candidates.shape[1],
        intervention_control_steps=1, initial_candidate_count=initial_count,
        selected_failures=sum(not c['selected_success'] for c in cells),
        failures_with_successful_candidate=sum(not c['selected_success'] and c['any_candidate_success'] for c in cells),
        all_candidates_failed=sum(not c['any_candidate_success'] for c in cells),
        mean_return_regret=sum(c['best_pool_return']-c['selected_return'] for c in cells)/len(cells),
        mean_actual_search_return_gain=sum(c['optimized_minus_initial_return'] for c in cells)/len(cells))
    summary['by_state'] = {}
    for label in ('initial', 'preterminal'):
        subset = [future for record in records if record['state'] == label for future in record['futures']]
        summary['by_state'][label] = dict(continuation_streams=len(subset),
            selected_failures=sum(not c['selected_success'] for c in subset),
            failures_with_successful_candidate=sum(not c['selected_success'] and c['any_candidate_success'] for c in subset),
            all_candidates_failed=sum(not c['any_candidate_success'] for c in subset))
    write_json(args.out/'results.json', dict(status='completed', summary=summary, records=records))
    write_json(args.out/'status.json', dict(status='completed', **summary))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
