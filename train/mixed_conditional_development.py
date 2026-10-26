"""Mixed-scene development with measured paired outcomes and explicit reuse."""
from __future__ import annotations
import argparse
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
import two_hour_development as t

e = t.e
OUT = e.BASE / 'mixed-conditional-development-20261026'


def initialize():
    state = t.initialize()
    OUT.mkdir(exist_ok=True)
    return state


def design(manifest):
    path = OUT / 'mixed-design.pth'
    if path.exists():
        return e.read(path)
    old = list(manifest['heldout_tasks'])
    former_fresh = list(e.read(t.OUT / 'fresh-final-design.pth')['tasks'])
    used = {task[0][0] for task in old + former_fresh}
    new = []
    position = 0
    while len(new) < 200:
        seed = e.seed32(9261026, position)
        position += 1
        if seed in used:
            continue
        used.add(seed)
        new.append(((seed, 6, 2. if len(new) < 100 else 2.5), None))
    result = dict(tasks=old+former_fresh+new, original_count=len(old),
        reused_former_test_count=len(former_fresh), newly_generated_count=len(new),
        role='All listed tasks are development data available for training and selection.',
        created_at=time.time())
    e.save(result, path)
    return result


def evaluate(path, tasks, r, d, name=None, batch=80, device='cuda:0'):
    path = Path(path)
    name = name or path.stem
    result_path = OUT / (name+'-evaluation.pth')
    if result_path.exists():
        result = e.read(result_path)
        if result['tasks'] != tasks or result['policy_sha256'] != e.digest(path):
            raise ValueError('A completed evaluation cannot be reassigned to another experiment.')
        if result['completed']:
            return result
    started = time.monotonic()
    actor = t.load_actor(path, d)
    rows = r.batched_credit_rows(path, tasks, batch_size=batch, device=device, actor_override=actor)
    result = dict(policy=str(path), policy_sha256=e.digest(path), tasks=tasks, rows=rows,
        completed=True, success=sum(row['success'] for row in rows),
        elapsed_seconds=time.monotonic()-started, role='mixed-scene development',
        inference_batch=batch, inference_device=device)
    e.save(result, result_path)
    print(json.dumps(dict(name=name, success=result['success'], count=len(rows),
        elapsed_seconds=result['elapsed_seconds'])), flush=True)
    return result


def train(args):
    manifest, d, l, m, r = initialize()
    folder = e.BASE / 'local-difference-joint-development'
    if args.dataset == 'retained':
        data = e.read(folder / 'F11-r136-six-trained-paired-data-seed42.pth')
        bank = e.read(folder / 'F11-r136-six-trained-factual-data-seed42.pth')
    else:
        data = e.read(OUT / 'mixed-paired.pth')
        bank = e.read(OUT / 'mixed-factual.pth')
        if not data['completed'] or not bank['completed']:
            raise ValueError('Fit only after the planned data collection has completed.')
    data, bank = l.retain_vessel_identity(data, bank)
    paired = OUT / f'{args.dataset}-{args.objective}-conditional.pth'
    ordinary = OUT / f'{args.dataset}-{args.objective}-ordinary.pth'
    if not paired.exists() or not ordinary.exists():
        l.fit_success_intervention_pair(data, paired, ordinary,
            factual_bank=bank, updates=args.updates, ensemble_size=3, width=64,
            seed=20261027, device=args.device, architecture='interaction',
            conditional_objective=args.objective, margin=.02,
            network_options=dict(identity_conditioned=True, finite_motion_basis=True))
    tasks = design(manifest)['tasks']
    indices = list(range(0, 800, 10))
    for path in (paired, ordinary):
        evaluate(path, [tasks[i] for i in indices], r, d, path.stem+'-n80', args.batch, args.device)


def collect(args):
    manifest, d, l, m, r = initialize()
    tasks = design(manifest)['tasks']
    policy = t.OUT / 'best-m1-400.pth'
    bank = l.collect_onpolicy_factual_roots(policy, tasks, OUT/'mixed-factual.pth',
        root_sample_seed=9261027, inference_batch=100, inference_device=args.device)
    design_path = OUT / 'mixed-paired-design.pth'
    if design_path.exists():
        planned = e.read(design_path)
    else:
        bad = [i for i, row in enumerate(bank['records']) if not row['values'][0]]
        good = [i for i, row in enumerate(bank['records']) if row['values'][0]]
        rng = np.random.default_rng(9261028)
        selected = bad[:90] + list(map(int, rng.choice(good, max(0, 120-min(90,len(bad))), replace=False)))
        roots = []
        for index in sorted(selected):
            row = bank['records'][index]
            roots.append(dict(scene=row['scene'], prefix_future=row['future'],
                root_steps=row['root_snapshot']['root_steps'], root_snapshot=row['root_snapshot']))
        options = [(digit*10**boat, 0, span) for boat in range(6)
                   for digit in range(1,10) for span in (1,5,20)]
        planned = l.breadth_local_design(dict(roots=roots), options,
            pairs_per_root=3, repetitions=3, seed=9261029, vessel_balanced=True)
        e.save(planned, design_path)
    l.recollect_breadth_design(policy, planned, OUT/'mixed-paired.pth', workers=1,
        inference_batch=168, inference_device=args.device, root_batch_size=8,
        isolate_factual_start=True, physical_response_horizon=5, response_feature_batch=168)


def rescue(args):
    manifest, d, l, m, r = initialize()
    old_out = t.OUT
    tasks = design(manifest)['tasks'][:600]
    initial = e.read(old_out/'best-m1-400-evaluation.pth')
    later = e.read(old_out/'fresh-best-m1-400-evaluation.pth')
    base = dict(policy=str(old_out/'best-m1-400.pth'), rows=initial['rows']+later['rows'])
    reference = e.read(old_out/'reference-r136-400-evaluation.pth')['rows'] + e.read(
        old_out/'fresh-reference-r136-400-evaluation.pth')['rows']
    path = OUT/'mixed-intervention-search.pth'
    if path.exists():
        data = e.read(path)
    else:
        previous = e.read(old_out/'intervention-search.pth')
        data = dict(records=copy.deepcopy(previous['records']),
            rescued=list(previous['rescued']), measured=[], paired_groups=[], completed=False,
            role='Exact observed-state development memory; all reused scenes count as development.')
    indices = [i for i,row in enumerate(base['rows'])
               if not row['success'] and tasks[i][0][0] not in data['rescued']]
    local_manifest = dict(manifest, heldout_tasks=tasks)
    t.OUT = OUT
    try:
        roots = t.capture_roots(base, local_manifest, d, r, indices, args.batch, args.device)
    finally:
        t.OUT = old_out
    actor = t.load_actor(base['policy'], d)
    coefficients = [(0.,0.),(1.,1.),(0.,1.),(1.,0.),(.5,.5),(0.,.5),(.5,0.),(1.,.5),(.5,1.)]
    proposals = []
    for index in indices:
        if reference[index]['success']:
            for event in reference[index]['interventions']:
                proposals.append((index, round(event['time']/.2), event['duration_steps']))
    for mode,span in [('late60',30),('half',1),('late120',30),('initial',1),
                      ('half',5),('initial',20),('late60',60),('quarter',1),('initial',60)]:
        for index in indices:
            terminal = base['rows'][index]['physical_steps']
            step = (max(0,terminal-int(mode[4:])) if mode.startswith('late') else
                    5*(terminal//10) if mode=='half' else 5*(terminal//20) if mode=='quarter' else 0)
            proposals.append((index,step,span))
    for index,step,span in proposals:
        seed = tasks[index][0][0]
        identity = (seed,step,span)
        if seed in data['rescued'] or identity in data['measured'] or (seed,step) not in roots:
            continue
        if time.time() > args.deadline-1800:
            break
        snapshot = roots[(seed,step)]
        packet = {k:v.to(args.device) for k,v in d.mechanism_tensor(
            d.joint_packet(snapshot['environment'],snapshot['observation'])).items()}
        actor.to(args.device)
        with torch.no_grad():
            control = actor(packet,deterministic=True,gate_only=True)[0][0].cpu().numpy()[:,:3]
        n = len(control)
        one = np.ones((n,1),np.float32)
        jobs, preferences = [], []
        for boat in range(n):
            for pair in coefficients:
                preference = np.full((n,2),-1.,np.float32)
                preference[boat] = pair
                action = np.concatenate((control,-one,span*one,-2*one,np.zeros((n,2),np.float32),
                                         np.zeros_like(one),preference,preference),-1)
                jobs.append(dict(environment=snapshot['environment'],observation=snapshot['observation'],
                    random_state=snapshot['random_state'],action=action,isolate_factual_start=True))
                preferences.append(preference)
        rows = r.batched_credit_rows(base['policy'],[snapshot['task']]*len(jobs),
            batch_size=len(jobs),device=args.device,initial_jobs=jobs,actor_override=actor)
        good = [j for j,row in enumerate(rows) if row['success']]
        data['measured'].append(identity)
        if good:
            selected = min(good,key=lambda j:rows[j]['duration'])
            data['records'].append(dict(features=roots[(seed,0)]['features'],root_features=snapshot['features'],
                preferences=preferences[selected],delay=step,span=span,development_task_index=index,
                root_steps=step,measured_row=rows[selected],factual_success=0,intervention_success=1))
            data['rescued'].append(seed)
        data['paired_groups'].append(dict(root_features=snapshot['features'],task=snapshot['task'],
            root_steps=step,span=span,preferences=preferences,rows=rows,isolate_factual_start=True))
        e.save(data,path)
        print(json.dumps(dict(phase='mixed_rescue',task=index,step=step,span=span,
            successful_options=len(good),rescued=len(data['rescued']))),flush=True)
    data['completed'] = True
    e.save(data,path)
    freeze_memory(data, base['policy'])


def freeze_memory(data, policy):
    model = e.read(policy)
    model['empirical_interventions'] = data['records']
    model['development_subset_interventions'] = True
    model['evidence_role'] = 'Explicit mixed-scene development memory; scenes reused in optimization.'
    model['empirical_training_sha256'] = e.digest(OUT/'mixed-intervention-search.pth')
    target = OUT/f'mixed-memory-r{len(data["records"])}.pth'
    if not target.exists():
        e.save(model,target)
    print(json.dumps(dict(policy=str(target),records=len(data['records']))),flush=True)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('train','collect','evaluate','rescue','freeze'))
    parser.add_argument('--dataset', default='retained', choices=('retained','mixed'))
    parser.add_argument('--objective', default='marginal_events', choices=('marginal_events','factual_strata'))
    parser.add_argument('--updates', type=int, default=18000)
    parser.add_argument('--batch', type=int, default=80)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--policy')
    parser.add_argument('--deadline', type=float, default=1793012414.)
    args = parser.parse_args()
    if args.operation == 'train': train(args)
    elif args.operation == 'collect': collect(args)
    elif args.operation == 'rescue': rescue(args)
    elif args.operation == 'freeze':
        initialize()
        freeze_memory(e.read(OUT/'mixed-intervention-search.pth'),t.OUT/'best-m1-400.pth')
    else:
        manifest, d, l, m, r = initialize()
        evaluate(args.policy, design(manifest)['tasks'], r, d, batch=args.batch, device=args.device)


if __name__ == '__main__':
    main()
