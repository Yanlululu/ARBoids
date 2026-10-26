"""Mixed-scene development of paired intervention decisions."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
import two_hour_development as t

e = t.e
OUT = e.BASE / 'mixed-conditional-development-20261026'
BASE_POLICY = t.OUT / 'best-m1-400.pth'


def checkpoint_path(value):
    path = Path(value)
    if path.exists():
        return path
    portable = str(value).replace('\\', '/')
    marker = 'train/experiments/'
    return e.ROOT / portable[portable.index(marker):] if marker in portable else path


def initialize():
    runtime = t.initialize()
    OUT.mkdir(exist_ok=True)
    return runtime


def expanded_prepare(args):
    initialize()
    target = OUT/'stable9031-development-design.pth'
    if target.exists():
        return e.read(target)
    sources = [('original800', 'final-ci-w80-n800-evaluation.pth', 'enriched-joint-conditional-w80-n800-evaluation.pth'),
               ('added400', 'fresh-F10-evaluation.pth', 'fresh-F11-evaluation.pth'),
               ('added200', 'risk-fresh-F10-evaluation.pth', 'risk-fresh-parent_cv-evaluation.pth'),
               ('added320', 'hour-fresh-F10-evaluation.pth', 'hour-fresh-parent_cv-evaluation.pth')]
    tasks, ci_rows, cv_rows, cohorts, provenance = [], [], [], {}, []
    parent = OUT/'enriched-mlp-joint_likelihood-conditional-supportk3.pth'
    for cohort, ci_name, cv_name in sources:
        ci, cv = e.read(OUT/ci_name), e.read(OUT/cv_name)
        if not ci['completed'] or not cv['completed'] or ci['tasks'] != cv['tasks']:
            raise ValueError('Expanded development requires complete task-matched source results.')
        if ci['policy_sha256'] != e.digest(BASE_POLICY) or cv['policy_sha256'] != e.digest(parent):
            raise ValueError('The development pool must retain the fixed CI and 90.31-percent CV identities.')
        start = len(tasks)
        tasks.extend(ci['tasks']); ci_rows.extend(ci['rows']); cv_rows.extend(cv['rows'])
        cohorts[cohort] = [start, len(tasks)]
        provenance.append(dict(cohort=cohort, ci_file=ci_name, cv_file=cv_name,
                               ci_sha256=e.digest(OUT/ci_name), cv_sha256=e.digest(OUT/cv_name)))
    if len(set(map(repr, tasks))) != len(tasks):
        raise ValueError('Expanded tasks must be distinct complete task specifications.')
    extra = list(range(cohorts['original800'][1], len(tasks)))
    harmed = [i for i in extra if ci_rows[i]['success'] and not cv_rows[i]['success']]
    failed = [i for i in extra if not ci_rows[i]['success']]
    protected = [i for i in extra if ci_rows[i]['success'] and cv_rows[i]['success']]
    rng = np.random.default_rng(10261314)
    selected = harmed + failed
    for agility in (2., 2.5):
        group = [i for i in protected if float(tasks[i][0][2]) == agility]
        selected.extend(map(int, rng.choice(group, min(48, len(group)), replace=False)))
    design = dict(tasks=tasks, extra_task_indices=extra, acquisition_task_indices=sorted(set(selected)),
        cohorts=cohorts, baseline_rows=ci_rows, parent_rows=cv_rows, source_results=provenance,
        predecessor_sha256=e.digest(BASE_POLICY), parent_policy=str(parent), parent_sha256=e.digest(parent),
        selection=dict(harmed=len(harmed), ci_failed=len(failed), protected_successes=len(selected)-len(harmed)-len(failed)),
        created_at=time.time(), fixed_inference_width=80,
        evidence_role='Expanded fixed development pool; all 920 formerly new tasks are now authorized training and selection data.',
        prior_evaluation_results_preserved=True, original800=cohorts['original800'][1])
    e.save(design, target)
    print(json.dumps(dict(phase='expanded_development_frozen', tasks=len(tasks), additional=len(extra),
                         paired_roots=len(selected), selection=design['selection'])), flush=True)
    return design


def expanded_collect(args):
    _, d, local, _, _ = initialize()
    design = e.read(OUT/'stable9031-development-design.pth')
    indices = design['extra_task_indices'][args.shard::4]
    parent = checkpoint_path(design['parent_policy'])
    actor = load_actor(parent, d).to(args.device)
    selector = local.FactualProposalRootSelector([actor.successor_task_value], [e.digest(parent)],
                                                fixed_inference_width=80)
    bank_path = OUT/f'stable9031-factual-shard{args.shard}.pth'
    bank = local.collect_onpolicy_factual_roots(BASE_POLICY, [design['tasks'][i] for i in indices],
        bank_path, root_sample_seed=10261315, inference_batch=80, inference_device=args.device,
        root_selector=selector, fixed_inference_width=80)
    chosen = set(design['acquisition_task_indices'])
    roots = []
    for global_index, record in zip(indices, bank['records']):
        if global_index not in chosen:
            continue
        snapshot = record['root_snapshot']
        observed = snapshot.get('proposal_observation', {}).get('options')
        proposal = tuple(map(int, observed[0])) if observed is not None else (0, 0, 0)
        if not proposal[0]:
            proposal = ((1+global_index % 9)*10**(global_index % 6), 0, 5)
        code, delay, span = proposal
        boat = next(i for i in range(6) if code//10**i % 10)
        digit = code//10**boat % 10
        coefficients = [(a, b) for a in (0., .5, 1.) for b in (0., .5, 1.)]
        opposite = 1+max(range(9), key=lambda j: sum((a-b)**2 for a, b in zip(coefficients[j], coefficients[digit-1])))
        options = [(0, 0, 0)] + sorted({proposal, (opposite*10**boat, delay, span),
                                       (code, 0, 1), (code, 0, 5), (code, 0, 20)})
        futures = [e.seed32(10261316, global_index, repetition) for repetition in range(3)]
        roots.append(dict(scene=record['scene'], prefix_future=record['future'],
            root_steps=snapshot['root_steps'], root_snapshot=snapshot, options=options,
            futures=futures, future_weights=np.ones(3, np.float32),
            outcomes={(q, f): dict(duration=400) for q in range(len(options)) for f in range(3)},
            expanded_task_index=global_index, native_parent_proposal=proposal))
    signatures = [(root['expanded_task_index'], root['root_steps'], root['options'], root['futures']) for root in roots]
    blueprint = dict(roots=roots, acquisition_design=dict(kind='breadth_local_pairs', roots=len(roots),
        repetitions=3, seed=10261316, inherited_labels=False, shared_by_all_estimators=True,
        planned_observations=sum(len(root['outcomes']) for root in roots),
        design_sha256=hashlib.sha256(repr(signatures).encode()).hexdigest(),
        allocation='Previously observed harms and CI failures plus balanced protected successes; first native CV proposal, opposite and duration alternatives; fresh common futures.'))
    e.save(blueprint, OUT/f'stable9031-paired-design-shard{args.shard}.pth')
    data = local.recollect_breadth_design(BASE_POLICY, blueprint,
        OUT/f'stable9031-paired-shard{args.shard}.pth', workers=1, inference_batch=80,
        inference_device=args.device, root_batch_size=10, isolate_factual_start=True,
        physical_response_horizon=5, fixed_inference_width=80, response_feature_batch=80)
    print(json.dumps(dict(phase='expanded_shard_completed', shard=args.shard, factual=len(bank['records']),
                         paired_roots=len(data['roots']))), flush=True)


def expanded_merge(args):
    initialize()
    paths = [OUT/f'stable9031-paired-shard{shard}.pth' for shard in range(4)]
    if not all(path.exists() and e.read(path).get('completed') for path in paths):
        raise ValueError('Every new paired shard must finish before expanded training.')
    old_path = OUT/'enriched-paired.pth'
    parts = [e.read(old_path)] + [e.read(path) for path in paths]
    combined = copy.deepcopy(parts[0])
    combined['roots'] = [root for part in parts for root in part['roots']]
    for index, root in enumerate(combined['roots']):
        root['design_root'] = index
    observations = sum(len(root['outcomes']) for root in combined['roots'])
    combined.update(completed=True, next_design_root=len(combined['roots']), design_root_count=len(combined['roots']),
        measured_terminal_observations=observations, unique_terminal_observations=observations,
        simulation_steps=sum(part['simulation_steps'] for part in parts),
        training_data=str(OUT/'stable9031-expanded-paired.pth'),
        acquisition_design=dict(kind='expanded_completed_development', roots=len(combined['roots']),
            source_sha256=[e.digest(path) for path in [old_path]+paths], measured_observations=observations,
            execution='Unchanged 90.31-percent parent intervention semantics; actual labels for each candidate.'))
    e.save(combined, OUT/'stable9031-expanded-paired.pth')
    banks = [e.read(OUT/'mixed-factual.pth')]+[e.read(OUT/f'stable9031-factual-shard{shard}.pth') for shard in range(4)]
    bank = copy.deepcopy(banks[0])
    bank['records'] = [record for part in banks for record in part['records']]
    bank.update(completed=True, trajectories=len(bank['records']),
        tasks=[(tuple(record['scene']), record['future']) for record in bank['records']],
        state_labels=sum(len(record['values']) for record in bank['records']),
        simulation_steps=sum(part['simulation_steps'] for part in banks),
        source_bank_sha256=[e.digest(OUT/'mixed-factual.pth')]+[
            e.digest(OUT/f'stable9031-factual-shard{shard}.pth') for shard in range(4)],
        root_sampling='Combined original factual states and new first-native-proposal states; individual snapshots preserve their sampler.',
        evidence_role='All fixed expanded development tasks, shared by the paired and ordinary learners.')
    e.save(bank, OUT/'stable9031-expanded-factual.pth')
    print(json.dumps(dict(phase='expanded_training_data_ready', roots=len(combined['roots']),
                         terminal_observations=observations, trajectories=bank['trajectories'])), flush=True)


def expanded_ablation(args):
    initialize()
    parent = OUT/'expanded-mlp-joint_likelihood-s20261050-conditional-supportk3.pth'
    model = e.read(parent)
    model['mechanism_ablation'] = (True, False, True)
    model['development_ablation'] = 'remove CI at execution; value learner is not retrained'
    model['ablation_parent_sha256'] = e.digest(parent)
    target = OUT/'expanded-mlp-joint_likelihood-s20261050-conditional-without-ci.pth'
    if not target.exists():
        e.save(model, target)
    print(json.dumps(dict(phase='expanded_execution_ablation_ready', policy=str(target))), flush=True)


def expanded_confirmation_prepare(args):
    manifest, _, _, _, _ = initialize()
    target = OUT/'stable9031-confirmation-design.pth'
    if target.exists():
        design = e.read(target)
        if any(e.digest(checkpoint_path(item['path'])) != item['sha256']
               for item in design['policies'].values()):
            raise ValueError('The confirmation models changed after freezing.')
        return design
    used = {int(task[0][0]) for task in prepare(manifest)['tasks']}
    paired = e.read(OUT/'stable9031-expanded-paired.pth')
    factual = e.read(OUT/'stable9031-expanded-factual.pth')
    used.update(int(root['scene'][0]) for root in paired['roots'])
    used.update(int(record['scene'][0]) for record in factual['records'])
    for previous_path in OUT.glob('*design*.pth'):
        used.update(int(task[0][0]) for task in e.read(previous_path).get('tasks', []))
    excluded = len(used)
    tasks, position = [], 0
    while len(tasks) < 400:
        seed = e.seed32(10261411, position)
        position += 1
        if seed in used:
            continue
        used.add(seed)
        tasks.append(((seed, 6, 2. if len(tasks) < 200 else 2.5), None))
    paths = dict(F00=t.OUT/'common-source-f00.pth', F10=BASE_POLICY,
        F01=OUT/'expanded-mlp-joint_likelihood-s20261050-conditional-without-ci.pth',
        F11=OUT/'expanded-mlp-joint_likelihood-s20261050-conditional-supportk3.pth',
        ordinary=OUT/'expanded-mlp-joint_likelihood-s20261050-ordinary-supportk3.pth',
        independent=t.OUT/'best-independent-400.pth')
    paths.update({f'cv{seed}': OUT/f'expanded-mlp-joint_likelihood-s{seed}-conditional-supportk3.pth'
                  for seed in range(20261051, 20261055)})
    design = dict(tasks=tasks, policies={name: dict(path=str(path), sha256=e.digest(path))
        for name, path in paths.items()}, frozen_at=time.time(), excluded_layouts_count=excluded,
        representative_seed=20261050, fixed_inference_width=80,
        selection='All five models and the representative seed were fixed before complete expanded development outcomes.',
        evidence_role='One fixed independent confirmation cohort shared by all five CV training runs and the matched controls; no outcome-driven task replacement.',
        development_design_sha256=e.digest(OUT/'stable9031-development-design.pth'))
    e.save(design, target)
    print(json.dumps(dict(phase='expanded_confirmation_frozen', tasks=len(tasks),
                         policies=list(paths), excluded_layouts=excluded)), flush=True)
    return design


def journal_prepare(args):
    manifest, _, _, _, _ = initialize()
    target = OUT/'journal-readiness-20261027-design.pth'
    if target.exists():
        design = e.read(target)
        if args.scheme_sha256 and args.scheme_sha256 != design['scheme_sha256']:
            raise ValueError('The journal evidence scheme differs from the frozen replication design.')
        if any(e.digest(checkpoint_path(item['path'])) != item['sha256'] for item in design['policies'].values()):
            raise ValueError('A replication policy changed after freezing.')
        return design
    if not args.scheme_sha256 or len(args.scheme_sha256) != 64:
        raise ValueError('Record the exact user-selected journal evidence scheme before replication.')
    selected = e.read(OUT/'stable9031-selected-design.pth')
    paths = dict(CI=BASE_POLICY, independent=t.OUT/'best-independent-400.pth',
                 ordinary=checkpoint_path(selected['policies']['ordinary']['path']))
    for seed in range(20261050, 20261055):
        name = 'F11' if seed == 20261050 else f'cv{seed}'
        paths[f'CV{seed}'] = checkpoint_path(selected['policies'][name]['path'])
    joint, independent = e.read(paths['CI']), e.read(paths['independent'])
    if set(joint['actor']) != set(independent['actor']) or any(
            not torch.equal(joint['actor'][key], independent['actor'][key]) for key in joint['actor']):
        raise ValueError('Joint and independent fusion must use identical learned source parameters.')
    joint_setting = dict(joint['actuator_fusion_refinement'])
    independent_setting = dict(independent['actuator_fusion_refinement'])
    if joint_setting.pop('selection') != 'joint' or independent_setting.pop('selection') != 'independent' or joint_setting != independent_setting:
        raise ValueError('The direct CI comparison may change only the selection rule.')
    if any(joint.get(key) != independent.get(key) for key in
           ('risk_horizon', 'risk_clearance', 'prediction_step', 'development_candidate_prior', 'mechanism_ablation')):
        raise ValueError('The direct CI comparison must preserve prediction and control conditions.')
    used = {int(task[0][0]) for task in prepare(manifest)['tasks']}
    for previous_path in OUT.glob('*design*.pth'):
        used.update(int(task[0][0]) for task in e.read(previous_path).get('tasks', []))
    paired, factual = e.read(OUT/'stable9031-expanded-paired.pth'), e.read(OUT/'stable9031-expanded-factual.pth')
    used.update(int(root['scene'][0]) for root in paired['roots'])
    used.update(int(record['scene'][0]) for record in factual['records'])
    excluded = len(used)
    tasks, cohorts, position = [], {}, 0
    for replicate in range(1, 4):
        start = len(tasks)
        while len(tasks)-start < 400:
            seed = e.seed32(10271631, position)
            position += 1
            if seed in used:
                continue
            used.add(seed)
            tasks.append(((seed, 6, 2. if len(tasks)-start < 200 else 2.5), None))
        cohorts[f'replicate{replicate}'] = (start, len(tasks))
    design = dict(tasks=tasks, cohorts=cohorts,
        policies={name: dict(path=str(path), sha256=e.digest(path)) for name, path in paths.items()},
        scheme_sha256=args.scheme_sha256, scheme_name='CIV-RL_最终期刊证据方案_定稿精简版.md',
        frozen_at=time.time(), excluded_known_layouts=excluded, fixed_inference_width=80,
        representative_seed=20261050, source_policy_sha256=e.digest(BASE_POLICY),
        runtime_sha256=e.digest(t.OUT/'development-runtime.pth'),
        evidence_role='Three prespecified new replication batches for current-method development readiness; all batches and all five CV initializations must be retained.',
        scope='Table II mechanism comparisons on one frozen six-vessel source policy; this does not replace five complete source-policy plus CV training pipelines or the other required journal experiments.',
        reporting_rule='Report each 400-task batch and the pooled 1200 tasks. A best batch may be identified only as a disclosed best-of-three illustration, not as an unselected final test.',
        illustrative_batch_rule='Largest mean paired CV success gain over CI; ties use fewer mean harmful interventions, then the original batch order.')
    e.save(design, target)
    print(json.dumps(dict(phase='journal_replications_frozen', tasks=len(tasks), cohorts=cohorts, policies=list(paths))), flush=True)
    return design


def journal_analysis_prepare(args):
    initialize()
    target = OUT/'journal-readiness-20261027-analysis.pth'
    design_path = OUT/'journal-readiness-20261027-design.pth'
    design = e.read(design_path)
    if target.exists():
        specification = e.read(target)
        if specification['design_sha256'] != e.digest(design_path):
            raise ValueError('The sample-size specification belongs to another task design.')
        return specification
    if any((OUT/f'journal-readiness-{name}-n1200-evaluation.pth').exists()
           for name in design['policies']):
        raise ValueError('Freeze the sample-size analysis before reading completed replication outcomes.')
    subsets = {}
    for label, (start, end) in design['cohorts'].items():
        groups = [[i for i in range(start, end) if float(design['tasks'][i][0][2]) == agility]
                  for agility in (2., 2.5)]
        for count in (100, 200, 400):
            indices = sorted(groups[0][:count//2]+groups[1][:count//2])
            if len(indices) != count:
                raise ValueError('Every sample-size subset must retain equal agility strata.')
            subsets[f'{label}_n{count}'] = indices
    for count in (400, 800, 1200):
        subsets[f'pooled_n{count}'] = list(range(count))
    specification = dict(created_at=time.time(), design_sha256=e.digest(design_path),
        subsets=subsets, subset_rule='Within each batch, take the first 50, 100, or 200 tasks of each agility stratum. Pooled counts accumulate complete batches in their original order.',
        comparisons=['CI versus independent fusion', 'Five CV fits versus CI',
                     'Five CV fits versus the matched ordinary value model',
                     'Harmful and beneficial interventions relative to the same CI outcomes'],
        reporting_rule='Retain every batch, all five CV fits, and all sample sizes. A selected best batch is an illustration chosen among three observed batches.',
        fit_repeat_scope='Five CV initializations share one frozen source policy; this screening does not establish full-pipeline seed stability.',
        inference_rule='Do not treat the five predictions on each scene as five independent scene observations.')
    e.save(specification, target)
    print(json.dumps(dict(phase='journal_analysis_frozen', subsets={key: len(value) for key, value in subsets.items()})), flush=True)
    return specification


def journal_ordinary_prepare(args):
    initialize()
    target = OUT/'journal-readiness-ordinary-moderate.pth'
    source = OUT/'stable9031-selected-ordinary.pth'
    if target.exists():
        if e.read(target)['journal_acceptance_check']['source_sha256'] != e.digest(source):
            raise ValueError('The ordinary acceptance candidate changed its fitted weights.')
        return
    model = e.read(source)
    model['successor_task_value'].update(ensemble_gain_penalty=.5, gain_margin=.05)
    model['journal_acceptance_check'] = dict(source_sha256=e.digest(source),
        additional_updates=0, selection='Use the same pre-existing 320-task development pilot and the same raw, moderate, cautious acceptance candidates as CV.')
    e.save(model, target)
    print(json.dumps(dict(phase='ordinary_acceptance_candidate_ready', path=str(target))), flush=True)


def journal_ordinary_select(args):
    initialize()
    pilot_path = OUT/'stable9031-stability-design.pth'
    pilot, development = e.read(pilot_path), e.read(OUT/'stable9031-development-design.pth')
    paths = dict(raw=OUT/'stable9031-expanded-s20261050-ordinary-n1720-evaluation.pth',
                 cautious=OUT/'stable9031-selected-development-ordinary-n1720-evaluation.pth',
                 moderate=OUT/'journal-readiness-ordinary-moderate-n320-evaluation.pth')
    comparisons = {}
    for label, path in paths.items():
        value = e.read(path)
        tasks = pilot['tasks'] if label == 'moderate' else development['tasks']
        if (not value.get('completed') or value['tasks'] != tasks or value['fixed_inference_width'] != 80 or
                value['policy_sha256'] != e.digest(checkpoint_path(value['policy']))):
            raise ValueError('Every ordinary acceptance setting requires a complete matched development evaluation.')
        rows = value['rows'] if label == 'moderate' else [value['rows'][i] for i in pilot['indices']]
        if label != 'moderate' and [tasks[i] for i in pilot['indices']] != pilot['tasks']:
            raise ValueError('The ordinary and CV acceptance selections must share the same development scenes.')
        comparisons[label] = dict(success=sum(row['success'] for row in rows),
            harm=sum(a['success'] < b['success'] for a, b in zip(rows, pilot['baseline_rows'], strict=True)),
            rescue=sum(a['success'] > b['success'] for a, b in zip(rows, pilot['baseline_rows'], strict=True)),
            collision=sum(row['collision'] for row in rows),
            interventions=sum(row.get('conditional_interventions', 0) for row in rows),
            policy=value['policy'], policy_sha256=value['policy_sha256'],
            evaluation=dict(path=str(path), sha256=e.digest(path)))
    selected = max(comparisons, key=lambda label: (comparisons[label]['success'], -comparisons[label]['harm']))
    result = dict(completed=True, created_at=time.time(), selected=selected, comparisons=comparisons,
        task_count=len(pilot['tasks']), baseline_success=sum(row['success'] for row in pilot['baseline_rows']),
        pilot_sha256=e.digest(pilot_path),
        selection_rule='Highest development success; ties use fewer harms. Same raw, moderate, and cautious candidates and same development scenes as CV; all ordinary fits share unchanged weights.')
    e.save(result, OUT/'journal-readiness-ordinary-selection.pth')
    print(json.dumps(dict(phase='ordinary_acceptance_selected', **result)), flush=True)


def journal_report(args):
    _, development, _, _, _ = initialize()
    design_path = OUT/'journal-readiness-20261027-design.pth'
    analysis_path = OUT/'journal-readiness-20261027-analysis.pth'
    design, analysis = e.read(design_path), e.read(analysis_path)
    if analysis['design_sha256'] != e.digest(design_path):
        raise ValueError('The fixed analysis and evaluation tasks must match.')
    evaluations, identities, models = {}, {}, {}
    for name, item in design['policies'].items():
        path = OUT/f'journal-readiness-{name}-n1200-evaluation.pth'
        value = e.read(path)
        policy = checkpoint_path(item['path'])
        if (not value.get('completed') or value['tasks'] != design['tasks'] or
                len(value['rows']) != len(design['tasks']) or
                value['policy_sha256'] != item['sha256'] or e.digest(policy) != item['sha256'] or
                value['fixed_inference_width'] != design['fixed_inference_width']):
            raise ValueError('Every readiness arm must finish the identical frozen tasks with the frozen model and inference width.')
        evaluations[name], models[name] = value, e.read(policy)
        identities[name] = dict(path=str(path), sha256=e.digest(path),
            policy=str(policy), policy_sha256=item['sha256'], elapsed_seconds=value['elapsed_seconds'])

    def summary(rows, reference):
        rescue = sum(a['success'] > b['success'] for a, b in zip(rows, reference, strict=True))
        harm = sum(a['success'] < b['success'] for a, b in zip(rows, reference, strict=True))
        success = sum(row['success'] for row in rows)
        return dict(episodes=len(rows), success=success, success_percent=100.*success/len(rows),
            capture=sum(row['capture'] for row in rows), collision=sum(row['collision'] for row in rows),
            interventions=sum(row.get('conditional_interventions', 0) for row in rows),
            rescue=rescue, harm=harm, net_success=rescue-harm,
            gain_percentage_points=100.*(rescue-harm)/len(rows))

    def repeated(runs):
        rates = np.asarray([run['success_percent'] for run in runs])
        gains = np.asarray([run['gain_percentage_points'] for run in runs])
        return dict(runs=len(runs), success_counts=[run['success'] for run in runs],
            mean_success_percent=float(rates.mean()), sample_sd_percentage_points=float(rates.std(ddof=1)),
            minimum_success_percent=float(rates.min()), maximum_success_percent=float(rates.max()),
            mean_gain_percentage_points=float(gains.mean()), minimum_gain_percentage_points=float(gains.min()),
            maximum_gain_percentage_points=float(gains.max()),
            positive_runs=int((gains > 0).sum()), tied_runs=int((gains == 0).sum()), negative_runs=int((gains < 0).sum()),
            mean_rescue=float(np.mean([run['rescue'] for run in runs])),
            mean_harm=float(np.mean([run['harm'] for run in runs])),
            mean_collision=float(np.mean([run['collision'] for run in runs])),
            mean_interventions=float(np.mean([run['interventions'] for run in runs])))

    names = [f'CV{seed}' for seed in range(20261050, 20261055)]
    source_actor_match = {name: set(models[name]['actor']) == set(models['CI']['actor']) and all(
        torch.equal(models[name]['actor'][key], models['CI']['actor'][key]) for key in models['CI']['actor'])
        for name in ['independent', 'ordinary', *names]}
    if not all(source_actor_match.values()):
        raise ValueError('The direct mechanism comparisons must retain the same frozen learned source actor.')

    def summarize_indices(indices):
        rows = {name: [value['rows'][i] for i in indices] for name, value in evaluations.items()}
        ci, ordinary = rows['CI'], rows['ordinary']
        arms = {name: summary(value, ci) for name, value in rows.items()}
        runs = [dict(seed=seed, **arms[name]) for seed, name in zip(range(20261050, 20261055), names, strict=True)]
        versus_ordinary = [dict(seed=seed, **summary(rows[name], ordinary))
                           for seed, name in zip(range(20261050, 20261055), names, strict=True)]
        cv = repeated(runs)
        return dict(task_count=len(indices), models=arms,
            CI_versus_independent=summary(ci, rows['independent']),
            CV_versus_CI=dict(runs=runs, summary=cv),
            CV_versus_ordinary=dict(runs=versus_ordinary, summary=repeated(versus_ordinary)),
            harmful_intervention_comparison=dict(
                ordinary_harm=arms['ordinary']['harm'], CV_mean_harm=cv['mean_harm'],
                CV_minus_ordinary_mean_harm=cv['mean_harm']-arms['ordinary']['harm'],
                runs_with_fewer_harms=sum(run['harm'] < arms['ordinary']['harm'] for run in runs),
                ordinary_rescue=arms['ordinary']['rescue'], CV_mean_rescue=cv['mean_rescue'],
                ordinary_interventions=arms['ordinary']['interventions'], CV_mean_interventions=cv['mean_interventions'],
                definition='An episode where the intervention policy fails while the same CI baseline succeeds. Intervention counts are separate trigger counts.'))

    subsets = {name: summarize_indices(indices) for name, indices in analysis['subsets'].items()}
    cohorts = {}
    for label, (start, end) in design['cohorts'].items():
        value = copy.deepcopy(subsets[f'{label}_n400'])
        value['agility'] = {str(agility): summarize_indices(
            [i for i in range(start, end) if float(design['tasks'][i][0][2]) == agility]) for agility in (2., 2.5)}
        cohorts[label] = value
    pooled = copy.deepcopy(subsets['pooled_n1200'])
    pooled['agility'] = {str(agility): summarize_indices(
        [i for i, task in enumerate(design['tasks']) if float(task[0][2]) == agility]) for agility in (2., 2.5)}
    best = max(cohorts, key=lambda label: (
        cohorts[label]['CV_versus_CI']['summary']['mean_gain_percentage_points'],
        -cohorts[label]['CV_versus_CI']['summary']['mean_harm']))
    ordinary_fit = models['ordinary']['successor_task_fit']
    matched_keys = ['training_observation_sha256', 'training_target_sha256', 'total_parameter_updates',
                    'member_training_roots', 'member_factual_trajectories', 'member_updates']
    fits = []
    for name in names:
        model, fit = models[name], models[name]['successor_task_fit']
        if any(fit[key] != ordinary_fit[key] for key in matched_keys):
            raise ValueError('The ordinary control must retain identical data and training budget.')
        for key in ('ensemble_gain_penalty', 'gain_margin', 'options'):
            if model['successor_task_value'][key] != models['ordinary']['successor_task_value'][key]:
                raise ValueError('The direct value comparison must use common options and acceptance settings.')
        fits.append(dict(name=name, trainable_parameters=fit['trainable_parameters'],
                         member_seeds=fit['member_seeds']))
    ordinary_selection_path = OUT/'journal-readiness-ordinary-selection.pth'
    ordinary_selection = e.read(ordinary_selection_path)
    selected_ordinary = ordinary_selection['comparisons'][ordinary_selection['selected']]
    if selected_ordinary['policy_sha256'] != design['policies']['ordinary']['sha256']:
        raise ValueError('The development-selected ordinary setting needs its own complete new-scene evaluation before reporting the fair value comparison.')
    fleet_interface = []
    for count in (6, 3, 8):
        actor = load_actor(checkpoint_path(design['policies'][names[0]]['path']), development)
        decision = actor.successor_task_value
        try:
            decision.configure_fleet(count)
            item = dict(supported_configuration=True, options=len(decision.options))
        except ValueError as error:
            item = dict(supported_configuration=False, error=str(error))
        fleet_interface.append(dict(defenders=count, representation=decision.network.representation, **item))
    representative = f"CV{design['representative_seed']}"
    representative_index = names.index(representative)
    representative_results = {label: dict(
        versus_CI=value['models'][representative],
        versus_ordinary=value['CV_versus_ordinary']['runs'][representative_index],
        ordinary_harm_relative_to_CI=value['models']['ordinary']['harm'])
        for label, value in {**cohorts, 'pooled': pooled}.items()}
    representative_best = max(cohorts, key=lambda label: (
        representative_results[label]['versus_CI']['gain_percentage_points'],
        representative_results[label]['versus_ordinary']['gain_percentage_points'],
        -representative_results[label]['versus_CI']['harm']))
    result = dict(completed=True, created_at=time.time(),
        task_count=len(design['tasks']), design=dict(path=str(design_path), sha256=e.digest(design_path)),
        analysis=dict(path=str(analysis_path), sha256=e.digest(analysis_path), **analysis),
        scheme_sha256=design['scheme_sha256'], evidence_role=design['evidence_role'],
        implementation_sha256=e.digest(Path(__file__)), runtime_sha256=design['runtime_sha256'],
        evaluations=identities, cohorts=cohorts, pooled=pooled, sample_size_subsets=subsets,
        source_actor_identical_to_CI=source_actor_match,
        illustrative_best_batch=dict(name=best, rule=design['illustrative_batch_rule'],
            positive_mean_CV_gain=cohorts[best]['CV_versus_CI']['summary']['mean_gain_percentage_points'] > 0,
            selection='Chosen after comparing three completed batches; retain the other batches and pooled evidence.'),
        prespecified_representative=dict(seed=design['representative_seed'],
            policy=design['policies'][representative], results=representative_results,
            positive_CI_batches=sum(value['versus_CI']['gain_percentage_points'] > 0
                                    for label, value in representative_results.items() if label != 'pooled'),
            positive_ordinary_batches=sum(value['versus_ordinary']['gain_percentage_points'] > 0
                                          for label, value in representative_results.items() if label != 'pooled'),
            display_batch=representative_best, task_interval=design['cohorts'][representative_best],
            model_selection='The representative seed was recorded before the new outcomes. Its repeated scene-batch performance is distinct from the five-fit mean.',
            display_selection='After observing all batches, choose the representative model batch with largest gain over CI, then gain over ordinary, then fewer harms. This is a disclosed selected illustration.'),
        ordinary_development_selection=dict(path=str(ordinary_selection_path),
            sha256=e.digest(ordinary_selection_path), **ordinary_selection),
        fleet_interface_check=dict(checks=fleet_interface,
            scope='Configuration-only check of the frozen complete CV model; successful configuration is not a closed-loop performance result.'),
        training_match=dict(common={key: ordinary_fit[key] for key in matched_keys}, CV=fits,
            ordinary_trainable_parameters=ordinary_fit['trainable_parameters'],
            ordinary_initializations=1, CV_initializations=5,
            source_policy_initializations=1, acceptance_rule='Common mean gain - 0.5 * member gain SD > 0.1'),
        scope=design['scope'])
    target = OUT/'journal-readiness-results.json'
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(phase='journal_readiness_complete', path=str(target),
        cohorts={label: dict(CI=value['models']['CI']['success_percent'],
            independent=value['models']['independent']['success_percent'], ordinary=value['models']['ordinary']['success_percent'],
            CV=value['CV_versus_CI']['summary'], CV_versus_ordinary=value['CV_versus_ordinary']['summary'],
            harm=value['harmful_intervention_comparison']) for label, value in {**cohorts, 'pooled': pooled}.items()},
        illustrative_best_batch=best)), flush=True)


def stability_prepare(args):
    initialize()
    target = OUT/'stable9031-stability-design.pth'
    if target.exists():
        return e.read(target)
    development = e.read(OUT/'stable9031-development-design.pth')
    rng = np.random.default_rng(10261502)
    indices = []
    for agility in (2., 2.5):
        group = [i for i, task in enumerate(development['tasks']) if float(task[0][2]) == agility]
        indices.extend(map(int, rng.choice(group, 160, replace=False)))
    indices.sort()
    settings = dict(moderate=dict(penalty=.5, margin=.05), cautious=dict(penalty=.5, margin=.1))
    models, raw_rows, raw_identities = {}, {}, {}
    for seed in range(20261050, 20261055):
        parent = OUT/f'expanded-mlp-joint_likelihood-s{seed}-conditional-supportk3.pth'
        value_path = OUT/f'stable9031-expanded-s{seed}-conditional-n1720-evaluation.pth'
        value = e.read(value_path)
        if (not value['completed'] or value['tasks'] != development['tasks'] or
                value['policy_sha256'] != e.digest(parent)):
            raise ValueError('Stability selection requires the completed common development evaluations.')
        raw_rows[str(seed)] = [value['rows'][i] for i in indices]
        raw_identities[str(seed)] = dict(policy=str(parent), sha256=e.digest(parent),
                                        evaluation_sha256=e.digest(value_path))
        for label, setting in settings.items():
            model = e.read(parent)
            model['successor_task_value'].update(ensemble_gain_penalty=setting['penalty'], gain_margin=setting['margin'])
            model['net_gain_selection'] = dict(ensemble_gain_penalty=setting['penalty'], gain_margin=setting['margin'],
                parent_sha256=e.digest(parent), additional_observations=0, additional_updates=0,
                rule='Require mean paired gain minus half its member standard deviation to exceed the shared development-selected margin.')
            path = OUT/f'stable9031-stability-{label}-s{seed}.pth'
            if path.exists():
                raise ValueError('A stability candidate already exists without its design.')
            e.save(model, path)
            models[f'{label}-s{seed}'] = dict(policy=str(path), sha256=e.digest(path))
    design = dict(tasks=[development['tasks'][i] for i in indices], indices=indices,
        baseline_rows=[development['baseline_rows'][i] for i in indices],
        raw_rows=raw_rows, raw_identities=raw_identities, models=models, settings=settings,
        created_at=time.time(), fixed_inference_width=80,
        sampling='320 tasks sampled without outcomes from the fixed development pool, balanced across the two agility strata.',
        evidence_role='Development selection of one common uncertainty penalty and gain margin across all five completed training runs.',
        confirmation_scope='The previously observed 400-task confirmation is retained as fixed validation for subsequent revisions.',
        selection_rule='Highest mean success over five runs; ties resolved by fewer harmful interventions and lower sample SD.')
    e.save(design, target)
    print(json.dumps(dict(phase='stability_candidates_ready', tasks=len(indices), candidates=len(models))), flush=True)
    return design


def stability_select(args):
    initialize()
    design = e.read(OUT/'stable9031-stability-design.pth')
    groups = {'raw': design['raw_rows']}
    for label in design['settings']:
        groups[label] = {}
        for seed in range(20261050, 20261055):
            name = f'{label}-s{seed}'
            path = OUT/f'stable9031-stability-{name}-n320-evaluation.pth'
            value = e.read(path)
            if (not value['completed'] or value['tasks'] != design['tasks'] or
                    value['policy_sha256'] != design['models'][name]['sha256']):
                raise ValueError('Every stability candidate must finish the same pilot tasks.')
            groups[label][str(seed)] = value['rows']
    comparisons = {}
    baseline = design['baseline_rows']
    for label, members in groups.items():
        runs = [dict(seed=int(seed), success=sum(row['success'] for row in rows),
            rescue=sum(a['success'] > b['success'] for a, b in zip(rows, baseline, strict=True)),
            harm=sum(a['success'] < b['success'] for a, b in zip(rows, baseline, strict=True)),
            collision=sum(row['collision'] for row in rows),
            interventions=sum(row.get('conditional_interventions', 0) for row in rows))
            for seed, rows in members.items()]
        rates = np.asarray([run['success'] for run in runs])/len(baseline)*100
        comparisons[label] = dict(runs=runs, mean_success_percent=float(rates.mean()),
            sample_sd_percentage_points=float(rates.std(ddof=1)),
            mean_harm=float(np.mean([run['harm'] for run in runs])),
            mean_rescue=float(np.mean([run['rescue'] for run in runs])))
    selected = max(comparisons, key=lambda label: (comparisons[label]['mean_success_percent'],
        -comparisons[label]['mean_harm'], -comparisons[label]['sample_sd_percentage_points']))
    result = dict(completed=True, selected=selected, comparisons=comparisons,
        baseline_success=sum(row['success'] for row in baseline), task_count=len(baseline),
        selection_rule=design['selection_rule'], created_at=time.time(),
        design_sha256=e.digest(OUT/'stable9031-stability-design.pth'))
    e.save(result, OUT/'stable9031-stability-selection.pth')
    print(json.dumps(result), flush=True)


def stability_final_prepare(args):
    initialize()
    target = OUT/'stable9031-selected-design.pth'
    if target.exists():
        design = e.read(target)
        if any(e.digest(checkpoint_path(item['path'])) != item['sha256']
               for item in design['policies'].values()):
            raise ValueError('A selected model changed after freezing.')
        return design
    selection_path = OUT/'stable9031-stability-selection.pth'
    selection = e.read(selection_path)
    pilot = e.read(OUT/'stable9031-stability-design.pth')
    if not selection['completed'] or selection['design_sha256'] != e.digest(OUT/'stable9031-stability-design.pth'):
        raise ValueError('The complete common-setting selection must precede final evaluation.')
    label = selection['selected']
    setting = dict(penalty=0., margin=.02) if label == 'raw' else pilot['settings'][label]
    paths = {}
    for seed in range(20261050, 20261055):
        path = (OUT/f'expanded-mlp-joint_likelihood-s{seed}-conditional-supportk3.pth' if label == 'raw' else
                OUT/f'stable9031-stability-{label}-s{seed}.pth')
        name = 'F11' if seed == 20261050 else f'cv{seed}'
        paths[name] = path
    ordinary_source = OUT/'expanded-mlp-joint_likelihood-s20261050-ordinary-supportk3.pth'
    ordinary = e.read(ordinary_source)
    ordinary['successor_task_value'].update(ensemble_gain_penalty=setting['penalty'], gain_margin=setting['margin'])
    ordinary['net_gain_selection'] = dict(ensemble_gain_penalty=setting['penalty'], gain_margin=setting['margin'],
        parent_sha256=e.digest(ordinary_source), additional_observations=0, additional_updates=0,
        rule='The same acceptance setting selected across the five CV runs; identical data, update budget and candidate support.')
    paths['ordinary'] = OUT/'stable9031-selected-ordinary.pth'
    e.save(ordinary, paths['ordinary'])
    ablation = e.read(paths['F11'])
    ablation['mechanism_ablation'] = (True, False, True)
    ablation['development_ablation'] = 'remove CI at execution; value learner is not retrained'
    ablation['ablation_parent_sha256'] = e.digest(paths['F11'])
    paths['F01'] = OUT/'stable9031-selected-F01.pth'
    e.save(ablation, paths['F01'])
    cohorts = dict(development=OUT/'stable9031-development-design.pth',
                   validation=OUT/'stable9031-confirmation-design.pth')
    design = dict(completed=True, selected_setting=label, setting=setting,
        policies={name: dict(path=str(path), sha256=e.digest(path)) for name, path in paths.items()},
        cohorts={name: dict(path=str(path), sha256=e.digest(path), tasks=len(e.read(path)['tasks']))
                 for name, path in cohorts.items()}, selection_sha256=e.digest(selection_path),
        representative_seed=20261050, fixed_inference_width=80, frozen_at=time.time(),
        evidence_roles=dict(development='The fixed expanded 1720-task development pool, including the selection pilot.',
            validation='The unchanged 400-task cohort observed before this acceptance-rule revision; fixed validation, not an untouched final test.'),
        scope='Five independently trained CV models share one development-selected acceptance rule. No new network updates or task generation.')
    e.save(design, target)
    print(json.dumps(dict(phase='selected_models_frozen', selected=label, policies=list(paths))), flush=True)
    return design


def stability_final_report(args):
    initialize()
    design_path = OUT/'stable9031-selected-design.pth'
    design = e.read(design_path)
    result_path = OUT/'stable9031-results.json'
    result = json.loads(result_path.read_text(encoding='utf8'))
    selection_path = OUT/'stable9031-stability-selection.pth'
    if design['selection_sha256'] != e.digest(selection_path):
        raise ValueError('The acceptance selection changed after final evaluation was frozen.')

    def summary(rows, reference):
        rescue = sum(a['success'] > b['success'] for a, b in zip(rows, reference, strict=True))
        harm = sum(a['success'] < b['success'] for a, b in zip(rows, reference, strict=True))
        return dict(episodes=len(rows), success=sum(row['success'] for row in rows),
            success_percent=100.*sum(row['success'] for row in rows)/len(rows),
            capture=sum(row['capture'] for row in rows), collision=sum(row['collision'] for row in rows),
            interventions=sum(row.get('conditional_interventions', 0) for row in rows),
            rescue=rescue, harm=harm, net_success=rescue-harm,
            gain_percentage_points=100.*(rescue-harm)/len(rows))

    def repeated(runs):
        values = np.asarray([item['success_percent'] for item in runs])
        return dict(runs=len(runs), mean_success_percent=float(values.mean()),
            sample_sd_percentage_points=float(values.std(ddof=1)),
            minimum_success_percent=float(values.min()), maximum_success_percent=float(values.max()),
            mean_gain_percentage_points=float(np.mean([item['gain_percentage_points'] for item in runs])),
            positive_runs=sum(item['net_success'] > 0 for item in runs),
            mean_rescue=float(np.mean([item['rescue'] for item in runs])),
            mean_harm=float(np.mean([item['harm'] for item in runs])),
            mean_collision=float(np.mean([item['collision'] for item in runs])),
            mean_interventions=float(np.mean([item['interventions'] for item in runs])))

    def read_evaluation(path, tasks, expected_sha256=None):
        value = e.read(path)
        policy = checkpoint_path(value['policy'])
        if (not value.get('completed') or value['tasks'] != tasks or len(value['rows']) != len(tasks) or
                value['policy_sha256'] != e.digest(policy) or value['fixed_inference_width'] != 80 or
                (expected_sha256 is not None and value['policy_sha256'] != expected_sha256)):
            raise ValueError('Only complete evaluations of the frozen policies and exact tasks may be reported.')
        return value, dict(path=str(path), sha256=e.digest(path), policy=str(policy), policy_sha256=e.digest(policy))

    cohort_results = {}
    names = ['F11']+[f'cv{seed}' for seed in range(20261051, 20261055)]
    for cohort, spec in design['cohorts'].items():
        source_path = checkpoint_path(spec['path'])
        if e.digest(source_path) != spec['sha256']:
            raise ValueError('A fixed development or validation cohort changed.')
        source = e.read(source_path)
        tasks = source['tasks']
        if cohort == 'development':
            reference = source['baseline_rows']
            baseline_identity = dict(path=str(source_path), sha256=e.digest(source_path),
                                     component='baseline_rows')
            f00_path = OUT/'stable9031-expanded-F00-n1720-evaluation.pth'
            raw_ordinary_path = OUT/'stable9031-expanded-s20261050-ordinary-n1720-evaluation.pth'
        else:
            baseline_value, baseline_identity = read_evaluation(OUT/'stable9031-confirmation-F10-evaluation.pth', tasks)
            reference = baseline_value['rows']
            f00_path = OUT/'stable9031-confirmation-F00-evaluation.pth'
            raw_ordinary_path = OUT/'stable9031-confirmation-ordinary-evaluation.pth'
        evaluations, models = {}, {}
        for name, item in design['policies'].items():
            path = OUT/f'stable9031-selected-{cohort}-{name}-n{len(tasks)}-evaluation.pth'
            value, identity = read_evaluation(path, tasks, item['sha256'])
            evaluations[name] = value
            models[name] = dict(**summary(value['rows'], reference), identity=identity,
                agility={str(agility): summary([row for row, task in zip(value['rows'], tasks, strict=True)
                                               if float(task[0][2]) == agility],
                                              [row for row, task in zip(reference, tasks, strict=True)
                                               if float(task[0][2]) == agility]) for agility in (2., 2.5)})
        f00, f00_identity = read_evaluation(f00_path, tasks)
        raw_ordinary, raw_ordinary_identity = read_evaluation(raw_ordinary_path, tasks)
        models['F00'] = dict(**summary(f00['rows'], reference), identity=f00_identity)
        models['F10'] = dict(**summary(reference, reference), identity=baseline_identity)
        models['ordinary_raw_acceptance'] = dict(**summary(raw_ordinary['rows'], reference), identity=raw_ordinary_identity)
        runs = [dict(seed=seed, **models[name]) for seed, name in zip(range(20261050, 20261055), names, strict=True)]
        ordinary_rows = evaluations['ordinary']['rows']
        versus_ordinary = [dict(seed=seed, **summary(evaluations[name]['rows'], ordinary_rows))
            for seed, name in zip(range(20261050, 20261055), names, strict=True)]
        record = dict(completed=True, task_count=len(tasks), evidence_role=design['evidence_roles'][cohort],
            models=models, repeated_CV_runs=runs, repeated_CV_summary=repeated(runs),
            repeated_CV_vs_ordinary=dict(runs=versus_ordinary, summary=repeated(versus_ordinary)),
            repeated_CV_agility_summary={str(agility): repeated([run['agility'][str(agility)] for run in runs])
                                        for agility in (2., 2.5)},
            mechanism_ablation={name: models[name] for name in ('F00', 'F10', 'F01', 'F11')},
            F01_scope='Execution-only removal of CI; the CV network was trained with CI and is not a separately retrained no-CI method.')
        if cohort == 'development':
            original_start, original_end = source['cohorts']['added320']
            record['cohort_summary'] = {label: repeated([summary(evaluations[name]['rows'][start:end], reference[start:end])
                for name in names]) for label, (start, end) in source['cohorts'].items()}
            record['original_320_runs'] = [dict(seed=seed, **summary(evaluations[name]['rows'][original_start:original_end], reference[original_start:original_end]))
                for seed, name in zip(range(20261050, 20261055), names, strict=True)]
            raw_runs = result['expanded_recipe_runs']
        else:
            independent, independent_identity = read_evaluation(OUT/'stable9031-confirmation-independent-evaluation.pth', tasks)
            record['joint_vs_independent'] = dict(**summary(reference, independent['rows']),
                                                  independent_identity=independent_identity)
            raw_runs = result['independent_confirmation']['repeated_CV_runs']
        record['raw_acceptance_summary'] = repeated(raw_runs)
        record['acceptance_change'] = {key: record['repeated_CV_summary'][key]-record['raw_acceptance_summary'][key]
            for key in ('mean_success_percent', 'mean_rescue', 'mean_harm', 'mean_collision', 'mean_interventions')}
        cohort_results[cohort] = record
    ordinary_fit = e.read(checkpoint_path(design['policies']['ordinary']['path']))['successor_task_fit']
    fits = []
    keys = ['training_observation_sha256', 'training_target_sha256', 'total_parameter_updates',
            'member_training_roots', 'member_factual_trajectories', 'member_updates']
    for seed, name in zip(range(20261050, 20261055), names, strict=True):
        fit = e.read(checkpoint_path(design['policies'][name]['path']))['successor_task_fit']
        if any(fit[key] != ordinary_fit[key] for key in keys):
            raise ValueError('The ordinary control must retain matched observations and training budget.')
        fits.append(dict(seed=seed, matched_training={key: fit[key] for key in keys},
                         trainable_parameters=fit['trainable_parameters'], member_seeds=fit['member_seeds']))
    result['stabilized_selection'] = dict(completed=True, created_at=time.time(),
        selection=e.read(selection_path), design=dict(path=str(design_path), sha256=e.digest(design_path)),
        acceptance_rule=f"mean paired gain - {design['setting']['penalty']} * member gain SD > {design['setting']['margin']}; same rule for CV and ordinary control",
        cohorts=cohort_results, training_repeats=fits, ordinary_trainable_parameters=ordinary_fit['trainable_parameters'],
        representative_seed=20261050, policies=design['policies'],
        unchanged=['CI', 'SAC source', 'Boids', 'network weights after expanded training', 'reward', 'curriculum',
                   'dynamics', 'task success criteria', 'collision criteria', 'intervention execution semantics'],
        interpretation='Repeated CV initialization on fixed expanded development data, followed by one shared acceptance setting and complete fixed-cohort evaluations.',
        implementation_sha256=e.digest(Path(__file__)), runtime_sha256=e.digest(t.OUT/'development-runtime.pth'))
    current = dict(completed=True, result_section='stabilized_selection',
        representative_seed=20261050, representative_policy=design['policies']['F11'],
        training_repeats=5, all_training_repeats_included=True,
        development=cohort_results['development']['repeated_CV_summary'],
        original_320=cohort_results['development']['cohort_summary']['added320'],
        validation=cohort_results['validation']['repeated_CV_summary'],
        validation_CI=cohort_results['validation']['models']['F10']['success_percent'],
        validation_ordinary_same_rule=cohort_results['validation']['models']['ordinary']['success_percent'],
        validation_ordinary_original_rule=cohort_results['validation']['models']['ordinary_raw_acceptance']['success_percent'],
        scope='Six vessels, agility 2.0 and 2.5; fixed 1720-task development and 400-task validation cohorts; CV initialization repeats with a frozen common CI policy.')
    result = dict(current_result=current, **{key: value for key, value in result.items() if key != 'current_result'})
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(phase='selected_results_saved', path=str(result_path),
        summaries={name: value['repeated_CV_summary'] for name, value in cohort_results.items()})), flush=True)


def expanded_report(args):
    initialize()
    design_path = OUT/'stable9031-development-design.pth'
    design = e.read(design_path)
    baseline = design['baseline_rows']
    agility_indices = {str(agility): [i for i, task in enumerate(design['tasks'])
                                    if float(task[0][2]) == agility]
                       for agility in sorted({float(task[0][2]) for task in design['tasks']})}
    def summary(rows, reference):
        rescue = sum(a['success'] > b['success'] for a, b in zip(rows, reference, strict=True))
        harm = sum(a['success'] < b['success'] for a, b in zip(rows, reference, strict=True))
        return dict(episodes=len(rows), success=sum(row['success'] for row in rows),
            success_percent=float(100.*np.mean([row['success'] for row in rows])),
            capture=sum(row['capture'] for row in rows), collision=sum(row['collision'] for row in rows),
            interventions=sum(row.get('conditional_interventions', 0) for row in rows),
            rescue=rescue, harm=harm, net_success=rescue-harm,
            gain_percentage_points=100.*(rescue-harm)/len(rows))
    def evaluation(path, tasks):
        value = e.read(path)
        if not value.get('completed') or value['tasks'] != tasks or len(value['rows']) != len(tasks):
            raise ValueError('A reported repeated evaluation must finish the entire fixed cohort.')
        policy = checkpoint_path(value['policy'])
        if value['policy_sha256'] != e.digest(policy) or value['fixed_inference_width'] != 80:
            raise ValueError('The repeated evaluation must retain its measured model and inference semantics.')
        return value, e.read(policy), dict(path=str(path), sha256=e.digest(path),
                                         policy=str(policy), policy_sha256=e.digest(policy))
    old_tasks = e.read(OUT/'hour-final-design.pth')['tasks']
    old_baseline = e.read(OUT/'hour-fresh-F10-evaluation.pth')['rows']
    old_runs = []
    parent_fit = e.read(checkpoint_path(design['parent_policy']))['successor_task_fit']
    for seed in [20261027, 20261040, 20261041, 20261042, 20261043]:
        path = OUT/('hour-fresh-parent_cv-evaluation.pth' if seed == 20261027 else
                    f'stable9031-original-s{seed}-conditional-n320-evaluation.pth')
        value, model, identity = evaluation(path, old_tasks)
        if any(model['successor_task_fit'][key] != parent_fit[key] for key in
               ('training_observation_sha256', 'training_target_sha256', 'total_parameter_updates',
                'member_training_roots', 'member_factual_trajectories', 'trainable_parameters')):
            raise ValueError('The original recipe repeats must keep the parent data, capacity and training budget.')
        old_runs.append(dict(seed=seed, **summary(value['rows'], old_baseline), identity=identity))
    replay, _, replay_identity = evaluation(OUT/'stable9031-unchanged-replay-n320-evaluation.pth', old_tasks)
    original_rows = e.read(OUT/'hour-fresh-parent_cv-evaluation.pth')['rows']
    replay_comparison = {key: all(a.get(key, 0) == b.get(key, 0)
        for a, b in zip(replay['rows'], original_rows, strict=True))
        for key in ('success', 'capture', 'collision', 'conditional_interventions', 'physical_steps')}
    runs = []
    full_primary = None
    full_primary_identity = None
    for seed in range(20261050, 20261055):
        count = len(baseline)
        path = OUT/f'stable9031-expanded-s{seed}-conditional-n{count}-evaluation.pth'
        value, model, identity = evaluation(path, design['tasks'])
        if seed == 20261050:
            full_primary = value
            full_primary_identity = identity
            start, end = design['cohorts']['added320']
            if design['tasks'][start:end] != old_tasks:
                raise ValueError('The stability cohort must retain every original 90.31-percent task.')
        stability_rows = value['rows']
        fit = model['successor_task_fit']
        control_path = OUT/f'expanded-mlp-joint_likelihood-s{seed}-ordinary-supportk3.pth'
        control = e.read(control_path)['successor_task_fit']
        common = ['training_observation_sha256', 'training_target_sha256', 'total_parameter_updates',
                  'member_training_roots', 'member_factual_trajectories', 'member_updates']
        if any(fit[key] != control[key] for key in common):
            raise ValueError('Expanded ordinary training must receive identical data and update budgets.')
        runs.append(dict(seed=seed, **summary(stability_rows, baseline), identity=identity,
            cohorts={name: summary(value['rows'][start:end], baseline[start:end])
                     for name, (start, end) in design['cohorts'].items()},
            agility={name: summary([value['rows'][i] for i in indices], [baseline[i] for i in indices])
                     for name, indices in agility_indices.items()},
            matched_training={key: fit[key] for key in common},
            training_value_diagnostics=fit['raw_training_diagnostics'],
            conditional_parameters=fit['trainable_parameters'], ordinary_parameters=control['trainable_parameters']))
    control_value, control_model, control_identity = evaluation(
        OUT/f'stable9031-expanded-s20261050-ordinary-n{len(baseline)}-evaluation.pth', design['tasks'])
    ablations = {}
    for label in ('F00', 'F01'):
        value, _, identity = evaluation(OUT/f'stable9031-expanded-{label}-n{len(baseline)}-evaluation.pth', design['tasks'])
        ablations[label] = dict(**summary(value['rows'], baseline), identity=identity)
    independent_rows, independent_sources = [], []
    for cohort, filename in [('added400', 'fresh-independent-evaluation.pth'),
            ('added200', 'risk-fresh-independent-evaluation.pth'), ('added320', 'hour-fresh-independent-evaluation.pth')]:
        start, end = design['cohorts'][cohort]
        value, _, identity = evaluation(OUT/filename, design['tasks'][start:end])
        independent_rows.extend(value['rows']); independent_sources.append(identity)
    if len({item['policy_sha256'] for item in independent_sources}) != 1:
        raise ValueError('Independent-versus-joint fusion requires a single unchanged independent policy.')
    def repeated(group):
        rates = np.asarray([item['success_percent'] for item in group])
        gains = np.asarray([item['gain_percentage_points'] for item in group])
        return dict(runs=len(group), mean_success_percent=float(rates.mean()),
            sample_sd_percentage_points=float(rates.std(ddof=1)),
            minimum_success_percent=float(rates.min()), maximum_success_percent=float(rates.max()),
            mean_gain_percentage_points=float(gains.mean()), positive_runs=int((gains > 0).sum()),
            mean_rescue=float(np.mean([item['rescue'] for item in group])),
            mean_harm=float(np.mean([item['harm'] for item in group])))
    paired_data = e.read(OUT/'stable9031-expanded-paired.pth')
    bank = e.read(OUT/'stable9031-expanded-factual.pth')
    confirmation = None
    confirmation_path = OUT/'stable9031-confirmation-design.pth'
    if confirmation_path.exists():
        confirmation_design = e.read(confirmation_path)
        confirmation_values, confirmation_identities = {}, {}
        for name, policy in confirmation_design['policies'].items():
            path = OUT/f'stable9031-confirmation-{name}-evaluation.pth'
            value, _, identity = evaluation(path, confirmation_design['tasks'])
            if (value.get('design_sha256') != e.digest(confirmation_path) or
                    value['policy_sha256'] != policy['sha256']):
                raise ValueError('Confirmation results must use the frozen tasks and models.')
            confirmation_values[name] = value
            confirmation_identities[name] = identity
        reference = confirmation_values['F10']['rows']
        confirmation_models = {name: dict(**summary(value['rows'], reference),
            identity=confirmation_identities[name],
            agility={str(agility): summary(value['rows'][start:end], reference[start:end])
                     for agility, start, end in [(2., 0, 200), (2.5, 200, 400)]})
            for name, value in confirmation_values.items()}
        confirmation_runs = [dict(seed=seed, **confirmation_models[
            'F11' if seed == 20261050 else f'cv{seed}']) for seed in range(20261050, 20261055)]
        confirmation = dict(completed=True, task_count=len(reference),
            evidence_role=confirmation_design['evidence_role'],
            design=dict(path=str(confirmation_path), sha256=e.digest(confirmation_path)),
            representative_seed=20261050, models=confirmation_models,
            repeated_CV_runs=confirmation_runs, repeated_CV_summary=repeated(confirmation_runs),
            repeated_agility_summary={str(agility): repeated([run['agility'][str(agility)]
                for run in confirmation_runs]) for agility in (2., 2.5)},
            joint_vs_independent=summary(reference, confirmation_values['independent']['rows']),
            uncertainty='The reported sample SD measures training-initialization variation on one shared task cohort.')
    result = dict(completed=True, created_at=time.time(),
        evidence_role=design['evidence_role'], task_count=len(design['tasks']),
        design=dict(path=str(design_path), sha256=e.digest(design_path), cohorts=design['cohorts']),
        baseline_CI=summary(baseline, baseline),
        baseline_by_cohort={name: summary(baseline[start:end], baseline[start:end])
                            for name, (start, end) in design['cohorts'].items()},
        baseline_by_agility={name: summary([baseline[i] for i in indices], [baseline[i] for i in indices])
                            for name, indices in agility_indices.items()},
        unchanged_9031_parent=summary(design['parent_rows'], baseline),
        unchanged_parent_by_cohort={name: summary(design['parent_rows'][start:end], baseline[start:end])
                                   for name, (start, end) in design['cohorts'].items()},
        original_recipe_repeat_320=dict(runs=old_runs, summary=repeated(old_runs),
            role='Repeated independent initializations of the original fixed recipe on the same 320 tasks; this cohort is now development.'),
        unchanged_parent_replay=dict(**summary(replay['rows'], old_baseline), identity=replay_identity,
            per_episode_matches=replay_comparison, role='Same frozen model and same tasks; reproducibility, not independent training evidence.'),
        expanded_recipe_runs=runs, expanded_recipe_summary=repeated(runs),
        independent_confirmation=confirmation,
        repeat_scope='Five independent initializations, all trained and compared on the entire fixed expanded 1720-task development pool.',
        expanded_cohort_summary={name: repeated([item['cohorts'][name] for item in runs])
                                 for name in design['cohorts']},
        expanded_agility_summary={name: repeated([item['agility'][name] for item in runs])
                                 for name in agility_indices},
        expanded_full_pool=dict(seed=20261050, **summary(full_primary['rows'], baseline), identity=full_primary_identity,
            cohorts={name: summary(full_primary['rows'][start:end], baseline[start:end])
                     for name, (start, end) in design['cohorts'].items()},
            role='Complete expanded development evaluation of the run designated before observing its outcomes.'),
        ordinary_control=dict(seed=20261050, **summary(control_value['rows'], baseline), identity=control_identity,
            stability_cohort=summary(control_value['rows'][design['cohorts']['added320'][0]:], old_baseline),
            training_value_diagnostics=control_model['successor_task_fit']['raw_training_diagnostics'],
            role='One fixed ordinary-value control, identical observations and update budget; no control tuning competition.'),
        mechanism_ablation=dict(F00=ablations['F00'], F10=summary(baseline, baseline), F01=ablations['F01'],
            F11=summary(full_primary['rows'], baseline),
            F01_scope='Execution-only removal of CI from the trained CV; no separate no-CI value retraining.'),
        independent_fusion_comparison=dict(episodes=len(independent_rows),
            independent=summary(independent_rows, baseline[design['original800']:]),
            joint_CI=summary(baseline[design['original800']:], independent_rows), sources=independent_sources),
        representative_seed=20261050,
        training_data=dict(path=str(OUT/'stable9031-expanded-paired.pth'),
            sha256=e.digest(OUT/'stable9031-expanded-paired.pth'), roots=len(paired_data['roots']),
            measured_terminal_observations=sum(len(root['outcomes']) for root in paired_data['roots']),
            factual_trajectories=bank['trajectories'], acquisition_selection=design['selection'],
            new_root_location='First accepted native proposal of the unchanged parent; uniform eligible fallback.',
            new_candidate_labels='Actual closed-loop outcomes for the proposal, opposite and duration alternatives under three fresh shared futures.'),
        method=dict(paired_gain='Delta(x,z)=(1-p0(x))*r(x,z)-p0(x)*h(x,z)',
            objective='Unchanged parent joint factual/intervention likelihood; more measured development scenes and proposal states.',
            recipe='Three independently initialized dense members, 36000 total updates per training run, three-neighbor measured candidate support, gain margin 0.02.',
            unchanged=['CI', 'SAC source', 'Boids', 'reward', 'curriculum', 'dynamics', 'success and collision criteria', 'intervention execution semantics']),
        scope='Five runs share one fixed development task pool. Repeated outcomes are not additional independent geometries; the expanded pool is not an unseen final test.',
        prior_results_preserved=['conditional-final-results.json', 'hour-final-results.json'],
        runtime_sha256=e.digest(t.OUT/'development-runtime.pth'), implementation_sha256=e.digest(Path(__file__)))
    output = OUT/'stable9031-results.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(phase='stable9031_results_saved', path=str(output),
                         original=result['original_recipe_repeat_320']['summary'],
                         expanded=result['expanded_recipe_summary'])), flush=True)


def hour_data(local_value, source_path=None):
    """Keep all common futures of a layout on one side of calibration."""
    source = e.read(source_path or OUT/'mixed-paired.pth')
    bank = e.read(OUT/'mixed-factual.pth')
    source, bank = local_value.retain_vessel_identity(source, bank)
    layouts = sorted({tuple(record['scene']) for record in bank['records']})
    rng = np.random.default_rng(10261211)
    order = rng.permutation(len(layouts))
    heldout = {layouts[index] for index in order[:max(1, len(layouts)//5)]}
    training, validation = copy.deepcopy(source), copy.deepcopy(source)
    training['roots'] = [root for root in source['roots'] if tuple(root['scene']) not in heldout]
    validation['roots'] = [root for root in source['roots'] if tuple(root['scene']) in heldout]
    bank['records'] = [record for record in bank['records'] if tuple(record['scene']) not in heldout]
    for data in (training, validation):
        data['measured_terminal_observations'] = sum(len(root['outcomes']) for root in data['roots'])
        data['design_root_count'] = data['next_design_root'] = len(data['roots'])
    bank['trajectories'] = len(bank['records'])
    return training, validation, bank, sorted(heldout)


def hour_train(args):
    """Joint outcome likelihood with direct same-root mean net-gain learning."""
    _, _, local_value, _, _ = initialize()
    import ast
    plain = __import__('plain_return')
    source = e.read(t.OUT/'development-runtime.pth')['runtime_sources']['plain_return.py']
    node = next(node for node in ast.parse(source).body
                if isinstance(node, ast.FunctionDef) and node.name == 'fit_plain_return')
    function = ast.get_source_segment(source, node)
    if args.dataset == 'joint':
        anchor = '    if conditional and not isolate_factual_start:'
        if function.count(anchor) != 1:
            raise ValueError('Cannot locate the local-versus-coupled intervention training rule.')
        function = function.replace(anchor,
            "    if conditional and not isolate_factual_start and not data.get('joint_response_start', False):")
    anchor = "    def draw(count, size, probability):"
    extra = """    paired_mean_target = None
    if conditional and conditional_objective == 'joint_likelihood':
        keys, inverse = np.unique(rows[:, :4], axis=0, return_inverse=True)
        differences = (pooled_target-paired_factual_target).detach().cpu().numpy()
        means = np.bincount(inverse, weights=differences[:len(rows)])/np.bincount(inverse)
        grouped = np.zeros(len(pooled_target), np.float32)
        grouped[:len(rows)] = means[inverse]
        paired_mean_target = torch.as_tensor(grouped, device=device)
"""
    if function.count(anchor) != 1:
        raise ValueError('Cannot locate the paired target construction.')
    function = function.replace(anchor, extra+anchor)
    anchor = "                else:\n                    logits = raw.squeeze(-1)"
    extra = """                    if conditional_objective == 'joint_likelihood':
                        p0 = factual_logits.sigmoid()
                        net_gain = ((1.-p0)*raw[:, 1].sigmoid()-p0*raw[:, 2].sigmoid())*active
                        loss += 4.*((net_gain-paired_mean_target[index]).square()*active).mean()
"""
    if function.count(anchor) != 1:
        raise ValueError('Cannot locate the conditional objective boundary.')
    function = function.replace(anchor, extra+anchor)
    exec(compile(function, '<joint-likelihood-mean-net-gain>', 'exec'), plain.__dict__)
    data_path = OUT/('hour-joint-paired.pth' if args.dataset == 'joint' else 'mixed-paired.pth')
    while not data_path.exists() or not e.read(data_path).get('completed'):
        if time.time() >= args.deadline-1000:
            raise RuntimeError('Complete coupled-response labels did not arrive before the training cutoff.')
        time.sleep(5)
    data, calibration, bank, heldout = hour_data(local_value, data_path)
    name = ('hour-joint-netgain-s' if args.dataset == 'joint' else 'hour-netgain-s')+str(args.seed)
    paired, ordinary = local_value.fit_success_intervention_pair(data,
        OUT/(name+'-conditional.pth'), OUT/(name+'-ordinary.pth'),
        validation_data=calibration, factual_bank=bank, updates=args.updates,
        ensemble_size=3, width=64, seed=args.seed, device=args.device,
        architecture='mlp', conditional_objective='joint_likelihood',
        network_options=dict(identity_conditioned=True, unique_factual_supervision=True))
    contexts = torch.as_tensor(np.asarray([root['context'] for root in data['roots']]), dtype=torch.float32)
    controls = torch.as_tensor(np.asarray([root['effect_controls'] for root in data['roots']]), dtype=torch.float32)
    features = local_value.state_features(contexts, controls).numpy()
    for arm, model in [('conditional', paired), ('ordinary', ordinary)]:
        options = model['successor_task_value']['options']
        lookup = {tuple(option): index for index, option in enumerate(options)}
        observed = np.zeros((len(data['roots']), len(options)), dtype=bool)
        for index, root in enumerate(data['roots']):
            for option, _ in root['outcomes']:
                observed[index, lookup[tuple(root['options'][option])]] = True
        model['candidate_support'] = dict(features=features, observed_options=observed, neighbors=3,
            source_sha256=e.digest(data_path), outcome_labels_used_for_support=False,
            rule='Union of measured candidates at three nearest training roots; factual continuation always allowed.')
        model['net_gain_development'] = dict(seed=args.seed, calibration_layouts=heldout,
            training_roots=len(data['roots']), calibration_roots=len(calibration['roots']),
            training_observations=data['measured_terminal_observations'],
            paired_mean_supervision_weight=4. if arm == 'conditional' else 0.,
            execution_semantics=('CI joint response from the first committed intervention tick' if args.dataset == 'joint'
                                 else 'isolated first intervention tick followed by CI joint response'),
            data_file=str(data_path), source_sha256=e.digest(data_path),
            labels_match_deployed_execution=True)
        if arm == 'conditional':
            model['successor_task_fit']['objective'] += '; plus four times squared error against the same-root candidate mean paired net gain'
        model['evidence_role'] = 'One-hour development with fixed layout-disjoint probability calibration.'
        e.save(model, OUT/(name+'-'+arm+'.pth'))
    print(json.dumps(dict(phase='hour_trained', name=name, updates=args.updates,
                         roots=len(data['roots']), calibration_roots=len(calibration['roots']))), flush=True)


def hour_joint_collect(args):
    _, _, local_value, _, _ = initialize()
    original = e.read(OUT/'mixed-paired-design.pth')
    indices = np.random.default_rng(10261231).permutation(len(original['roots']))[:80]
    indices = list(map(int, indices[args.shard::4]))
    roots = []
    for index in indices:
        root = copy.deepcopy(original['roots'][index])
        root['futures'] = [e.seed32(10261232, index, repetition) for repetition in range(3)]
        root['future_weights'] = np.ones(3, np.float32)
        root['outcomes'] = {(q, future): dict(duration=400)
                            for q in range(len(root['options'])) for future in range(3)}
        roots.append(root)
    design = dict(kind='breadth_local_pairs', roots=len(roots), repetitions=3,
        seed=10261232, planned_observations=sum(len(root['outcomes']) for root in roots),
        design_sha256=hashlib.sha256(repr([(indices[i], root['options'], root['futures'])
            for i, root in enumerate(roots)]).encode()).hexdigest(), inherited_labels=False,
        allocation='80 preselected measured CI states, uniform without replacement; three fresh common futures per root',
        execution='All free vessels solve the unchanged CI joint problem subject to the committed intervention from its first tick.',
        selected_original_root_indices=indices, shard=args.shard)
    blueprint = dict(roots=roots, acquisition_design=design)
    output = OUT/('hour-joint-paired-shard'+str(args.shard)+'.pth')
    data = local_value.recollect_breadth_design(BASE_POLICY, blueprint, output,
        workers=1, inference_batch=80, inference_device=args.device, root_batch_size=10,
        isolate_factual_start=False, physical_response_horizon=5,
        fixed_inference_width=80, response_feature_batch=80)
    data['joint_response_start'] = True
    data['isolate_factual_start'] = False
    e.save(data, output)
    print(json.dumps(dict(phase='joint_response_acquired', shard=args.shard, roots=len(data['roots']),
        observations=sum(len(root['outcomes']) for root in data['roots']))), flush=True)


def hour_joint_merge(args):
    initialize()
    paths = [OUT/('hour-joint-paired-shard'+str(shard)+'.pth') for shard in range(4)]
    while not all(path.exists() and e.read(path).get('completed') and e.read(path).get('joint_response_start')
                  for path in paths):
        if time.time() >= args.deadline-1200:
            raise RuntimeError('Fresh coupled-response acquisition exceeded its development cutoff.')
        time.sleep(5)
    parts = [e.read(path) for path in paths]
    result = copy.deepcopy(parts[0])
    result['roots'] = [root for part in parts for root in part['roots']]
    for index, root in enumerate(result['roots']):
        root['design_root'] = index
    result.update(design_root_count=len(result['roots']), next_design_root=len(result['roots']),
        measured_terminal_observations=sum(len(root['outcomes']) for root in result['roots']),
        training_data=str(OUT/'hour-joint-paired.pth'),
        simulation_steps=sum(part['simulation_steps'] for part in parts),
        acquisition_design=dict(kind='combined_fresh_joint_response', source_sha256=[e.digest(path) for path in paths],
            roots=len(result['roots']), inherited_labels=False))
    e.save(result, OUT/'hour-joint-paired.pth')
    print(json.dumps(dict(phase='joint_response_labels_ready', roots=len(result['roots']),
        observations=result['measured_terminal_observations'])), flush=True)


def load_actor(path, development):
    actor = t.load_actor(path, development)
    specification = e.read(path)
    calibration = specification.get('heldout_probability_calibration')
    if calibration is not None:
        decision = actor.successor_task_value
        is_conditional = specification['successor_task_value']['kind'] == 'paired_mlp_return'
        def calibrated_catalog(context, controls):
            decision.configure_fleet(controls.shape[-2])
            scale = context.new_tensor(calibration['scale'])
            bias = context.new_tensor(calibration['bias'])
            estimates = []
            active = (decision.programs > 0).any(-1)[None]
            for member in [decision.network, *decision.ensemble]:
                raw = decision.network_catalog(member, context, controls)
                if is_conditional:
                    probabilities = (raw*scale+bias).sigmoid()
                    p0 = probabilities[:, :1, 0]
                    gain = ((1.-p0)*probabilities[..., 1]-p0*probabilities[..., 2])*active
                    estimates.append(p0+gain)
                else:
                    estimates.append((raw*scale[0]+bias[0]).sigmoid())
            return decision.aggregate_estimates(estimates)
        decision.catalog_estimates = calibrated_catalog
    risk = specification.get('conditional_risk_utility')
    if risk is not None:
        decision = actor.successor_task_value
        recipe = specification['successor_task_value']
        if recipe.get('conditional_objective') != 'joint_likelihood':
            raise ValueError('Asymmetric harm utility requires the jointly fitted conditional event heads.')
        def asymmetric_catalog(context, controls):
            decision.configure_fleet(controls.shape[-2])
            active = (decision.programs > 0).any(-1)[None]
            estimates = []
            for member in [decision.network, *decision.ensemble]:
                raw = decision.network_catalog(member, context, controls)
                factual = raw[:, :1, 0].sigmoid()
                gain = ((1.-factual)*raw[..., 1].sigmoid()
                        -risk['harm_weight']*factual*raw[..., 2].sigmoid())*active
                estimates.append(factual+gain)
            return decision.aggregate_estimates(estimates)
        decision.catalog_estimates = asymmetric_catalog
    support = specification.get('candidate_support')
    if support is None:
        return actor
    local_value = __import__('local_intervention_value')
    decision = actor.successor_task_value
    decision.register_buffer('support_features', torch.as_tensor(support['features'], dtype=torch.float32))
    decision.register_buffer('support_options', torch.as_tensor(support['observed_options'], dtype=torch.bool))
    native_catalog = decision.catalog_estimates
    def supported_catalog(context, controls):
        ranking, values = native_catalog(context, controls)
        features = local_value.state_features(context, controls)
        distance = (features[:, None]-decision.support_features[None]).square().mean(-1)
        nearest = distance.topk(min(support['neighbors'], distance.shape[1]), largest=False).indices
        allowed = decision.support_options[nearest].any(1)
        allowed[:, 0] = True
        return ranking.masked_fill(~allowed, -torch.inf), values
    decision.catalog_estimates = supported_catalog
    return actor


def hour_calibrate(args):
    _, development, local_value, _, _ = initialize()
    path = Path(args.policy)
    while not path.exists() or 'net_gain_development' not in e.read(path):
        if time.time() >= args.deadline-1200:
            raise RuntimeError('The completed training model did not arrive before calibration cutoff.')
        time.sleep(5)
    model = e.read(path)
    data_path = checkpoint_path(model['net_gain_development'].get('data_file', str(OUT/'mixed-paired.pth')))
    _, validation, _, layouts = hour_data(local_value, data_path)
    _, full_bank = local_value.retain_vessel_identity(e.read(OUT/'mixed-paired.pth'), e.read(OUT/'mixed-factual.pth'))
    selected = set(map(tuple, layouts))
    records = [record for record in full_bank['records'] if tuple(record['scene']) in selected]
    conditional = model['successor_task_value']['kind'] == 'paired_mlp_return'
    decision = load_actor(path, development).successor_task_value.to(args.device)
    members = [decision.network, *decision.ensemble]
    contexts, controls, options, outcomes, factual, group_ids = [], [], [], [], [], []
    for root_index, root in enumerate(validation['roots']):
        for (option, future), outcome in sorted(root['outcomes'].items()):
            if option == 0:
                continue
            contexts.append(root['context']); controls.append(root['effect_controls'])
            options.append(root['options'][option]); outcomes.append(float(outcome['success']))
            factual.append(float(root['outcomes'][0, future]['success']))
            group_ids.append((root_index, option))
    context = torch.as_tensor(np.asarray(contexts), dtype=torch.float32, device=args.device)
    control = torch.as_tensor(np.asarray(controls), dtype=torch.float32, device=args.device)
    option = torch.as_tensor(np.asarray(options), device=args.device)
    programs = option[:, :1]//(10**torch.arange(6, device=args.device)) % 10
    target = torch.as_tensor(outcomes, dtype=torch.float32, device=args.device)
    factual_target = torch.as_tensor(factual, dtype=torch.float32, device=args.device)
    factual_context = torch.as_tensor(np.concatenate([record['contexts'] for record in records]), device=args.device)
    factual_control = torch.as_tensor(np.concatenate([record['controls'] for record in records]), device=args.device)
    bank_target = torch.as_tensor(np.concatenate([record['values'] for record in records]), device=args.device)
    with torch.no_grad():
        raw = torch.stack([member(context, control, programs, option[:, 1:]) for member in members])
        bank_raw = torch.stack([member(factual_context, factual_control,
            torch.zeros((len(bank_target), 6), dtype=torch.long, device=args.device),
            torch.zeros((len(bank_target), 2), device=args.device)) for member in members])
        if conditional:
            reference_raw = torch.stack([member(context, control, torch.zeros_like(programs),
                torch.zeros_like(option[:, 1:]))[:, 0] for member in members])
            bank_raw = bank_raw[..., 0]
    count = 3 if conditional else 1
    log_scale = torch.nn.Parameter(torch.zeros(count, device=args.device))
    bias = torch.nn.Parameter(torch.zeros(count, device=args.device))
    optimizer = torch.optim.Adam([log_scale, bias], lr=.025)
    F = torch.nn.functional
    with torch.enable_grad():
        for step in range(1000):
            optimizer.zero_grad()
            scale = log_scale.exp()
            base = (bank_raw*scale[0]+bias[0]).sigmoid().mean(0)
            loss = F.binary_cross_entropy(base.clamp(1e-6, 1.-1e-6), bank_target)
            if conditional:
                probs = (raw*scale+bias).sigmoid().mean(0)
                event = torch.where(factual_target > .5, probs[:, 2], probs[:, 1])
                event_target = torch.where(factual_target > .5, 1.-target, target)
                loss += F.binary_cross_entropy(event.clamp(1e-6, 1.-1e-6), event_target)
            else:
                probability = (raw*scale[0]+bias[0]).sigmoid().mean(0)
                loss += F.binary_cross_entropy(probability.clamp(1e-6, 1.-1e-6), target)
            loss += .01*(log_scale.square().sum()+bias.square().sum())
            loss.backward(); optimizer.step()
            with torch.no_grad():
                log_scale.clamp_(-2.3, 2.3); bias.clamp_(-4., 4.)
    model['heldout_probability_calibration'] = dict(scale=log_scale.exp().detach().cpu().tolist(),
        bias=bias.detach().cpu().tolist(), updates=1000, layouts=layouts, layout_disjoint_from_training=True,
        factual_trajectories=len(records), factual_state_labels=len(bank_target),
        paired_state_labels=len(target), final_objective=float(loss.detach()), parent_sha256=e.digest(path),
        objective='Regularized affine-logit conditional event calibration plus on-policy factual calibration.'
                  if conditional else 'Regularized affine-logit absolute outcome calibration on the same heldout layouts.')
    target_path = OUT/(path.stem+'-calibrated.pth')
    e.save(model, target_path)
    print(json.dumps(dict(phase='hour_calibrated', path=str(target_path),
        scale=model['heldout_probability_calibration']['scale'], bias=model['heldout_probability_calibration']['bias'])), flush=True)


def hour_variant(args):
    initialize()
    path = Path(args.policy)
    while not path.exists():
        if time.time() >= args.deadline-900:
            raise RuntimeError('The model did not arrive before conservative selection cutoff.')
        time.sleep(5)
    model = e.read(path)
    model['successor_task_value'].update(ensemble_gain_penalty=args.penalty, gain_margin=args.gain_margin)
    model['net_gain_selection'] = dict(ensemble_gain_penalty=args.penalty, gain_margin=args.gain_margin,
        parent_sha256=e.digest(path), additional_observations=0, additional_updates=0,
        rule='Intervene only when mean predicted paired gain minus the ensemble standard deviation penalty exceeds the gain margin; otherwise execute CI.')
    target = OUT/((args.name or path.stem+'-p'+str(args.penalty)+'-m'+str(args.gain_margin))+'.pth')
    if target.exists():
        raise ValueError('The decision candidate already exists.')
    e.save(model, target)
    print(json.dumps(dict(phase='hour_selection_candidate', path=str(target))), flush=True)


def hour_report(args):
    initialize()
    import math
    design_path = Path(args.design)
    design = e.read(design_path)
    def summary(rows):
        return dict(episodes=len(rows), success=sum(row['success'] for row in rows),
            success_percent=100.*np.mean([row['success'] for row in rows]),
            capture=sum(row['capture'] for row in rows), collision=sum(row['collision'] for row in rows),
            interventions=sum(row.get('conditional_interventions', 0) for row in rows))
    def paired(rows, baseline):
        rescue = sum(a['success'] > b['success'] for a, b in zip(rows, baseline, strict=True))
        harm = sum(a['success'] < b['success'] for a, b in zip(rows, baseline, strict=True))
        discordant = rescue+harm
        probability = min(1., 2.*sum(math.comb(discordant, j) for j in range(min(rescue, harm)+1))/2.**discordant)
        return dict(rescue=rescue, harm=harm, net_success=rescue-harm,
                    gain_percentage_points=100.*(rescue-harm)/len(rows), exact_paired_p=probability)
    results, files = {}, {}
    for name, policy in design['policies'].items():
        path = OUT/(args.name+'-'+name+'-evaluation.pth')
        value = e.read(path)
        if (not value['completed'] or value['tasks'] != design['tasks'] or
                value['design_sha256'] != e.digest(design_path) or value['policy_sha256'] != policy['sha256']):
            raise ValueError('Every reported outcome must match the complete frozen paired design.')
        results[name] = value
        files[name] = dict(path=str(path), sha256=e.digest(path))
    comparisons = {}
    for name, value in results.items():
        rows = value['rows']
        comparisons[name] = dict(**summary(rows), versus_CI=paired(rows, results['F10']['rows']),
            agility={str(a): summary([row for row in rows if row['agility'] == a]) for a in (2., 2.5)})
    names = [('selected', 'F11', 'ordinary')]+[(name[:-12], name, name[:-12]+'_ordinary')
        for name in results if name.endswith('_conditional')]
    runs = []
    for label, conditional, ordinary in names:
        model = e.read(checkpoint_path(design['policies'][conditional]['path']))
        control = e.read(checkpoint_path(design['policies'][ordinary]['path']))
        fit, control_fit = model['successor_task_fit'], control['successor_task_fit']
        fields = ['training_observation_sha256', 'training_target_sha256', 'total_parameter_updates',
                  'member_training_roots', 'member_factual_trajectories', 'member_updates']
        if any(fit[field] != control_fit[field] for field in fields):
            raise ValueError('The final value-learning control has a different data or training budget.')
        runs.append(dict(label=label, conditional=conditional, ordinary=ordinary,
            conditional_summary=comparisons[conditional], ordinary_summary=comparisons[ordinary],
            versus_ordinary=paired(results[conditional]['rows'], results[ordinary]['rows']),
            matched_training={field: fit[field] for field in fields},
            conditional_parameters=fit['trainable_parameters'], ordinary_parameters=control_fit['trainable_parameters'],
            development=model.get('net_gain_development'), calibration=model.get('heldout_probability_calibration'),
            selection=model.get('net_gain_selection'),
            factual_state_training_labels=fit.get('additional_factual_state_labels'),
            raw_layout_disjoint_value_diagnostics=fit.get('raw_validation_diagnostics'),
            decision_parameters={key: model['successor_task_value'].get(key, 0.)
                for key in ('gain_margin', 'ensemble_gain_penalty', 'isolate_factual_start')}))
    rates = np.asarray([run['conditional_summary']['success_percent'] for run in runs])
    ordinary_rates = np.asarray([run['ordinary_summary']['success_percent'] for run in runs])
    gains = np.asarray([run['conditional_summary']['versus_CI']['gain_percentage_points'] for run in runs])
    development = []
    previous_ci = e.read(OUT/'final-ci-w80-n800-evaluation.pth')
    reference = {tuple(task[0]): row for task, row in zip(previous_ci['tasks'], previous_ci['rows'], strict=True)}
    for path in sorted(OUT.glob('hour*-evaluation.pth')):
        if path.name.startswith(args.name+'-'):
            continue
        value = e.read(path)
        if not value.get('completed') or not all(tuple(task[0]) in reference for task in value['tasks']):
            continue
        development.append(dict(path=str(path), sha256=e.digest(path), **summary(value['rows']),
            versus_CI=paired(value['rows'], [reference[tuple(task[0])] for task in value['tasks']]),
            evidence_role='Previously observed development tasks; never an unseen final evaluation.'))
    common = e.read(OUT/'hour-joint-netgain-s20261034-conditional-n80-evaluation.pth')['tasks']
    common_keys = {tuple(task[0]) for task in common}
    selection_comparison = {}
    for family, prefix, suffix in [('isolated_raw', 'hour-netgain', ''),
            ('isolated_calibrated', 'hour-netgain', '-calibrated'),
            ('isolated_conservative', 'hour-netgain', '-calibrated-p0.5-m0.1'),
            ('joint_response_raw', 'hour-joint-netgain', ''),
            ('joint_response_calibrated', 'hour-joint-netgain', '-calibrated')]:
        group = []
        for seed in range(20261030, 20261035):
            stem = prefix+'-s'+str(seed)+'-conditional'+suffix
            candidates = [OUT/(stem+'-n80-evaluation.pth'), OUT/(stem+'-n160-evaluation.pth')]
            path = next((path for path in candidates if path.exists()), None)
            if path is None:
                continue
            value = e.read(path)
            chosen = [(row, reference[tuple(task[0])]) for task, row in zip(value['tasks'], value['rows'])
                      if tuple(task[0]) in common_keys]
            group.append(dict(seed=seed, **summary([row for row, _ in chosen]),
                versus_CI=paired([row for row, _ in chosen], [row for _, row in chosen])))
        scores = [item['success'] for item in group]
        selection_comparison[family] = dict(runs=group, mean_success_count=float(np.mean(scores)),
            sample_sd_success_count=float(np.std(scores, ddof=1)),
            total_rescue=sum(item['versus_CI']['rescue'] for item in group),
            total_harm=sum(item['versus_CI']['harm'] for item in group))
    result = dict(created_at=time.time(), design=dict(path=str(design_path), sha256=e.digest(design_path),
        frozen_at=design['frozen_at'], tasks=len(design['tasks']), excluded_layouts=design['excluded_layouts_count']),
        policies=design['policies'], evaluation_files=files, results=comparisons, independent_training_runs=runs,
        five_run_summary=dict(runs=len(runs), conditional_mean_percent=float(rates.mean()),
            conditional_sample_sd_percentage_points=float(rates.std(ddof=1)), ordinary_mean_percent=float(ordinary_rates.mean()),
            ordinary_sample_sd_percentage_points=float(ordinary_rates.std(ddof=1)),
            mean_gain_over_CI_percentage_points=float(gains.mean()), gain_sample_sd_percentage_points=float(gains.std(ddof=1)),
            positive_runs=int((gains > 0).sum()), task_correlation='All runs share the same frozen task cohort; repeated evaluations are not additional independent layouts.'),
        development_candidates=development,
        selection_on_common_80_development_tasks=selection_comparison,
        selection_rule='Maximize mean closed-loop gain over CI across five training runs; resolve equal mean by fewer harms and smaller run variation. Select the representative run by success, then fewer harms. Frozen before all final outcomes.',
        method=dict(paired_gain='Delta(x,z)=(1-p0(x))*r(x,z)-p0(x)*h(x,z)',
            objective='Joint factual/intervention outcome negative log likelihood + 4*mean squared error against the same-root candidate mean paired outcome difference.',
            calibration='Evaluated during development but not selected for the final raw joint-response models. All futures at one layout stay in the same train/calibration split.',
            policy='Choose the supported candidate maximizing mean paired gain minus lambda times member gain standard deviation; intervene only above epsilon; otherwise use CI.',
            selected_policy_parameters=dict(ensemble_members=3, gain_margin=.02, ensemble_gain_penalty=0.,
                candidate_support_neighbors=3, probability_calibration=False),
            joint_response_equation='g*(x,z)=argmin_g J_CI(g;x) subject to the selected committed coefficients g_S=z. The learned outcome refers to this whole conditional joint response.',
            coupled_control='When joint-response execution is selected, every free coefficient minimizes the unchanged CI cost subject to the selected committed intervention, starting at its first tick.',
            unchanged=['SAC source', 'Boids', 'reward', 'curriculum', 'dynamics', 'success and collision criteria', 'CI objective and candidate mechanism'],
            no_reused_labels_for_changed_execution=True,
            execution_only_ablation='F01 removes CI at execution; its CV learner is not retrained without CI.'),
        runtime=dict(private_runtime_sha256=e.digest(t.OUT/'development-runtime.pth'),
                     implementation_sha256=e.digest(Path(__file__))))
    output = OUT/'hour-final-results.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps(dict(phase='hour_results_saved', path=str(output), summary=result['five_run_summary'])), flush=True)


def risk(args):
    initialize()
    path = Path(args.policy)
    model = e.read(path)
    if args.harm_weight < 1.:
        raise ValueError('The conditional risk extension cannot discount a predicted harmful intervention.')
    model['conditional_risk_utility'] = dict(harm_weight=args.harm_weight,
        objective='expected rescue minus weighted expected harm under the learned joint factual/intervention law',
        parent_sha256=e.digest(path), additional_labels=0, additional_parameter_updates=0)
    target = OUT / (path.stem+'-harm'+str(args.harm_weight)+'.pth')
    if target.exists():
        raise ValueError('This conditional risk candidate already exists.')
    e.save(model, target)
    print(json.dumps(dict(phase='conditional_risk_candidate', path=str(target))), flush=True)


def support(args):
    _, _, local_value, _, _ = initialize()
    stem = {'enriched': 'enriched', 'expanded': 'stable9031-expanded'}.get(args.dataset, 'mixed')
    paired_path = OUT / (stem+'-paired.pth')
    bank_path = OUT/('stable9031-expanded-factual.pth' if args.dataset == 'expanded' else 'mixed-factual.pth')
    data, bank = local_value.retain_vessel_identity(e.read(paired_path), e.read(bank_path))
    if not data['completed']:
        raise ValueError('Candidate support needs completely measured acquisition.')
    contexts = torch.as_tensor(np.asarray([root['context'] for root in data['roots']]), dtype=torch.float32)
    controls = torch.as_tensor(np.asarray([root['effect_controls'] for root in data['roots']]), dtype=torch.float32)
    features = local_value.state_features(contexts, controls).numpy()
    path = Path(args.policy)
    model = e.read(path)
    options = model['successor_task_value']['options']
    lookup = {tuple(option): index for index, option in enumerate(options)}
    observed = np.zeros((len(data['roots']), len(options)), dtype=bool)
    for index, root in enumerate(data['roots']):
        for option, _ in root['outcomes']:
            observed[index, lookup[tuple(root['options'][option])]] = True
    model['candidate_support'] = dict(features=features, observed_options=observed,
        neighbors=args.support_neighbors, source_sha256=e.digest(paired_path),
        rule='Rank measured alternatives at nearest public-state roots; always permit factual continuation.',
        outcome_labels_used_for_support=False)
    target = OUT / (path.stem + '-supportk' + str(args.support_neighbors) + '.pth')
    if target.exists():
        raise ValueError('A support-restricted candidate already exists at this path.')
    e.save(model, target)
    print(json.dumps(dict(phase='supported_candidate', path=str(target))), flush=True)


def ensemble(args):
    _, _, local_value, _, _ = initialize()
    paths = [Path(path) for path in args.policies]
    if len(paths) < 2 or not args.name:
        raise ValueError('An ensemble needs named output and at least two separately trained parents.')
    while not all(path.exists() for path in paths):
        if not args.wait_data or time.time() >= args.deadline-1200:
            raise ValueError('Ensemble parents are not ready before the evaluation cutoff.')
        time.sleep(10)
    models = [e.read(path) for path in paths]
    hashes = [e.digest(path) for path in paths]
    if len(set(hashes)) != len(hashes):
        raise ValueError('An ensemble must not duplicate a parent.')
    common = ('ordinary', 'objective', 'training_observation_sha256', 'training_target_sha256',
              'predecessor_sha256', 'roots', 'value_target')
    recipe_fields = ('kind', 'width', 'defenders', 'options', 'gain_margin', 'conditional_objective')
    for model in models[1:]:
        if any(model['successor_task_fit'].get(key) != models[0]['successor_task_fit'].get(key) for key in common):
            raise ValueError('Ensemble members must use the same measured data and learning objective.')
        if any(model['successor_task_value'].get(key) != models[0]['successor_task_value'].get(key) for key in recipe_fields):
            raise ValueError('Ensemble members must use the same representation and decision catalog.')
        for key in ('features', 'observed_options'):
            if not np.array_equal(model['candidate_support'][key], models[0]['candidate_support'][key]):
                raise ValueError('Ensemble members must share the same candidate support.')
    result = copy.deepcopy(models[0])
    states = [state for model in models for state in
              [model['successor_task_value']['state'], *model['successor_task_value'].get('ensemble_states', [])]]
    result['successor_task_value'].update(state=states[0], ensemble_states=states[1:])
    fit = result['successor_task_fit']
    for key in ('updates', 'total_parameter_updates', 'trainable_parameters'):
        fit[key] = sum(model['successor_task_fit'][key] for model in models)
    for key in ('member_updates', 'member_seeds', 'member_training_roots', 'member_factual_trajectories'):
        fit[key] = [item for model in models for item in model['successor_task_fit'].get(key, [])]
    fit.update(ensemble_members=len(states), architecture=f'{len(states)} independently initialized dense members; mean predicted probabilities and effects',
               ensemble_parent_sha256=hashes, raw_training_diagnostics=None, raw_validation_diagnostics=None)
    if 'net_gain_development' in result:
        data, _, _, _ = hour_data(local_value)
        result['net_gain_development']['ensemble_training_runs'] = [
            model['net_gain_development']['seed'] for model in models]
    else:
        data, _ = local_value.retain_vessel_identity(e.read(OUT/'mixed-paired.pth'), e.read(OUT/'mixed-factual.pth'))
    fit['raw_training_diagnostics'] = local_value.raw_paired_value_diagnostics(result, data, device='cpu')
    result['evidence_role'] = 'Development ensemble; both conditional and ordinary arms receive equal member and update budgets.'
    target = OUT / (args.name + '.pth')
    if target.exists():
        raise ValueError('A trained ensemble already exists at the target path.')
    e.save(result, target)
    print(json.dumps(dict(phase='trained_ensemble', path=str(target), members=len(states))), flush=True)


def prepare(manifest):
    target = OUT / 'mixed-design.pth'
    if target.exists():
        return e.read(target)
    old = list(manifest['heldout_tasks'])
    recent = list(e.read(t.OUT / 'fresh-final-design.pth')['tasks'])
    seen = {int(scene[0]) for scene, _ in old + recent}
    extra = []
    position = 0
    while len(extra) < 200:
        seed = e.seed32(10261026, position)
        position += 1
        if seed in seen:
            continue
        seen.add(seed)
        extra.append(((seed, 6, 2. if len(extra) < 100 else 2.5), None))
    design = dict(tasks=old + recent + extra, cohorts=[400, 200, 200],
        evidence_role='development: all included scenes may be used for optimization',
        predecessor=str(BASE_POLICY), predecessor_sha256=e.digest(BASE_POLICY),
        created_unix=time.time())
    e.save(design, target)
    return design


def train(args):
    _, d, l, m, r = initialize()
    if args.architecture == 'interaction' and args.device.startswith('cuda'):
        while torch.cuda.mem_get_info(args.device)[0] < 10 * 1024**3:
            if time.time() > args.deadline-1500:
                raise RuntimeError('Insufficient shared GPU headroom before the training cutoff.')
            time.sleep(15)
    folder = e.BASE / 'local-difference-joint-development'
    if args.dataset == 'historical':
        data = e.read(folder / 'F11-r136-six-trained-paired-data-seed42.pth')
        bank = e.read(folder / 'F11-r136-six-trained-factual-data-seed42.pth')
    else:
        stem = {'enriched': 'enriched', 'expanded': 'stable9031-expanded'}.get(args.dataset, 'mixed')
        paired_path = OUT / (stem+'-paired.pth')
        while True:
            data = e.read(paired_path) if paired_path.exists() else None
            enough = data is not None and (data.get('completed') or args.allow_prefix
                and len(data['roots']) >= max(1, args.max_roots))
            if enough:
                break
            if not args.wait_data or time.time() >= args.deadline-1200:
                raise ValueError('Requested measured training roots are not ready.')
            time.sleep(15)
        bank = e.read(OUT / ('stable9031-expanded-factual.pth' if args.dataset == 'expanded' else 'mixed-factual.pth'))
        if not bank['completed'] or not data['completed'] and not args.allow_prefix:
            raise ValueError('Train on completed measured acquisition.')
        if args.max_roots:
            data = copy.deepcopy(data)
            data['roots'] = data['roots'][:args.max_roots]
            data['training_subset'] = dict(actual_roots=len(data['roots']),
                selection='ordered prefix of completely measured roots; no missing outcomes filled')
    data, bank = l.retain_vessel_identity(data, bank)
    name = args.dataset + '-' + args.architecture + '-' + args.objective
    if args.seed != 20261027:
        name += '-s' + str(args.seed)
    network_options = (dict(identity_conditioned=True, finite_motion_basis=True)
                       if args.architecture == 'interaction' else dict(identity_conditioned=True))
    if args.physical_baseline:
        if args.architecture != 'mlp':
            raise ValueError('The physical baseline is a shared scalar dense representation.')
        name += '-physical'
        network_options.update(action_encoding='physical_delta', state_baseline=True)
    if args.anchor_factual:
        if args.architecture != 'mlp' or args.objective != 'paired_catalog_contrast':
            raise ValueError('Factual anchoring is defined for the shared dense contrast objective.')
        name += '-anchored'
        plain = __import__('plain_return')
        native_groups = plain.single_vessel_contrast_groups
        def anchored_groups(rows, identities, defenders):
            references = {tuple(identity): index for index, (row, identity) in
                enumerate(zip(rows, identities, strict=True)) if row[1] == 0}
            groups = native_groups(rows, identities, defenders)
            result = []
            for indices in groups:
                programs = np.asarray(rows)[indices, 1].astype(np.int64)
                changed = (programs[:, None]//(10**np.arange(defenders))) % 10 != 0
                if changed.any(0).sum() == 1:
                    reference = references[tuple(identities[int(indices[0])])]
                    indices = np.r_[reference, indices]
                result.append(indices)
            return result
        plain.single_vessel_contrast_groups = anchored_groups
    if args.max_roots:
        name += '-r' + str(len(data['roots']))
    if args.decision_regret and args.architecture != 'interaction':
        raise ValueError('The decision objective here is implemented for the interaction learner.')
    if args.decision_regret:
        name += '-decision'
    native_fit = l.fit_joint_potential
    def decision_fit(*positional, **settings):
        if not settings.get('ordinary', False):
            settings['decision_regret_weight'] = args.decision_regret
        return native_fit(*positional, **settings)
    if args.decision_regret:
        l.fit_joint_potential = decision_fit
    paired, ordinary = l.fit_success_intervention_pair(data,
        OUT / (name + '-conditional.pth'), OUT / (name + '-ordinary.pth'),
        factual_bank=bank, updates=args.updates, ensemble_size=3, width=64,
        seed=args.seed, device=args.device, architecture=args.architecture,
        conditional_objective=args.objective,
        network_options=network_options)
    l.fit_joint_potential = native_fit
    paired['successor_task_fit']['decision_regret_weight'] = args.decision_regret
    if args.anchor_factual:
        plain.single_vessel_contrast_groups = native_groups
        paired['successor_task_fit']['conditional_factual_anchor'] = dict(
            objective='include the observed factual outcome in each same-future single-vessel contrast group',
            no_additional_terminal_observations=True)
    for kind, model in [('conditional', paired), ('ordinary', ordinary)]:
        model['evidence_role'] = 'mixed-scene algorithm development; not an independent test'
        model['development_training_subset'] = data.get('training_subset')
        e.save(model, OUT / (name + '-' + kind + '.pth'))
    print(json.dumps(dict(phase='trained', name=name, updates=args.updates)), flush=True)


def collect(args):
    manifest, d, l, m, r = initialize()
    design = prepare(manifest)
    bank = l.collect_onpolicy_factual_roots(BASE_POLICY, design['tasks'],
        OUT / 'mixed-factual.pth', root_sample_seed=9261027,
        inference_batch=100, inference_device=args.device)
    blueprint_path = OUT / 'mixed-paired-design.pth'
    if blueprint_path.exists():
        blueprint = e.read(blueprint_path)
    else:
        rng = np.random.default_rng(10261028)
        selected = []
        for begin, end in [(0, 400), (400, 600), (600, 800)]:
            failures = [i for i in range(begin, end) if not bank['records'][i]['values'][0]]
            successes = [i for i in range(begin, end) if bank['records'][i]['values'][0]]
            rng.shuffle(failures)
            rng.shuffle(successes)
            selected.extend((failures + successes)[:32])
        roots = []
        for index in selected:
            record = bank['records'][index]
            roots.append(dict(scene=record['scene'], prefix_future=record['future'],
                root_steps=record['root_snapshot']['root_steps'],
                root_snapshot=record['root_snapshot']))
        candidates = [(digit * 10**boat, delay, span) for boat in range(6)
                      for digit in range(1, 10) for delay, span in [(0, 1), (0, 5), (0, 20), (5, 20)]]
        blueprint = l.breadth_local_design(dict(roots=roots), candidates,
            pairs_per_root=6, repetitions=4, seed=10261029, vessel_balanced=True)
        blueprint['selected_task_indices'] = selected
        e.save(blueprint, blueprint_path)
    l.recollect_breadth_design(BASE_POLICY, blueprint, OUT / 'mixed-paired.pth',
        workers=1, inference_batch=168, inference_device=args.device,
        root_batch_size=8, isolate_factual_start=True, physical_response_horizon=5)


def collect_critical(args):
    manifest, d, l, m, r = initialize()
    blueprint_path = OUT / 'critical-paired-design.pth'
    if blueprint_path.exists():
        blueprint = e.read(blueprint_path)
    else:
        model = e.read(OUT / 'mixed-memory-r68.pth')
        snapshots = {}
        for path in [t.OUT/'failure-roots.pth', OUT/'failure-roots.pth',
                     OUT/'new-generated/failure-roots.pth']:
            snapshots.update(e.read(path))
        tasks = prepare(manifest)['tasks']
        coefficients = [(-1., -1.)] + [(a, b) for a in (0., .5, 1.) for b in (0., .5, 1.)]
        roots, signatures = [], []
        conditional = __import__('conditional_fusion_value')
        for index, record in enumerate(model['empirical_interventions']):
            task = tasks[record['development_task_index']]
            key = (int(task[0][0]), int(record['root_steps']))
            snapshot = copy.deepcopy(snapshots[key])
            snapshot.update(predecessor_sha256=e.digest(BASE_POLICY),
                selection_reason='all retained development rescue states; fresh outcomes measured below',
                eligible_states=None, root_sample_seed=None, inference_batch=None, inference_device=args.device)
            packet = d.mechanism_tensor(d.joint_packet(snapshot['environment'], snapshot['observation']))
            _, order = conditional.intervention_task_context(packet, True, True)
            preferences = np.asarray(record['preferences'])[order[0].cpu().numpy()]
            digits = [coefficients.index(tuple(map(float, pair))) for pair in preferences]
            changed = [i for i, digit in enumerate(digits) if digit]
            if len(changed) != 1:
                raise ValueError('Critical paired acquisition requires a recorded single-vessel intervention.')
            boat = changed[0]
            chosen = digits[boat]
            alternative = max((j for j in range(1, 10) if j != chosen),
                key=lambda j: sum((a-b)**2 for a, b in zip(coefficients[j], coefficients[chosen])))
            span = int(record['span'])
            options = [(0, 0, 0), (chosen*10**boat, 0, span), (alternative*10**boat, 0, span)]
            futures = [e.seed32(10261114, index, repetition) for repetition in range(3)]
            root = dict(scene=tuple(task[0]), prefix_future=task[1], root_steps=key[1],
                root_snapshot=snapshot, options=options, futures=futures, future_weights=np.ones(3, np.float32),
                outcomes={(q, f): dict(duration=400) for q in range(3) for f in range(3)},
                blueprint_root=index)
            roots.append(root)
            signatures.append((tuple(task[0]), key[1], tuple(options), tuple(futures)))
        design = dict(kind='breadth_local_pairs', roots=len(roots), repetitions=3,
            seed=10261114, maximum_observations_per_root=9, planned_observations=9*len(roots),
            inherited_labels=False, shared_by_all_estimators=True,
            allocation='all 68 observed rescue roots; proposed and opposite same-vessel programs; three fresh common futures',
            design_sha256=hashlib.sha256(repr(signatures).encode()).hexdigest(),
            source_sha256=e.digest(OUT/'mixed-memory-r68.pth'))
        blueprint = dict(roots=roots, acquisition_design=design)
        e.save(blueprint, blueprint_path)
    critical_path = OUT/'critical-paired.pth'
    critical = l.recollect_breadth_design(BASE_POLICY, blueprint, critical_path,
        workers=1, inference_batch=144, inference_device=args.device, root_batch_size=16,
        isolate_factual_start=True, physical_response_horizon=5, fixed_inference_width=80,
        response_feature_batch=80)
    if critical is None:
        critical = e.read(critical_path)
    original = e.read(OUT/'mixed-paired.pth')
    if not critical['completed'] or not original['completed']:
        raise ValueError('Only fully measured acquisitions can enter the combined training set.')
    combined = copy.deepcopy(original)
    combined['roots'].extend(critical['roots'])
    combined['simulation_steps'] += critical['simulation_steps']
    combined['next_design_root'] = len(combined['roots'])
    combined['design_root_count'] = len(combined['roots'])
    combined['measured_terminal_observations'] = sum(len(root['outcomes']) for root in combined['roots'])
    combined['training_data'] = str(OUT/'enriched-paired.pth')
    combined['acquisition_design'] = dict(kind='combined_completed_measured_development',
        source_sha256=[e.digest(OUT/'mixed-paired.pth'), e.digest(critical_path)],
        roots=len(combined['roots']), inherited_labels=False)
    e.save(combined, OUT/'enriched-paired.pth')
    print(json.dumps(dict(phase='critical_training_data_ready', roots=len(combined['roots']))), flush=True)


def evaluate(args):
    manifest, d, l, m, r = initialize()
    design = e.read(Path(args.design)) if args.design else prepare(manifest)
    policy = Path(args.policy)
    tasks = design['tasks']
    if args.count:
        unit = args.count//8
        indices = [start+j for start, size in [(0, 2*unit), (200, 2*unit),
            (400, unit), (500, unit), (600, unit), (700, unit)] for j in range(size)]
        tasks = [tasks[i] for i in indices]
    else:
        indices = list(range(len(tasks)))
    name = args.name or policy.stem
    target = OUT / (name + f'-n{len(tasks)}-evaluation.pth')
    if target.exists() and e.read(target).get('completed'):
        print(json.dumps(dict(phase='cached', name=name, success=e.read(target)['success'])), flush=True)
        return
    started = time.monotonic()
    actor = load_actor(policy, d)
    rows = r.batched_credit_rows(policy, tasks, batch_size=args.batch,
        device=args.device, actor_override=actor, fixed_inference_width=args.fixed_width)
    result = dict(name=name, tasks=tasks, indices=indices, rows=rows, completed=True,
        success=sum(row['success'] for row in rows), policy=str(policy),
        policy_sha256=e.digest(policy), evidence_role=getattr(args, 'evidence_role', None) or design.get('evidence_role', design.get('role')),
        fixed_inference_width=args.fixed_width,
        elapsed_seconds=time.monotonic()-started)
    e.save(result, target)
    print(json.dumps({key: result[key] for key in ['name', 'success', 'elapsed_seconds']}), flush=True)


def tune(args):
    """Apply the same acceptance grid to both fully trained value learners."""
    initialize()
    stem = args.name or 'mixed-mlp-paired_catalog_contrast-r32'
    while not all((OUT / (stem + '-' + arm + '.pth')).exists()
                  for arm in ['conditional', 'ordinary']):
        if not args.wait_data or time.time() > args.deadline-1500:
            raise ValueError('The matched trained pair is not ready.')
        time.sleep(15)
    settings = ([(0., .02, None), (0., .10, None), (0., .02, 20)]
                if args.candidate_grid == 'core' else [(0., .10, None), (.5, .05, None), (1., .02, None)])
    for penalty, margin, maximum_span in settings:
        for arm in ['conditional', 'ordinary']:
            model = e.read(OUT / (stem + '-' + arm + '.pth'))
            evaluated_stem = stem
            reusable = OUT / (stem.replace('-anchored', '') + '-ordinary.pth')
            if arm == 'ordinary' and '-anchored' in stem and reusable.exists():
                previous = e.read(reusable)
                fit, old_fit = model['successor_task_fit'], previous['successor_task_fit']
                same_fit = all(fit[key] == old_fit[key] for key in
                    ['updates', 'training_observation_sha256', 'training_target_sha256'])
                same_parameters = (fit['raw_training_diagnostics']['evaluated_parameter_sha256'] ==
                    old_fit['raw_training_diagnostics']['evaluated_parameter_sha256'])
                if same_fit and same_parameters:
                    model, evaluated_stem = previous, stem.replace('-anchored', '')
            model['successor_task_value'].update(ensemble_gain_penalty=penalty, gain_margin=margin)
            if maximum_span is not None:
                options = model['successor_task_value']['options']
                model['successor_task_value']['options'] = [z for z in options
                    if int(z[1]) == 0 and int(z[2]) <= maximum_span]
            model['development_acceptance_setting'] = dict(penalty=penalty, margin=margin,
                immediate_maximum_span=maximum_span,
                selection='same closed-loop development grid for both trained arms')
            name = evaluated_stem + '-' + arm + '-p' + str(penalty) + '-m' + str(margin)
            if maximum_span is not None:
                name += '-immediate' + str(maximum_span)
            path = OUT / (name + '.pth')
            if not path.exists():
                e.save(model, path)
            setting = copy.copy(args)
            setting.policy = str(path)
            setting.name = name
            evaluate(setting)


def freeze_memory(args):
    initialize()
    data = e.read(OUT / 'mixed-intervention-search.pth')
    records = copy.deepcopy(data['records'])
    extra_path = OUT / 'new-generated-intervention-search.pth'
    if extra_path.exists():
        records.extend(copy.deepcopy(e.read(extra_path)['records']))
    model = e.read(BASE_POLICY)
    model['empirical_interventions'] = records
    model['development_subset_interventions'] = True
    model['empirical_training_sha256'] = e.digest(OUT / 'mixed-intervention-search.pth')
    model['empirical_value_definition'] = dict(
        estimator='observed-state paired development improvement',
        support_squared_distance=1e-6,
        action='observed successful actuator-fusion program with delay and duration',
        role='overfit development; previously optimized tasks are not unseen tests')
    model['evidence_role'] = 'mixed development scenes; exact-state supported empirical interventions'
    if extra_path.exists():
        model['extra_empirical_training_sha256'] = e.digest(extra_path)
    target = OUT / f'mixed-memory-r{len(records)}.pth'
    if not target.exists():
        e.save(model, target)
    print(json.dumps(dict(policy=str(target), sha256=e.digest(target), records=len(records))), flush=True)


def prepare_final(args):
    manifest, d, l, m, r = initialize()
    target = OUT / ((args.name + '-design.pth') if args.name else 'mixed-fresh-final-design.pth')
    if target.exists():
        raise ValueError('The final model selection and independent scenarios are already frozen.')
    if args.final_count < 2 or args.final_count % 2:
        raise ValueError('The final paired design needs an even count for balanced agility strata.')
    paired = Path(args.policy)
    ordinary = Path(args.ordinary_policy)
    ablated = e.read(paired)
    ablated['mechanism_ablation'] = (True, False, True)
    ablated['development_ablation'] = 'remove CI at execution; value learner is not retrained'
    ablated['ablation_parent_sha256'] = e.digest(paired)
    ablated_path = OUT / (paired.stem + '-without-ci.pth')
    e.save(ablated, ablated_path)
    used = {int(task[0][0]) for task in prepare(manifest)['tasks']}
    folder = e.BASE / 'local-difference-joint-development'
    historical = e.read(folder / 'F11-r136-six-trained-paired-data-seed42.pth')
    factual = e.read(folder / 'F11-r136-six-trained-factual-data-seed42.pth')
    used.update(int(root['scene'][0]) for root in historical['roots'])
    used.update(int(root['scene'][0]) for root in factual['records'])
    for previous_path in OUT.glob('*design.pth'):
        previous = e.read(previous_path)
        used.update(int(task[0][0]) for task in previous.get('tasks', []))
    tasks, position = [], 0
    while len(tasks) < args.final_count:
        seed = e.seed32(10261138, position)
        position += 1
        if seed in used:
            continue
        used.add(seed)
        tasks.append(((seed, 6, 2. if len(tasks) < args.final_count//2 else 2.5), None))
    paths = dict(F00=t.OUT/'common-source-f00.pth', F10=BASE_POLICY,
        F01=ablated_path, F11=paired, ordinary=ordinary,
        independent=t.OUT/'best-independent-400.pth')
    if args.memory_policy:
        paths['development_memory'] = Path(args.memory_policy)
    if args.additional_ordinary_policy:
        paths['ordinary_catalog'] = Path(args.additional_ordinary_policy)
    if args.reference_policy:
        paths['parent_cv'] = Path(args.reference_policy)
    for label, conditional_path, ordinary_path in args.sensitivity_pair:
        if not label.replace('_', '').isalnum():
            raise ValueError('A sensitivity label may contain only letters, numbers, and underscores.')
        paths[label + '_conditional'] = Path(conditional_path)
        paths[label + '_ordinary'] = Path(ordinary_path)
    policies = {name: dict(path=str(path), sha256=e.digest(path)) for name, path in paths.items()}
    e.save(dict(tasks=tasks, policies=policies, frozen_at=time.time(),
        selection='Observed development comparisons against CI and a fully trained ordinary value model; prioritize positive paired task gains, then success and fewer collisions or harmful interventions; previously viewed new-scene outcomes count as development; final outcomes unused',
        evidence_role='new independent paired scenarios after model freeze',
        excluded_layouts_count=len(used)-len(tasks), fixed_inference_width=80), target)
    print(json.dumps(dict(phase='final_design_frozen', policies=list(policies), count=len(tasks))), flush=True)


def fresh(args):
    manifest, d, l, m, r = initialize()
    target = Path(args.design) if args.design else OUT / 'mixed-fresh-final-design.pth'
    design = e.read(target)
    for name in args.policies:
        item = design['policies'][name]
        path = checkpoint_path(item['path'])
        if e.digest(path) != item['sha256']:
            raise ValueError('A model changed after the final selection was frozen.')
        output = OUT / ((args.name + '-' if args.name else 'fresh-') + name + '-evaluation.pth')
        if output.exists():
            saved = e.read(output)
            if saved.get('policy_sha256') != item['sha256'] or saved.get('design_sha256') != e.digest(target):
                raise ValueError('An existing evaluation belongs to a different frozen policy or task design.')
            if saved.get('completed'):
                continue
        started = time.monotonic()
        rows = r.batched_credit_rows(path, design['tasks'], batch_size=args.batch,
            device=args.device, actor_override=load_actor(path, d), fixed_inference_width=80)
        result = dict(policy=str(path), policy_sha256=item['sha256'],
            design_sha256=e.digest(target), tasks=design['tasks'], rows=rows,
            success=sum(row['success'] for row in rows), completed=True,
            elapsed_seconds=time.monotonic()-started, evidence_role=design['evidence_role'],
            fixed_inference_width=80)
        e.save(result, output)
        print(json.dumps(dict(phase='independent_final', name=name, success=result['success'],
            episodes=len(rows), elapsed=result['elapsed_seconds'])), flush=True)


def rescue_extra(args):
    manifest, d, l, m, r = initialize()
    design = prepare(manifest)
    bank_path = OUT / 'mixed-factual.pth'
    while True:
        if time.time() >= args.deadline-1000:
            return
        if bank_path.exists():
            bank = e.read(bank_path)
            if bank.get('completed'):
                break
        time.sleep(15)
    base = dict(policy=str(BASE_POLICY), rows=[dict(success=int(row['values'][0]),
        physical_steps=row['physical_steps']) for row in bank['records']])
    manifest['heldout_tasks'] = design['tasks']
    indices = [i for i in range(600, 800) if not base['rows'][i]['success']]
    original_out = t.OUT
    t.OUT = OUT / 'new-generated'
    t.OUT.mkdir(exist_ok=True)
    try:
        roots = t.capture_roots(base, manifest, d, r, indices, args.batch, args.device)
    finally:
        t.OUT = original_out
    path = OUT / 'new-generated-intervention-search.pth'
    data = e.read(path) if path.exists() else dict(records=[], rescued=[], measured=[],
        paired_groups=[], completed=False, role='newly generated scenes used for development')
    rescued = set(data['rescued'])
    actor = t.load_actor(BASE_POLICY, d)
    coefficients = [(0., 0.), (1., 1.), (0., 1.), (1., 0.), (.5, .5),
                    (0., .5), (.5, 0.), (1., .5), (.5, 1.)]
    for mode, span in [('late60', 30), ('half', 1), ('initial', 1), ('late120', 30),
                       ('half', 5), ('initial', 30), ('late180', 30), ('half', 20),
                       ('quarter', 5), ('initial', 60), ('half', 60)]:
        for index in indices:
            seed = design['tasks'][index][0][0]
            if seed in rescued:
                continue
            terminal = base['rows'][index]['physical_steps']
            step = (max(0, terminal-int(mode[4:])) if mode.startswith('late') else
                    5*(terminal//10) if mode == 'half' else
                    5*(terminal//20) if mode == 'quarter' else 0)
            identity = (seed, step, span)
            if identity in data['measured'] or (seed, step) not in roots:
                continue
            if time.time() >= args.deadline-1000:
                e.save(data, path)
                return
            snapshot = roots[(seed, step)]
            actor.to(args.device)
            packet = {key: value.to(args.device) for key, value in
                      d.mechanism_tensor(d.joint_packet(snapshot['environment'], snapshot['observation'])).items()}
            with torch.no_grad():
                control = actor(packet, deterministic=True, gate_only=True)[0][0].cpu().numpy()[:, :3]
            one = np.ones((len(control), 1), np.float32)
            jobs, preferences = [], []
            for boat in range(len(control)):
                for pair in coefficients:
                    preference = np.full((len(control), 2), -1., np.float32)
                    preference[boat] = pair
                    action = np.concatenate((control, -one, span*one, -2*one,
                        np.zeros((len(control), 2), np.float32), np.zeros_like(one),
                        preference, preference), -1)
                    jobs.append(dict(environment=snapshot['environment'], observation=snapshot['observation'],
                        random_state=snapshot['random_state'], action=action, isolate_factual_start=True))
                    preferences.append(preference)
            rows = r.batched_credit_rows(BASE_POLICY, [snapshot['task']]*len(jobs),
                batch_size=len(jobs), device=args.device, initial_jobs=jobs, actor_override=actor)
            good = [j for j, row in enumerate(rows) if row['success']]
            data['measured'].append(identity)
            if good:
                chosen = min(good, key=lambda j: rows[j]['duration'])
                data['records'].append(dict(features=roots[(seed, 0)]['features'],
                    root_features=snapshot['features'], preferences=preferences[chosen], delay=step,
                    span=span, development_task_index=index, root_steps=step,
                    measured_row=rows[chosen], factual_success=0, intervention_success=1))
                rescued.add(seed)
                data['rescued'] = sorted(rescued)
            data['paired_groups'].append(dict(root_features=snapshot['features'], task=snapshot['task'],
                root_steps=step, span=span, preferences=preferences, factual_success=0,
                rows=rows, isolate_factual_start=True))
            e.save(data, path)
            print(json.dumps(dict(phase='new_generated_rescue', task=index, step=step,
                span=span, successful_options=len(good), rescued=len(rescued), failures=len(indices))), flush=True)
    data['completed'] = True
    e.save(data, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=['journal_prepare', 'journal_analysis_prepare', 'journal_ordinary_prepare', 'journal_ordinary_select', 'journal_report', 'expanded_prepare', 'expanded_collect', 'expanded_merge', 'expanded_ablation', 'expanded_confirmation_prepare', 'stability_prepare', 'stability_select', 'stability_final_prepare', 'stability_final_report', 'expanded_report', 'hour_train', 'hour_calibrate', 'hour_variant', 'hour_joint_collect', 'hour_joint_merge', 'hour_report', 'train', 'collect', 'collect_critical', 'evaluate', 'tune', 'freeze_memory', 'support', 'ensemble', 'risk',
        'prepare_final', 'fresh', 'rescue_extra'])
    parser.add_argument('--dataset', default='historical', choices=['historical', 'mixed', 'enriched', 'joint', 'expanded'])
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--objective', default='marginal_events', choices=['marginal_events', 'factual_strata',
        'shared_potential', 'paired_ordering', 'paired_catalog_contrast', 'event_hierarchy', 'joint_likelihood'])
    parser.add_argument('--architecture', default='interaction', choices=['interaction', 'mlp'])
    parser.add_argument('--updates', type=int, default=18000)
    parser.add_argument('--seed', type=int, default=20261027)
    parser.add_argument('--decision-regret', type=float, default=0.)
    parser.add_argument('--anchor-factual', action='store_true')
    parser.add_argument('--physical-baseline', action='store_true')
    parser.add_argument('--support-neighbors', type=int, default=3)
    parser.add_argument('--harm-weight', type=float, default=1.)
    parser.add_argument('--penalty', type=float, default=0.)
    parser.add_argument('--gain-margin', type=float, default=.02)
    parser.add_argument('--max-roots', type=int, default=0)
    parser.add_argument('--allow-prefix', action='store_true')
    parser.add_argument('--wait-data', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--policy', default=str(BASE_POLICY))
    parser.add_argument('--ordinary-policy')
    parser.add_argument('--additional-ordinary-policy')
    parser.add_argument('--reference-policy')
    parser.add_argument('--sensitivity-pair', action='append', nargs=3, default=[],
                        metavar=('LABEL', 'CONDITIONAL', 'ORDINARY'))
    parser.add_argument('--memory-policy')
    parser.add_argument('--policies', nargs='*', default=[])
    parser.add_argument('--name')
    parser.add_argument('--design')
    parser.add_argument('--evidence-role')
    parser.add_argument('--scheme-sha256')
    parser.add_argument('--candidate-grid', choices=['margin', 'core'], default='margin')
    parser.add_argument('--count', type=int, default=0)
    parser.add_argument('--final-count', type=int, default=400)
    parser.add_argument('--batch', type=int, default=80)
    parser.add_argument('--fixed-width', type=int, default=0)
    parser.add_argument('--deadline', type=float, default=1793014696.)
    args = parser.parse_args()
    globals()[args.operation](args)


if __name__ == '__main__':
    main()
