"""Paired net-benefit learning and conservative intervention improvement."""
from __future__ import annotations
import argparse
import ast
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import conditional_value_search_1700 as previous

e = previous.e
t = previous.t
PRIOR = previous.OUT
OUT = e.BASE / 'conditional-policy-improvement-20261026-1918'
BASE = t.OUT / 'best-m1-400.pth'


def initialize():
    state = previous.initialize()
    OUT.mkdir(exist_ok=True)
    return state


def install_gain_objective(manifest, gain_weight, harm_weight, rank_weight=0.):
    plain = sys.modules['plain_return']
    source = manifest['runtime_sources']['plain_return.py']
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == 'fit_plain_return')
    function = ast.get_source_segment(source, node)
    if rank_weight:
        future_marker = "if conditional and conditional_objective == 'paired_catalog_contrast':\n                common_future_ids.append((root_id, future))"
        if function.count(future_marker) != 1:
            raise RuntimeError('The common-future row constructor changed.')
        function = function.replace(future_marker, 'if conditional:\n                common_future_ids.append((root_id, future))')
    marker = '    def draw(count, size, probability):'
    target_code = '''    grouped_gain_target = None
    if conditional:
        differences = (pooled_target-paired_factual_target).detach().cpu().numpy()
        groups = {}
        for row_index, row in enumerate(rows):
            groups.setdefault(tuple(map(int, row[:4])), []).append(row_index)
        averaged = np.zeros(len(pooled_target), np.float32)
        for ids in groups.values():
            averaged[ids] = differences[ids].mean()
        grouped_gain_target = torch.as_tensor(averaged, device=device)
'''
    if rank_weight:
        target_code += '''    ranking_pairs = []
    ranking_gaps = []
    ranking_mass = []
    if conditional:
        root_options = {}
        for key, ids in groups.items():
            root_options.setdefault(key[0], []).append(ids)
        observed = pooled_target.detach().cpu().numpy()
        for choices in root_options.values():
            first_pair = len(ranking_pairs)
            for position, left in enumerate(choices):
                left_by_future = {tuple(common_future_ids[i]): i for i in left}
                for right in choices[position+1:]:
                    right_by_future = {tuple(common_future_ids[i]): i for i in right}
                    shared = sorted(left_by_future.keys() & right_by_future.keys())
                    if not shared:
                        continue
                    gap = float(np.mean([observed[left_by_future[f]]-observed[right_by_future[f]] for f in shared]))
                    if abs(gap) < 1e-7:
                        continue
                    pair = [left[0], right[0]] if gap > 0. else [right[0], left[0]]
                    ranking_pairs.append(pair); ranking_gaps.append(abs(gap))
            rank_pair_count = len(ranking_pairs)-first_pair
            if rank_pair_count:
                ranking_mass.extend([1./rank_pair_count]*rank_pair_count)
        if not ranking_pairs:
            raise ValueError('Ranking supervision requires unequal measured outcomes on common futures.')
        ranking_pairs = torch.as_tensor(ranking_pairs, device=device)
        ranking_gaps = torch.as_tensor(ranking_gaps, device=device)
        ranking_probability = np.asarray(ranking_mass)/sum(ranking_mass)
        ranking_rng = np.random.default_rng(np.random.SeedSequence([seed, 11261925]))
'''
    if function.count(marker) != 1:
        raise RuntimeError('The frozen paired-target constructor changed.')
    function = function.replace(marker, target_code + marker)
    marker = '''                            loss = net.joint_outcome_loss(factual_logits, raw, paired_factual_target[index],
                                                         pooled_target[index], active)'''
    auxiliary = f'''
                        factual_probability = factual_logits.sigmoid()
                        predicted_gain = ((1.-factual_probability)*raw[:, 1].sigmoid()
                                          -factual_probability*raw[:, 2].sigmoid())
                        measured_gain = grouped_gain_target[index]
                        loss += {float(gain_weight)!r}*((predicted_gain-measured_gain).square()*active).mean()
                        loss += {float(harm_weight)!r}*(torch.relu(predicted_gain+.02).square()
                            *(measured_gain < 0.)*active).mean()'''
    if rank_weight:
        auxiliary += f'''
                        selected_pairs = torch.as_tensor(ranking_rng.choice(len(ranking_pairs), size=32,
                            p=ranking_probability), device=device)
                        pair_indices = ranking_pairs[selected_pairs]
                        pair_raw = net.forward_features(pooled_features[pair_indices.flatten()]).reshape(32, 2, 3)
                        pair_factual = net.forward_features(pooled_reference[pair_indices[:, 0]])[:, 0].sigmoid()[:, None]
                        pair_gain = ((1.-pair_factual)*pair_raw[..., 1].sigmoid()
                            -pair_factual*pair_raw[..., 2].sigmoid())*paired_active[pair_indices]
                        loss += {float(rank_weight)!r}*torch.relu(.25*ranking_gaps[selected_pairs]
                            -(pair_gain[:, 0]-pair_gain[:, 1])).square().mean()'''
    if function.count(marker) != 1:
        raise RuntimeError('The frozen joint-likelihood objective changed.')
    function = function.replace(marker, marker + auxiliary)
    exec(compile(function, '<paired-net-benefit-objective>', 'exec'), plain.__dict__)
    return hashlib.sha256(function.encode()).hexdigest()


def support(model, data, local):
    contexts = torch.as_tensor(np.asarray([r['context'] for r in data['roots']]), dtype=torch.float32)
    controls = torch.as_tensor(np.asarray([r['effect_controls'] for r in data['roots']]), dtype=torch.float32)
    options = model['successor_task_value']['options']
    lookup = {tuple(option): i for i, option in enumerate(options)}
    observed = np.zeros((len(data['roots']), len(options)), bool)
    for index, root in enumerate(data['roots']):
        for option, future in root['outcomes']:
            observed[index, lookup[tuple(root['options'][option])]] = True
    model['candidate_support'] = dict(features=local.state_features(contexts, controls).numpy(),
        observed_options=observed, neighbors=3, outcome_labels_used_for_support=False,
        rule='Measured candidate support at the three nearest public-state roots; factual is always admissible.')


def load_actor(path, development):
    actor = previous.load_actor(path, development)
    model = e.read(Path(path))
    calibration = model.get('independent_layout_calibration')
    factual_floor = model.get('factual_success_floor')
    if calibration is None and factual_floor is None:
        return actor
    ordinary = model['successor_task_fit']['ordinary']
    if calibration is None:
        calibration = dict(scale=np.ones(1 if ordinary else 3), bias=np.zeros(1 if ordinary else 3))
    decision = actor.successor_task_value
    decision.register_buffer('calibration_scale', torch.as_tensor(calibration['scale'], dtype=torch.float32))
    decision.register_buffer('calibration_bias', torch.as_tensor(calibration['bias'], dtype=torch.float32))
    if 'mixing' in calibration:
        decision.register_buffer('calibration_mixing', torch.as_tensor(calibration['mixing'], dtype=torch.float32))
    local = sys.modules['local_intervention_value']
    support_specification = model.get('candidate_support')

    def calibrated_catalog(context, controls):
        decision.configure_fleet(controls.shape[-2])
        estimates = []
        for member in [decision.network, *decision.ensemble]:
            raw = decision.network_catalog(member, context, controls)
            if ordinary:
                calibrated = raw*decision.calibration_scale[0]+decision.calibration_bias[0]
                if hasattr(decision, 'calibration_mixing'):
                    calibrated = calibrated+decision.calibration_mixing[0]*(raw[:, :1]-raw)
                probability = calibrated.sigmoid()
                if factual_floor is not None:
                    probability = torch.cat((probability[:, :1].clamp_min(factual_floor), probability[:, 1:]), -1)
                estimates.append(probability)
            else:
                raw = torch.cat((raw[:, :1, :1].expand(-1, raw.shape[1], -1), raw[..., 1:]), -1)
                calibrated = raw*decision.calibration_scale+decision.calibration_bias
                if hasattr(decision, 'calibration_mixing'):
                    calibrated = calibrated+torch.nn.functional.linear(raw, decision.calibration_mixing)
                probability = calibrated.sigmoid()
                factual = probability[:, :1, 0]
                if factual_floor is not None:
                    factual = factual.clamp_min(factual_floor)
                active = (decision.programs > 0).any(-1)[None]
                gain = ((1.-factual)*probability[..., 1]-factual*probability[..., 2])*active
                estimates.append(factual+gain)
        ranking, values = decision.aggregate_estimates(estimates)
        if support_specification is not None:
            features = local.state_features(context, controls)
            distance = (features[:, None]-decision.support_features[None]).square().mean(-1)
            nearest = distance.topk(min(support_specification['neighbors'], distance.shape[1]), largest=False).indices
            allowed = decision.support_options[nearest].any(1)
            allowed[:, 0] = True
            ranking = ranking.masked_fill(~allowed, -torch.inf)
        return ranking, values

    decision.catalog_estimates = calibrated_catalog
    return actor


def calibrate(args):
    manifest, d, local, m, r = initialize()
    path = Path(args.policy)
    model = e.read(path)
    data_path = OUT/'proposal-paired.pth'
    data, _ = local.retain_vessel_identity(e.read(data_path))
    if not data['completed']:
        raise ValueError('Calibration needs completed actual paired continuations.')
    layouts = {tuple(root['scene']) for root in data['roots']}
    training_layouts = {tuple(scene) for scene in model['successor_task_fit']['training_layouts']}
    if layouts & training_layouts:
        raise ValueError('Calibration geometry must be absent from the network training observations.')
    bank = e.read(OUT/'proposal-factual.pth')
    stratum = {tuple(record['scene']): bool(record['values'][0]) for record in bank['records']}
    population = {value: sum(label == value for label in stratum.values()) for value in (False, True)}
    sampled = {value: sum(stratum[tuple(root['scene'])] == value for root in data['roots'])
               for value in (False, True)}
    actor = previous.load_actor(path, d).to(args.device)
    decision = actor.successor_task_value
    ordinary = model['successor_task_fit']['ordinary']
    joint_calibration = args.calibration_mode == 'joint'
    members = [decision.network, *decision.ensemble]
    observed_logits = [[] for member in members]
    factual, outcome, active, weights = [], [], [], []
    with torch.no_grad():
        for root in data['roots']:
            options = torch.as_tensor(root['options'], device=args.device)
            programs = (options[:, :1]//(10**torch.arange(6, device=args.device))) % 10
            contexts = torch.as_tensor(root['context'], dtype=torch.float32, device=args.device)[None].expand(len(options), -1)
            controls = torch.as_tensor(root['effect_controls'], dtype=torch.float32, device=args.device)[None].expand(len(options), -1, -1)
            raw_members = [member(contexts, controls, programs, options[:, 1:]).reshape(len(options), -1)
                           for member in members]
            for (option, future), row in root['outcomes'].items():
                y0 = float(root['outcomes'][(0, future)]['success'])
                for index, raw in enumerate(raw_members):
                    logits = raw[option].clone()
                    if not ordinary:
                        logits[0] = raw[0, 0]
                    elif joint_calibration:
                        logits = torch.cat((logits, raw[0]))
                    observed_logits[index].append(logits)
                factual.append(y0); outcome.append(float(row['success']))
                active.append(bool(root['options'][option][0]))
                group = stratum[tuple(root['scene'])]
                weights.append(population[group]/sampled[group]/len(root['outcomes']))
    logits = torch.stack([torch.stack(values) for values in observed_logits]).detach()
    y0 = torch.as_tensor(factual, device=args.device)
    yz = torch.as_tensor(outcome, device=args.device)
    changed = torch.as_tensor(active, device=args.device)
    weight = torch.as_tensor(weights, device=args.device)
    weight /= weight.sum()
    width = 1 if ordinary else 3
    log_scale = torch.zeros(width, device=args.device, requires_grad=True)
    bias = torch.zeros(width, device=args.device, requires_grad=True)
    mixing = torch.zeros(1 if ordinary else (3, 3), device=args.device, requires_grad=True)
    mixing_mask = None if ordinary else torch.tensor([[0., 0., 0.], [1., 0., 1.], [1., 1., 0.]], device=args.device)
    optimizer = torch.optim.Adam([log_scale, bias]+([mixing] if joint_calibration else []), lr=.03)
    previous_grad_mode = torch.is_grad_enabled()
    torch.set_grad_enabled(True)
    for update in range(args.calibration_updates):
        calibrated = logits[..., :width]*log_scale.exp()+bias
        if joint_calibration:
            if ordinary:
                calibrated = calibrated+mixing[0]*(logits[..., 1:]-logits[..., :1])
            else:
                calibrated = calibrated+torch.nn.functional.linear(logits, mixing*mixing_mask)
        probability = calibrated.sigmoid()
        if ordinary:
            predicted = probability[..., 0].mean(0).clamp(1e-6, 1.-1e-6)
            loss = (torch.nn.functional.binary_cross_entropy(predicted, yz, reduction='none')*weight).sum()
        else:
            p0, rescue, harm = probability.unbind(-1)
            factual_probability = torch.where(y0[None] > .5, p0, 1.-p0)
            event_probability = torch.where(y0[None] > .5,
                torch.where(yz[None] > .5, 1.-harm, harm),
                torch.where(yz[None] > .5, rescue, 1.-rescue))
            joint_probability = (factual_probability*torch.where(changed[None], event_probability, 1.)).mean(0)
            loss = -(weight*joint_probability.clamp_min(1e-6).log()).sum()
        loss += .02*(log_scale.square().mean()+.04*bias.square().mean())
        if joint_calibration:
            loss += .1*mixing.square().mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        with torch.no_grad():
            log_scale.clamp_(-2.3, 2.3); bias.clamp_(-4., 4.)
            mixing.clamp_(-2., 2.)
    torch.set_grad_enabled(previous_grad_mode)
    model['independent_layout_calibration'] = dict(scale=log_scale.exp().detach().cpu().numpy(),
        bias=bias.detach().cpu().numpy(), parent_sha256=e.digest(path),
        data_sha256=e.digest(data_path), layouts=sorted(layouts), training_layout_overlap=0,
        updates=args.calibration_updates, observations=len(yz), root_balanced=True,
        selection_probability_corrected=True, stratum_population=population, stratum_sample=sampled,
        objective='absolute Bernoulli likelihood' if ordinary else 'paired factual/intervention joint likelihood',
        calibrated_confidence_bound=False)
    if joint_calibration:
        model['independent_layout_calibration'].update(mode='joint',
            mixing=(mixing if ordinary else mixing*mixing_mask).detach().cpu().numpy())
    target = OUT/((args.name or path.stem+'-calibrated')+'.pth')
    if target.exists():
        raise ValueError('A calibrated model is immutable.')
    e.save(model, target)
    print(json.dumps(dict(phase='independent_layout_calibration', path=str(target),
        scale=log_scale.exp().detach().cpu().tolist(), bias=bias.detach().cpu().tolist(),
        observations=len(yz), layouts=len(layouts))), flush=True)


def train(args):
    manifest, d, local, m, r = initialize()
    path = OUT/'combined-paired.pth' if args.dataset == 'combined' else PRIOR/'enriched-paired.pth'
    bank_path = OUT/'combined-factual.pth' if args.dataset == 'combined' else PRIOR/'mixed-factual.pth'
    while not path.exists() or not bank_path.exists():
        if not args.wait_data:
            raise FileNotFoundError(path)
        time.sleep(10)
    data, bank = e.read(path), e.read(bank_path)
    if not data['completed'] or not bank['completed']:
        raise ValueError('Training requires completely measured observations.')
    data, bank = local.retain_vessel_identity(data, bank)
    code_sha = install_gain_objective(manifest, args.gain_weight, args.harm_weight, args.rank_weight)
    stem = args.name or f'{args.dataset}-gain{args.gain_weight:g}-harm{args.harm_weight:g}-seed{args.seed}'
    targets = [OUT/(stem+'-'+arm+'.pth') for arm in ('conditional', 'ordinary')]
    if any(path.exists() for path in targets):
        raise ValueError('A trained pair must have a new immutable identity.')
    models = local.fit_success_intervention_pair(data, *targets, factual_bank=bank,
        updates=args.updates, ensemble_size=3, width=64, seed=args.seed,
        device=args.device, architecture='mlp', conditional_objective='joint_likelihood',
        network_options=dict(identity_conditioned=True, action_encoding=args.encoding))
    for arm, model, target in zip(('conditional', 'ordinary'), models, targets):
        support(model, data, local)
        model['paired_policy_improvement'] = dict(training_data_sha256=e.digest(path),
            factual_data_sha256=e.digest(bank_path), seed=args.seed, objective_source_sha256=code_sha,
            gain_weight=args.gain_weight if arm == 'conditional' else 0.,
            false_positive_harm_weight=args.harm_weight if arm == 'conditional' else 0.,
            common_future_ranking_weight=args.rank_weight if arm == 'conditional' else 0.,
            target='mean same-root same-program common-future terminal difference',
            role='development fit; independent evaluation is saved separately')
        if arm == 'conditional':
            model['successor_task_fit']['objective'] += '; mean paired net-benefit regression and false-positive harm penalty'
        e.save(model, target)
    print(json.dumps(dict(phase='matched_training_complete', stem=stem, roots=len(data['roots']),
        updates=args.updates, paths=[str(p) for p in targets])), flush=True)


def make_candidate(args):
    initialize()
    path = Path(args.policy)
    model = e.read(path)
    model['successor_task_value']['ensemble_gain_penalty'] = args.beta
    model['successor_task_value']['gain_margin'] = args.margin
    if args.factual_floor is not None:
        if not 0. <= args.factual_floor <= 1.:
            raise ValueError('A factual probability floor must lie in [0,1].')
        model['factual_success_floor'] = args.factual_floor
    model['conservative_policy_improvement'] = dict(parent_sha256=e.digest(path),
        beta=args.beta, margin=args.margin,
        rule='mean paired gain minus beta times ensemble gain standard deviation',
        calibrated_confidence_bound=False, matched_ordinary_execution_rule=True)
    target = OUT/((args.name or path.stem+f'-b{args.beta:g}-m{args.margin:g}')+'.pth')
    if target.exists():
        raise ValueError('A decision candidate is immutable.')
    e.save(model, target)
    print(json.dumps(dict(phase='candidate_frozen', path=str(target), sha256=e.digest(target))), flush=True)


def ensemble(args):
    initialize()
    paths = [Path(p) for p in args.policies]
    models = [e.read(p) for p in paths]
    for model in models[1:]:
        for key in ('training_observation_sha256', 'training_target_sha256', 'predecessor_sha256', 'ordinary'):
            if model['successor_task_fit'][key] != models[0]['successor_task_fit'][key]:
                raise ValueError('Ensemble members must retain the matched data and estimator.')
        if model['successor_task_value']['options'] != models[0]['successor_task_value']['options']:
            raise ValueError('Ensemble candidate catalogs differ.')
    model = copy.deepcopy(models[0])
    states = [s for item in models for s in
              [item['successor_task_value']['state'], *item['successor_task_value'].get('ensemble_states', [])]]
    model['successor_task_value'].update(state=states[0], ensemble_states=states[1:])
    for key in ('updates', 'total_parameter_updates', 'trainable_parameters'):
        model['successor_task_fit'][key] = sum(item['successor_task_fit'][key] for item in models)
    for key, value in list(model['successor_task_fit'].items()):
        if key.startswith('member_') and isinstance(value, list):
            model['successor_task_fit'][key] = [entry for item in models
                for entry in item['successor_task_fit'][key]]
    model['successor_task_fit'].update(ensemble_members=len(states),
        architecture=f'{len(states)} independently initialized dense members; mean predicted probabilities and effects',
        ensemble_training_seeds=[item['paired_policy_improvement']['seed'] for item in models],
        ensemble_parent_sha256=[e.digest(path) for path in paths])
    target = OUT/(args.name+'.pth')
    if target.exists():
        raise ValueError('An ensemble is immutable.')
    e.save(model, target)
    print(json.dumps(dict(path=str(target), members=len(states))), flush=True)


def collect(args):
    manifest, d, local, m, r = initialize()
    tasks = e.read(PRIOR/'joint-final-new200-design.pth')['tasks']
    # These previously inspected scenes are now explicitly development data.
    paths = [PRIOR/f'mixed-mlp-joint_likelihood-seed20261029-supportk3-{arm}.pth'
             for arm in ('conditional', 'ordinary')]
    actors = [previous.load_actor(path, d).to(args.device) for path in paths]
    selector = local.FactualProposalRootSelector([actor.successor_task_value for actor in actors],
        [e.digest(path) for path in paths], fixed_inference_width=80)
    bank = local.collect_onpolicy_factual_roots(BASE, tasks, OUT/'proposal-factual.pth',
        root_sample_seed=11261918, inference_batch=200, inference_device=args.device,
        root_selector=selector, fixed_inference_width=80)
    design_path = OUT/'proposal-paired-design.pth'
    if design_path.exists():
        design = e.read(design_path)
    else:
        rng = np.random.default_rng(11261919)
        failures = [i for i, record in enumerate(bank['records']) if not record['values'][0]]
        successes = [i for i, record in enumerate(bank['records']) if record['values'][0]]
        rng.shuffle(failures); rng.shuffle(successes)
        indices = (failures[:32]+successes)[:80]
        roots = []
        for index in indices:
            record = bank['records'][index]
            snapshot = copy.deepcopy(record['root_snapshot'])
            proposed = snapshot.get('proposal_observation', {}).get('options', [])
            options = {(0, 0, 0)}
            for option in proposed:
                option = tuple(map(int, option))
                if option[0]:
                    options.add(option)
                    digits = [(option[0]//(10**boat)) % 10 for boat in range(6)]
                    boat = next(j for j, digit in enumerate(digits) if digit)
                    for digit in (1, 3, 7, 9):
                        options.add((digit*10**boat, option[1], option[2]))
            if len(options) == 1:
                boat = int(rng.integers(6))
                options.update((digit*10**boat, 0, 5) for digit in (1, 3, 7, 9))
            options = [(0, 0, 0)]+sorted(options-{(0, 0, 0)})
            futures = [e.seed32(11261920, index, repeat) for repeat in range(3)]
            roots.append(dict(scene=tuple(record['scene']), prefix_future=record['future'],
                root_steps=snapshot['root_steps'], root_snapshot=snapshot, options=options,
                futures=futures, future_weights=np.ones(3, np.float32), blueprint_root=index,
                outcomes={(q, f):dict(duration=400) for q in range(len(options)) for f in range(3)}))
        acquisition = dict(kind='breadth_local_pairs', roots=len(roots), repetitions=3,
            seed=11261920, planned_observations=sum(len(root['outcomes']) for root in roots),
            inherited_labels=False, shared_by_all_estimators=True,
            root_selection='failure and success states at first CV-or-ordinary proposed intervention',
            evidence_role='previously inspected geometry reused only for development')
        design = dict(roots=roots, acquisition_design=acquisition, selected_task_indices=indices)
        e.save(design, design_path)
    data = local.recollect_breadth_design(BASE, design, OUT/'proposal-paired.pth',
        workers=1, inference_batch=240, inference_device=args.device, root_batch_size=8,
        isolate_factual_start=True, physical_response_horizon=5, fixed_inference_width=80,
        response_feature_batch=80)
    if data is None:
        data = e.read(OUT/'proposal-paired.pth')
    original = e.read(PRIOR/'enriched-paired.pth')
    combined = local.merge_paired_data([original, data], allow_distinct_contexts=True)
    combined.update(completed=True, training_data=str(OUT/'combined-paired.pth'))
    e.save(combined, OUT/'combined-paired.pth')
    old_bank = e.read(PRIOR/'mixed-factual.pth')
    result = copy.deepcopy(old_bank)
    result['records'].extend(bank['records'])
    for key in ('simulation_steps', 'trajectories', 'state_labels'):
        result[key] = old_bank.get(key, 0)+bank.get(key, 0)
    result.update(completed=True, acquisition_sources=[e.digest(PRIOR/'mixed-factual.pth'),
        e.digest(OUT/'proposal-factual.pth')], tasks=old_bank['tasks']+bank['tasks'])
    e.save(result, OUT/'combined-factual.pth')
    print(json.dumps(dict(phase='proposal_acquisition_complete', roots=len(combined['roots']),
        trajectories=len(result['records']))), flush=True)


def design(args):
    initialize()
    used = {int(scene[0]) for scene, _ in e.read(PRIOR/'mixed-design.pth')['tasks']}
    used.update(int(scene[0]) for scene, _ in e.read(PRIOR/'joint-final-new200-design.pth')['tasks'])
    tasks = []
    index = 0
    while len(tasks) < args.count:
        seed = e.seed32(112619210, index); index += 1
        if seed in used:
            continue
        used.add(seed)
        tasks.append(((seed, 6, 2. if len(tasks) < args.count//2 else 2.5), None))
    policies = {str(Path(path)):e.digest(Path(path)) for path in args.policies}
    result = dict(tasks=tasks, policies=policies, frozen_unix=time.time(),
        driver_sha256=e.digest(Path(__file__)),
        selection='All checkpoints and decision settings fixed using development outcomes before this evaluation.',
        role='New geometry after fixing algorithms and model checkpoints; no tuning on these outcomes.')
    target = OUT/(args.name+'.pth')
    if target.exists():
        raise ValueError('The independent evaluation design is immutable.')
    e.save(result, target)
    print(json.dumps(dict(path=str(target), tasks=len(tasks), policies=len(policies))), flush=True)


def evaluate(args):
    manifest, d, local, m, r = initialize()
    policy = Path(args.policy)
    if args.design:
        design_path = Path(args.design)
        specification = e.read(design_path)
        if specification.get('driver_sha256', e.digest(Path(__file__))) != e.digest(Path(__file__)):
            raise ValueError('The independent evaluation requires its frozen decision implementation.')
        if specification['policies'].get(str(policy)) != e.digest(policy):
            raise ValueError('Independent evaluation must use a frozen listed policy.')
        tasks = specification['tasks']
        role = specification['role']
    else:
        tasks = e.read(PRIOR/'joint-final-new200-design.pth')['tasks']
        if args.count != 200:
            tasks = [tasks[int(index)] for index in np.linspace(0, 199, args.count, dtype=int)]
        role = 'Previously inspected scenes reused for development and decision selection.'
    target = OUT/((args.name or policy.stem)+f'-n{len(tasks)}-evaluation.pth')
    if target.exists():
        result = e.read(target)
        if result['policy_sha256'] != e.digest(policy) or result['tasks'] != tasks:
            raise ValueError('A result identity cannot be reused for another policy or tasks.')
        if result['completed']:
            print(json.dumps(dict(cached=True, success=result['success'], path=str(target))), flush=True)
            return
    started = time.monotonic()
    rows = r.batched_credit_rows(policy, tasks, batch_size=args.batch, device=args.device,
        actor_override=load_actor(policy, d), fixed_inference_width=80)
    result = dict(policy=str(policy), policy_sha256=e.digest(policy), tasks=tasks, rows=rows,
        success=sum(row['success'] for row in rows), completed=True, role=role,
        fixed_inference_width=80, elapsed_seconds=time.monotonic()-started)
    e.save(result, target)
    print(json.dumps(dict(path=str(target), success=result['success'], count=len(rows),
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('train', 'candidate', 'ensemble', 'collect', 'design', 'evaluate', 'calibrate'))
    parser.add_argument('--name')
    parser.add_argument('--policy')
    parser.add_argument('--policies', nargs='+', default=[])
    parser.add_argument('--design')
    parser.add_argument('--dataset', choices=('enriched', 'combined'), default='enriched')
    parser.add_argument('--updates', type=int, default=18000)
    parser.add_argument('--calibration-updates', type=int, default=500)
    parser.add_argument('--calibration-mode', choices=('diagonal', 'joint'), default='diagonal')
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--gain-weight', type=float, default=8.)
    parser.add_argument('--harm-weight', type=float, default=4.)
    parser.add_argument('--rank-weight', type=float, default=0.)
    parser.add_argument('--encoding', choices=('categorical', 'physical_delta'), default='categorical')
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--margin', type=float, default=.02)
    parser.add_argument('--factual-floor', type=float)
    parser.add_argument('--count', type=int, default=80)
    parser.add_argument('--batch', type=int, default=200)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--wait-data', action='store_true')
    args = parser.parse_args()
    {'train':train, 'candidate':make_candidate, 'ensemble':ensemble, 'collect':collect,
     'design':design, 'evaluate':evaluate, 'calibrate':calibrate}[args.operation](args)


if __name__ == '__main__':
    main()
