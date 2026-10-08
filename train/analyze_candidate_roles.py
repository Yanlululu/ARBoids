"""Apply the locked task, paired uncertainty, and preservation criteria once."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta


def binary_lower(delta, confidence=.95):
    n, wins, losses = len(delta), int((delta > 0).sum()), int((delta < 0).sum())
    tail = (1.-confidence)/2.
    lower = 0. if wins == 0 else float(beta.ppf(tail, wins, n-wins+1))
    upper = 1. if losses == n else float(beta.ppf(1.-tail, losses+1, n-losses))
    return dict(wins=wins, losses=losses, difference=float(delta.mean()),
                conservative_lower=lower-upper, noninferior_3pp=lower-upper >= -.03)


def analyze(root):
    result = json.loads((root/'results.json').read_text())
    protocol = json.loads((root/'protocol.json').read_text())
    if not result['complete'] or protocol['stage'] not in ('independent_locked_confirmation', 'independent_interaction_replication'):
        raise ValueError('A completed locked evaluation is required.')
    if result.get('protocol') != protocol:
        raise ValueError('Result and frozen protocol disagree.')
    for name, expected in protocol['files'].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            # Completed runs retain their original source. Subsequent research
            # edits must neither invalidate those runs nor silently replace it.
            frozen = protocol.get('source_text', {}).get(name)
            encodings = [] if frozen is None else [frozen.encode(), frozen.replace('\n', '\r\n').encode()]
            if not any(hashlib.sha256(source).hexdigest() == expected for source in encodings):
                raise ValueError('Frozen input cannot be verified: '+name)
    rows, count = result['episodes'], protocol['count']
    expected = {(method, n, a, bank+i) for method in protocol['methods']
                for n, a, bank in protocol['conditions'] for i in range(count)}
    actual = [(r['method'], r['defenders'], r['agility'], r['scene_seed']) for r in rows]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError('Missing, duplicate, or unexpected evaluation episodes.')
    primary, references = protocol['primary_method'], protocol['primary_references']
    index = {(r['scene_seed'], r['method']): r for r in rows}
    strata = [np.array([bank+i for i in range(count)]) for _, _, bank in protocol['conditions']]
    seeds = np.concatenate(strata)
    intervals, contrasts, table = {}, {}, {}
    for method in protocol['methods']:
        group = [index[int(s), method] for s in seeds]
        table[method] = dict(episodes=len(group), capture_time=float(np.mean([r['capture_time'] for r in group])),
            **{key: int(sum(r[key] for r in group)) for key in ('capture', 'success', 'collision', 'breach', 'timeout')})
    alpha = protocol['familywise_alpha']/len(references)
    rng = np.random.default_rng(20261008)
    draws = [rng.integers(0, len(s), size=(protocol['bootstrap_repetitions'], len(s))) for s in strata]
    for reference in references:
        difference, ratio, boot_a, boot_b = [], [], [], []
        for s, take in zip(strata, draws):
            a = np.array([index[int(seed), primary]['capture_time'] for seed in s])
            b = np.array([index[int(seed), reference]['capture_time'] for seed in s])
            boot_a.append(a[take].mean(1))
            boot_b.append(b[take].mean(1))
        boot_a, boot_b = np.mean(boot_a, axis=0), np.mean(boot_b, axis=0)
        ci = np.quantile(boot_a-boot_b, [alpha/2., 1.-alpha/2.]).tolist()
        improvement_ci = np.quantile(1.-boot_a/boot_b, [alpha/2., 1.-alpha/2.]).tolist()
        preservation = {metric: binary_lower(np.array([
            index[int(seed), primary][metric]-index[int(seed), reference][metric] for seed in seeds]))
            for metric in ('capture', 'success')}
        a, b = table[primary], table[reference]
        improvement = 1.-a['capture_time']/b['capture_time']
        passed = (improvement >= protocol.get('minimum_time_improvement', .10) and ci[1] < 0. and a['collision'] <= b['collision']
                  and a['success'] >= b['success'] and all(v['noninferior_3pp'] for v in preservation.values()))
        contrasts[reference] = dict(time_difference=a['capture_time']-b['capture_time'],
            time_interval=ci, improvement=improvement, improvement_interval=improvement_ci,
            preservation=preservation, passed=bool(passed))
    return dict(complete=True, criteria_passed=all(c['passed'] for c in contrasts.values()),
        primary_population=protocol['primary_population'], table=table, contrasts=contrasts,
        time_interval_confidence=1.-alpha, binary_one_sided_confidence=.95,
        results_sha256=hashlib.sha256((root/'results.json').read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((root/'protocol.json').read_bytes()).hexdigest(),
        scope='Task and matched predictive interaction evidence; learned-policy and runtime claims remain separate.')


def analyze_linked_strata(root):
    """Keep same-offset strata together when their initial/future seeds overlap."""
    original = analyze(root)
    result = json.loads((root/'results.json').read_text())
    protocol = result['protocol']
    design_path = root/'linked-strata-protocol.json'
    design = json.loads(design_path.read_text())
    if design['conditions'] != protocol['conditions'] or design['count'] != protocol['count']:
        raise ValueError('Linked-strata design does not match the frozen evaluation.')
    count, strata = protocol['count'], len(protocol['conditions'])
    index = {(r['scene_seed'], r['method']): r for r in result['episodes']}
    primary = protocol['primary_method']
    alpha = protocol['familywise_alpha']/len(protocol['primary_references'])
    draws = np.random.default_rng(20261008).integers(0, count,
        size=(protocol['bootstrap_repetitions'], count))

    def paired_values(reference, metric):
        return np.array([[index[bank+i, primary][metric]-index[bank+i, reference][metric]
            for i in range(count)] for _, _, bank in protocol['conditions']]).mean(0)

    def bounded_binary_lower(delta):
        # A positive block difference is at least 1/strata; a negative one
        # can have magnitude at most 1, including unobserved severe losses.
        wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
        lower_win = 0. if wins == 0 else float(beta.ppf(.025, wins, count-wins+1))
        upper_loss = 1. if losses == count else float(beta.ppf(.975, losses+1, count-losses))
        lower = lower_win/strata-upper_loss
        return dict(winning_blocks=wins, losing_blocks=losses,
            difference=float(delta.mean()), conservative_lower=lower,
            noninferior_3pp=lower >= -.03)

    contrasts = {}
    for reference in protocol['primary_references']:
        delta = paired_values(reference, 'capture_time')
        interval = np.quantile(delta[draws].mean(1), [alpha/2., 1.-alpha/2.]).tolist()
        preservation = {key: bounded_binary_lower(paired_values(reference, key))
                        for key in ('capture', 'success')}
        a, b = original['table'][primary], original['table'][reference]
        passed = (original['contrasts'][reference]['improvement'] >= protocol.get('minimum_time_improvement', .10)
                  and interval[1] < 0. and a['collision'] <= b['collision'] and a['success'] >= b['success']
                  and all(p['noninferior_3pp'] for p in preservation.values()))
        contrasts[reference] = dict(time_difference=float(delta.mean()), time_interval=interval,
            preservation=preservation, passed=bool(passed))
    return dict(complete=True, independent_random_stream_blocks=count, scenarios=count*strata,
        criteria_passed=all(c['passed'] for c in contrasts.values()), contrasts=contrasts,
        time_interval_confidence=1.-alpha, binary_one_sided_confidence=.95,
        results_sha256=original['results_sha256'], protocol_sha256=original['protocol_sha256'],
        sensitivity_protocol_sha256=hashlib.sha256(design_path.read_bytes()).hexdigest(),
        scope='Additional dependence-aware sensitivity analysis; preserves the original registered analysis.')


def analyze_learning(root):
    design_path = root/'predictive-learning-analysis-protocol.json'
    design = json.loads(design_path.read_text())
    banks, count = design['scene_banks'], design['scenes_per_stratum']
    scene_ids = [bank+i for bank in banks for i in range(count)]
    indexes, hashes = [], {str(design_path): hashlib.sha256(design_path.read_bytes()).hexdigest()}
    for training_seed, name in zip(design['training_seeds'], design['runs']):
        directory = root/name
        result = json.loads((directory/'results.json').read_text())
        protocol = json.loads((directory/'protocol.json').read_text())
        if not result['complete'] or protocol['training_seed'] != training_seed:
            raise ValueError('Missing completed independent training replicate.')
        if (protocol['validation_bank'] != banks[0] or protocol['validation_scenes_per_cell'] != count
                or protocol['strengths'] != [design['fixed_strength']]):
            raise ValueError('Learning replication settings differ from the fixed design.')
        arms = ['conditional', 'absolute']+([] if protocol['omit_prior'] else ['predictive'])
        rows = result['episodes']
        expected = {(arm, bank+i, agility) for arm in arms
                    for bank, agility in zip(banks, (4., 6.)) for i in range(count)}
        actual = [(r['arm'], r['scene_seed'], r['agility']) for r in rows]
        if len(set(actual)) != len(actual) or set(actual) != expected:
            raise ValueError('Missing or duplicated learning evaluation scenes.')
        if any(r['strength'] != (0. if r['arm'] == 'predictive' else design['fixed_strength']) for r in rows):
            raise ValueError('Evaluation used an unexpected residual strength.')
        for source, saved in protocol['source'].items():
            current = Path(source)
            if current.exists() and hashlib.sha256(current.read_bytes()).hexdigest() == saved['sha256']:
                continue
            text = saved['text']
            choices = [text.encode(), text.replace('\n', '\r\n').encode()]
            if not any(hashlib.sha256(x).hexdigest() == saved['sha256'] for x in choices):
                raise ValueError('Invalid frozen training source: '+source)
        indexes.append({(r['arm'], r['scene_seed']): r for r in rows})
        for p in (directory/'results.json', directory/'protocol.json'):
            hashes[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    baseline = {s: indexes[0]['predictive', s] for s in scene_ids}
    arrays, per_seed = {}, {}
    for arm in ('conditional', 'absolute', 'predictive'):
        arrays[arm] = np.array([[index[arm, s]['capture_time'] if arm != 'predictive'
                                else baseline[s]['capture_time'] for s in scene_ids] for index in indexes])
    rng = np.random.default_rng(20261008)
    training_draws = rng.integers(0, len(indexes), (20000, len(indexes)))
    scenario_draws = [rng.integers(0, count, (20000, count)) for _ in banks]

    def bootstrap(values):
        values = values.reshape(len(indexes), len(banks), count)
        return np.mean([values[training_draws[:, :, None], cell, take[:, None, :]].mean((1, 2))
                        for cell, take in enumerate(scenario_draws)], axis=0)

    alpha = .05/len(design['references'])
    contrasts = {}
    for reference in design['references']:
        difference = arrays['conditional']-arrays[reference]
        interval = np.quantile(bootstrap(difference), [alpha/2., 1.-alpha/2.]).tolist()
        per_seed[reference] = []
        for i, (training_seed, index) in enumerate(zip(design['training_seeds'], indexes)):
            a = [index['conditional', s] for s in scene_ids]
            b = [baseline[s] if reference == 'predictive' else index[reference, s] for s in scene_ids]
            task_counts = {key: dict(conditional=int(sum(r[key] for r in a)),
                                     reference=int(sum(r[key] for r in b)))
                           for key in ('capture', 'success', 'collision')}
            preservation = {key: binary_lower(np.array([x[key]-y[key] for x, y in zip(a, b)]))
                            for key in ('capture', 'success')}
            observed_ok = (task_counts['capture']['conditional'] >= task_counts['capture']['reference']
                and task_counts['success']['conditional'] >= task_counts['success']['reference']
                and task_counts['collision']['conditional'] <= task_counts['collision']['reference'])
            per_seed[reference].append(dict(training_seed=training_seed, scenes=len(scene_ids),
                conditional_time=float(arrays['conditional'][i].mean()),
                reference_time=float(arrays[reference][i].mean()), time_difference=float(difference[i].mean()),
                counts=task_counts, preservation=preservation,
                directional_and_observed_preservation=bool(difference[i].mean() < 0. and observed_ok)))
        contrasts[reference] = dict(time_difference=float(difference.mean()), time_interval=interval,
            improvement=1.-float(arrays['conditional'].mean()/arrays[reference].mean()),
            passed=bool(interval[1] < 0. and all(r['directional_and_observed_preservation']
                                              for r in per_seed[reference])))
    return dict(complete=True, criteria_passed=all(c['passed'] for c in contrasts.values()),
        training_datasets=len(indexes), unique_scenarios=len(scene_ids),
        crossed_bootstrap_preserves_shared_scenarios=True, time_interval_confidence=1.-alpha,
        contrasts=contrasts, per_training_seed=per_seed, source_sha256=hashes,
        scope='Fixed residual conditional-value learning pilot; two independent training datasets, not five-seed formal evidence.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--learning', action='store_true')
    parser.add_argument('--linked-strata', action='store_true')
    args = parser.parse_args()
    if args.learning and args.linked_strata:
        parser.error('Choose one analysis population.')
    output = (analyze_linked_strata(args.directory) if args.linked_strata else
              analyze_learning(args.directory) if args.learning else analyze(args.directory))
    destination = args.directory/('linked-strata-analysis.json' if args.linked_strata else
        'predictive-learning-analysis.json' if args.learning else 'analysis.json')
    text = json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False)
    if destination.exists() and destination.read_text(encoding='utf-8') != text:
        raise RuntimeError('A different analysis already exists.')
    destination.write_text(text, encoding='utf-8')
    print(text)
