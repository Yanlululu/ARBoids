"""Paired, stratified analysis of the finite fleet-size parameter study."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.stats import binomtest

from evaluate_fleet_selection import METHODS, CONFIGS, frozen_inputs
from gen_fig_gate_contribution import read_csv, save_csv, save, wilson


COLORS = {'original': '#777777', 'cbf': '#009E73', 'current': '#56B4E9',
          'tuned': '#0072B2', 'held_tuned': '#E69F00', 'gate_tuned': '#CC79A7'}
LABELS = {'original': 'Original ARBoids', 'cbf': 'CBF', 'current': 'Current parameters',
          'tuned': 'Tuned feedback', 'held_tuned': 'Held-prediction ablation',
          'gate_tuned': 'Gate-projection ablation'}


def summarize(rows, by_cell=False):
    results = []
    keys = sorted({r['cell'] if by_cell else int(r['defenders']) for r in rows})
    for key in keys:
        for method in METHODS:
            part = [r for r in rows if r['method'] == method and
                    (r['cell'] == key if by_cell else int(r['defenders']) == key)]
            ci = wilson(sum(int(r['success']) for r in part), len(part))
            results.append(dict(**({'cell': key} if by_cell else {'defenders': key}), method=method,
                episodes=len(part), **{k: sum(int(r[k]) for r in part) for k in
                    ('success', 'collision', 'source_loss', 'capture', 'timeout')},
                success_ci_low=float(ci[0]), success_ci_high=float(ci[1]),
                mean_return=float(np.mean([float(r['team_return']) for r in part])),
                mean_duration=float(np.mean([float(r['duration']) for r in part]))))
    return results


def stratified_bootstrap(values, strata, rng, repetitions=10000):
    values = np.asarray(values, dtype=float)
    samples = np.zeros(repetitions)
    for stratum in sorted(set(strata)):
        subset = values[np.asarray(strata) == stratum]
        samples += subset[rng.integers(0, len(subset), (repetitions, len(subset)))].sum(axis=1)
    return map(float, np.quantile(samples / len(values), [.025, .975]))


def comparisons(rows):
    index = {(int(r['scene_seed']), r['method']): r for r in rows}
    rng = np.random.default_rng(194918)
    results = []
    for n in (2, 3, 4, 5, 6):
        seeds = sorted({int(r['scene_seed']) for r in rows if int(r['defenders']) == n})
        actual = [index[s, 'tuned'] for s in seeds]
        strata = [r['cell'] for r in actual]
        for reference in METHODS:
            if reference == 'tuned':
                continue
            controls = [index[s, reference] for s in seeds]
            delta = np.array([int(a['success']) - int(b['success']) for a, b in zip(actual, controls)])
            return_delta = np.array([float(a['team_return']) - float(b['team_return'])
                                     for a, b in zip(actual, controls)])
            lo, hi = stratified_bootstrap(delta, strata, rng)
            rlo, rhi = stratified_bootstrap(return_delta, strata, rng)
            wins, losses = int((delta > 0).sum()), int((delta < 0).sum())
            results.append(dict(defenders=n, reference=reference, episodes=len(seeds), wins=wins, losses=losses,
                delta_success_pp=float(delta.mean()*100), ci_low_pp=lo*100, ci_high_pp=hi*100,
                p_raw=float(binomtest(wins, wins+losses).pvalue) if wins+losses else 1.,
                delta_return=float(return_delta.mean()), return_ci_low=rlo, return_ci_high=rhi))
    running = 0.
    for rank, row in enumerate(sorted(results, key=lambda r: r['p_raw'])):
        running = max(running, min(1., (len(results)-rank)*row['p_raw']))
        row['p_holm_25'] = running
    return results


def success_figure(table, folder):
    index = {(r['defenders'], r['method']): r for r in table}
    x = np.arange(2, 7)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.4), sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.27, top=.83, wspace=.15)
    styles = {'original': ('s', '--'), 'cbf': ('^', '-.'), 'current': ('D', ':'),
              'tuned': ('o', '-'), 'held_tuned': ('v', '--'), 'gate_tuned': ('P', ':')}
    for ax, methods, title in zip(axes,
            [('original', 'cbf', 'current', 'tuned'), ('tuned', 'held_tuned', 'gate_tuned')],
            ['a  Fleet size and tuning', 'b  Mechanism ablations']):
        for j, method in enumerate(methods):
            p = np.array([index[n, method]['success']/64*100 for n in x])
            ax.plot(x, p, label=LABELS[method], color=COLORS[method],
                    marker=styles[method][0], linestyle=styles[method][1], markersize=4)
            if method == 'tuned':
                lower = np.array([index[n, method]['success_ci_low']*100 for n in x])
                upper = np.array([index[n, method]['success_ci_high']*100 for n in x])
                ax.fill_between(x, lower, upper, alpha=.1, color=COLORS[method])
                for n, value in zip(x, p):
                    ax.annotate(f"{index[n,method]['success']}/64", (n, value),
                                xytext=(0, 7), textcoords='offset points', ha='center', fontsize=7,
                                color=COLORS[method])
        ax.set(xlabel='Number of defenders', xticks=x, ylim=(0, 112), xlim=(1.8, 6.2))
        ax.set_title(title, loc='left')
    axes[0].set_ylabel('Defense success (%)')
    handles = {}
    for ax in axes:
        hs, labels = ax.get_legend_handles_labels()
        handles.update(zip(labels, hs))
    fig.legend(handles.values(), handles.keys(), loc='lower center', bbox_to_anchor=(.51, -.015), ncol=3, fontsize=7.5)
    fig.suptitle('320 new scenes | 64 per fleet size | shaded: Wilson 95% CI for tuned method', fontsize=9, y=.985)
    save(fig, folder, 'fig_fleet_success')


def cell_figure(table, folder):
    index = {(r['cell'], r['method']): r['success'] for r in table}
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.1), layout='constrained')
    for ax, other, label in zip(axes.flat, ['original', 'cbf', 'held_tuned', 'gate_tuned'], 'abcd'):
        values = np.array([[index[f'n{n}-a{a:g}', 'tuned']-index[f'n{n}-a{a:g}', other]
                           for a in (1.5, 2., 2.5, 3.)] for n in (2, 3, 4, 5, 6)])
        mesh = ax.imshow(values, cmap='RdBu', norm=TwoSlopeNorm(0., -16., 16.), aspect='auto')
        for (i, j), value in np.ndenumerate(values):
            ax.text(j, i, f'{value:+d}', ha='center', va='center', fontsize=9,
                    color='white' if abs(value) >= 10 else '#222222')
        ax.grid(False)
        ax.set(title=f'{label}  Tuned minus {LABELS[other]}', xticks=range(4),
               xticklabels=['1.5', '2', '2.5', '3'], yticks=range(5), yticklabels=range(2, 7),
               xlabel='Attacker agility', ylabel='Number of defenders')
    fig.colorbar(mesh, ax=axes, shrink=.8, label='Difference in successes (16 paired scenes per cell)')
    save(fig, folder, 'fig_fleet_agility')


def development_figure(rows, selected, folder):
    config_ids = list(CONFIGS)
    index = {(int(r['defenders']), r['configuration']): int(r['successes']) for r in rows}
    values = np.array([[index[n, c] for n in range(2, 7)] for c in config_ids])
    fig, ax = plt.subplots(figsize=(5.1, 3.7), layout='constrained')
    mesh = ax.imshow(values, cmap='Blues', vmin=0, vmax=16, aspect='auto')
    for (i, j), value in np.ndenumerate(values):
        ax.text(j, i, f'{value}/16', ha='center', va='center', color='white' if value>=11 else '#222222')
    for j, n in enumerate(range(2, 7)):
        i = config_ids.index(selected[str(n)]['configuration'])
        ax.add_patch(Rectangle((j-.47, i-.47), .94, .94, fill=False, ec='#E69F00', lw=2.))
    ax.set(title='Development only | outlined: frozen selection', xticks=range(5), xticklabels=range(2, 7),
        yticks=range(6), yticklabels=[f"H={CONFIGS[c]['block_steps']*.2:g} s, threshold={CONFIGS[c]['departure_penalty']:g}"
                                    for c in config_ids], xlabel='Number of defenders')
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label='Successes / 16')
    save(fig, folder, 'fig_fleet_development')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('train/experiments/fleet-selection-20261006'))
    args = parser.parse_args()
    root = args.root
    spec = json.loads((root/'specification.json').read_text(encoding='utf-8'))
    if json.loads(json.dumps(frozen_inputs())) != spec:
        raise AssertionError('Frozen inputs changed before analysis.')
    for stage in ('develop', 'confirm'):
        if not json.loads((root/f'{stage}_summary.json').read_text(encoding='utf-8'))['passed']:
            raise AssertionError('Failed experiment stage.')
    rows = read_csv(root/'confirm.csv')
    if len(rows) != 1920 or len({(r['scene_seed'], r['method']) for r in rows}) != 1920:
        raise AssertionError('Missing or duplicate paired rows.')
    if any(r['original_equivalent'] != 'True' for r in rows if r['method'] == 'original'):
        raise AssertionError('Original-source replay mismatch.')
    selected = json.loads((root/'frozen_parameters.json').read_text(encoding='utf-8'))['selected']
    table, cells, paired = summarize(rows), summarize(rows, True), comparisons(rows)
    save_csv(root/'fleet_summary.csv', table)
    save_csv(root/'cell_summary.csv', cells)
    save_csv(root/'paired_comparisons.csv', paired)
    folder = root/'figures'
    folder.mkdir(exist_ok=True)
    success_figure(table, folder)
    cell_figure(cells, folder)
    development_figure(read_csv(root/'development_summary.csv'), selected, folder)
    index = {(r['defenders'], r['method']): r for r in table}
    maximum = max(index[n, 'tuned']['success'] for n in range(2, 7))
    best_absolute = [n for n in range(2, 7) if index[n, 'tuned']['success'] == maximum]
    current_maximum = max(index[n, 'current']['success'] for n in range(2, 7))
    best_current = [n for n in range(2, 7) if index[n, 'current']['success'] == current_maximum]
    original_gain = max(index[n, 'tuned']['success']-index[n, 'original']['success'] for n in range(2, 7))
    best_gain = [n for n in range(2, 7) if index[n, 'tuned']['success']-index[n, 'original']['success'] == original_gain]
    across_n_advantage = [n for n in range(2, 7) if all(any(r['defenders']==n and r['reference']==m
        and r['delta_success_pp']>0 and r['p_holm_25']<.05 for r in paired)
        for m in ('cbf', 'held_tuned', 'gate_tuned'))]
    summary = dict(passed=True, episodes=len(rows), independent_scenes=len(rows)//6,
                   best_absolute_success_fleet_sizes=best_absolute,
                   best_current_parameter_success_fleet_sizes=best_current,
                   current_successes=sum(index[n, 'current']['success'] for n in range(2, 7)),
                   tuned_successes=sum(index[n, 'tuned']['success'] for n in range(2, 7)),
                   largest_original_improvement_fleet_sizes=best_gain,
                   significant_advantage_over_cbf_and_both_ablations=across_n_advantage,
                   selected=selected)
    (root/'analysis_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
