"""Render the registered paper's measured results from complete formal runs.

No training, model selection, smoothing, seed removal, or added experiments.
The supplied study is the output directory of unified.run_study.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np


LABELS = dict(full='CITF', arboids='ARBoids', rp='RP', sac='SAC', boids='Boids',
              no_action='Without action interaction', no_feedback='Without generator feedback')
CN = dict(full='CITF', arboids='ARBoids', rp='RP', sac='SAC', boids='Boids',
          no_action='去候选动作交互', no_feedback='去生成器优化配置反馈')
COLORS = dict(full='#125F93', arboids='#D55E00', rp='#008568', sac='#8064A2',
              boids='#595959', no_action='#CC79A7', no_feedback='#B58A00')
METHODS = ['full', 'arboids', 'rp', 'sac', 'no_action', 'no_feedback']
MAIN = ['full', 'arboids', 'rp', 'sac', 'boids']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def formal_result(path):
    result = read(path)
    if result.get('research_stage') != 'formal' or result['tasks'] != 100:
        raise ValueError(f'Use the registered 100-scenario formal result: {path}')
    return result


def finish(fig, directory, name, source):
    fig.savefig(directory / (name + '.pdf'), bbox_inches='tight',
                metadata={'Subject': str(source), 'Title': name})
    fig.savefig(directory / (name + '.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)


def band(ax, x, values, method, marker=None):
    values = np.asarray(values, dtype=float) * 100
    mean, std = values.mean(0), values.std(0, ddof=1)
    ax.plot(x, mean, color=COLORS[method], label=LABELS[method], marker=marker,
            ls='--' if method.startswith('no_') else '-',
            linewidth=1.8 if method == 'full' else 1.2, markersize=3.5)
    ax.fill_between(x, mean-std, mean+std, color=COLORS[method], alpha=.10, linewidth=0)


def rate_summary(values):
    values = np.asarray(values, dtype=float) * 100
    return f'{values.mean():.1f} ± {values.std(ddof=1):.1f}'


def main_figures(study, directory):
    registry = read(study / 'registry.json')
    seeds = registry['seeds']
    if len(seeds) != 5 or read(study / 'status.json')['status'] != 'numeric_study_completed':
        raise ValueError('The main figures require the complete registered five-seed numeric study.')
    data = {method: [formal_result(study / method / str(seed) / 'main_test.json') for seed in seeds]
            for method in METHODS}
    reference = formal_result(study / 'boids' / 'main_test.json')
    table = ['表II：五个独立训练种子的均值 ± 标准差（%）；Boids为无训练参照。', '',
             '| 方法 | 防御成功率 | 捕获／到时防守 | 碰撞／突破 |', '|---|---|---|---|']
    for method in MAIN + ['no_action', 'no_feedback']:
        if method == 'boids':
            fields = {k: f'{100*reference[k]/reference["tasks"]:.1f}'
                      for k in ('success', 'capture', 'timeout', 'collision', 'breach')}
        else:
            fields = {k: rate_summary([r[k]/r['tasks'] for r in data[method]])
                      for k in ('success', 'capture', 'timeout', 'collision', 'breach')}
        table.append(f'| {CN[method]} | {fields["success"]} | {fields["capture"]}／{fields["timeout"]} | '
                     f'{fields["collision"]}／{fields["breach"]} |')
    table += ['', '同种子差值（完整方法减去消融，百分点）：']
    for method in ('no_action', 'no_feedback'):
        differences = [a['success_rate']-b['success_rate'] for a, b in zip(data['full'], data[method])]
        table.append(f'- {CN[method]}：{rate_summary(differences)}；五次分别为 '
                     + '、'.join(f'{100*d:+.1f}' for d in differences) + '。')
    fig, ax = plt.subplots(figsize=(7.1, 3.9), layout='constrained')
    common_grid = None
    learning_scores = {}
    for method in METHODS:
        values = []
        for seed in seeds:
            rows = [json.loads(line) for line in (study / method / str(seed) / 'evaluations.jsonl').read_text().splitlines()]
            grid = np.array([row['transitions'] for row in rows])
            if common_grid is None:
                common_grid = grid
                if not np.array_equal(common_grid, np.arange(5000, registry['team_transitions']+1, 5000)):
                    raise ValueError('The full-budget learning summary requires all 200 registered validation nodes.')
            if not np.array_equal(grid, common_grid):
                raise ValueError('Keep all registered evaluation points; resolve missing/duplicate records before rendering.')
            values.append([row['success_rate'] for row in rows])
        band(ax, common_grid/1e6, values, method)
        learning_scores[method] = np.asarray(values).mean(1)
    table += ['', '图4全程学习表现：每个种子在全部200个等间隔验证节点上的成功率均值，再按训练种子汇总。',
              f'完整方法：{rate_summary(learning_scores["full"])}（%）。']
    for method in ('no_action', 'no_feedback'):
        differences = learning_scores['full']-learning_scores[method]
        table.append(f'- {CN[method]}：{rate_summary(learning_scores[method])}（%）；'
                     f'完整方法的配对增益为{rate_summary(differences)}个百分点，五次分别为 '
                     + '、'.join(f'{100*d:+.1f}' for d in differences) + '。')
    (directory / 'table2_main.md').write_text('\n'.join(table)+'\n', encoding='utf-8')
    ax.set(xlabel='Team transitions (million)', ylabel='Defense success (%)', xlim=(0, 1), ylim=(0, 100))
    ax.legend(loc='lower right', ncol=2, fontsize=7, frameon=False)
    finish(fig, directory, 'fig4_learning', study / 'registry.json')

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.25), layout='constrained')
    for ax, setting, xs, xlabel in zip(axes, ['agility', 'team'],
            [[1.5, 1.75, 2., 2.25, 2.5, 2.75, 3.], list(range(2, 8))],
            ['Attacker agility', 'Number of defenders']):
        for method in MAIN:
            names = [f'agility_{x:.2f}.json' if setting == 'agility' else f'team_{x}.json' for x in xs]
            if method == 'boids':
                y = [100*formal_result(study / method / name)['success_rate'] for name in names]
                ax.plot(xs, y, color=COLORS[method], label=LABELS[method], marker='o', markersize=3.5, lw=1.2)
            else:
                values = [[formal_result(study / method / str(seed) / name)['success_rate'] for name in names]
                          for seed in seeds]
                band(ax, xs, values, method, 'o')
        ax.set(xlabel=xlabel, ylabel='Defense success (%)', xticks=xs, ylim=(0, 100))
    axes[0].set_title('(a) Attacker agility', loc='left', fontsize=9)
    axes[1].set_title('(b) Unseen team sizes', loc='left', fontsize=9)
    axes[0].legend(loc='lower left', fontsize=7, frameon=False)
    finish(fig, directory, 'fig5_applicability', study / 'registry.json')


def alternating_figure(root, directory):
    rows = [formal_result(root / 'phase2' / 'phase1_apf.json')]
    rows += [formal_result(root / f'phase{i}' / 'endpoint.json') for i in range(2, 6)]
    rows += [formal_result(root / 'phase5' / 'apf_retention.json')]
    fig, ax = plt.subplots(figsize=(7.1, 3.3), layout='constrained')
    bottom = np.zeros(6)
    for outcome, label, color in [('capture', 'Capture', '#125F93'), ('timeout', 'Safe timeout', '#55A6A4'),
                                   ('collision', 'Defender collision', '#D69526'), ('breach', 'Breach', '#C44E52')]:
        values = np.array([100*r[outcome]/r['tasks'] for r in rows])
        ax.bar(np.arange(6), values, bottom=bottom, color=color, label=label, width=.62)
        bottom += values
    ax.set(xticks=np.arange(6), xticklabels=['1 Def/APF', '2 Att', '3 Def', '4 Att', '5 Def', '5 APF revisit'],
           ylabel='Episode outcomes (%)', ylim=(0, 100))
    ax.axvline(4.5, color='#999999', linestyle=':', linewidth=.8)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.18), ncol=4, frameon=False, fontsize=7)
    finish(fig, directory, 'fig6_alternating', root)


def vrx_figure(root, directory):
    summary = read(root / 'full' / 'summary.json')
    if summary.get('research_stage') != 'formal' or summary['episodes'] != 100:
        raise ValueError('Use the completed 100-episode formal VRX batch.')
    # The first registered episode is captured during the actual batch.
    # Keep its outcome; do not pick a different episode after viewing results.
    episode = root / 'full' / 'episode-000'
    result = read(episode / 'result.json')
    frames = result['frames']
    if not frames:
        raise ValueError('The first episode needs its recorded camera frames.')
    selected = np.linspace(0, len(frames)-1, min(3, len(frames))).round().astype(int)
    fig = plt.figure(figsize=(7.1, 5.5), layout='constrained')
    layout = fig.add_gridspec(2, len(selected), height_ratios=[1., 1.6])
    for column, index in enumerate(selected):
        frame = frames[index]
        ax = fig.add_subplot(layout[0, column])
        ax.imshow(plt.imread(episode / frame['file']))
        ax.set_title(f't = {frame["simulation_seconds"]:.1f} s', fontsize=8)
        ax.axis('off')
    ax = fig.add_subplot(layout[1, :])
    with np.load(episode / 'trajectory.npz') as saved:
        attacker = saved['AttPos']
        defenders = saved['DefPos'].reshape(-1, 3, 2)
    terminal = np.asarray(result['terminal_positions'])
    attacker = np.concatenate((attacker, terminal[None, 0]), axis=0)
    defenders = np.concatenate((defenders, terminal[None, 1:]), axis=0)
    ax.add_patch(Circle((0, 0), 15., color='#E7EEF2'))
    ax.add_patch(Circle((0, 0), 15., fill=False, edgecolor='#647F8C', linestyle='--', linewidth=.9))
    for positions, label, color in [(attacker, 'Attacker', '#C44E52')]+[
            (defenders[:, i], f'D{i+1}', color) for i, color in enumerate(['#125F93', '#008568', '#8064A2'])]:
        ax.plot(positions[:, 0], positions[:, 1], color=color, label=label, linewidth=1.4)
        ax.scatter(*positions[0], facecolors='white', edgecolors=color, s=25, zorder=5)
        ax.scatter(*positions[-1], color=color, marker='x', s=30, zorder=5)
    labels = {1: 'Breach', 2: 'Defender collision', 3: 'Capture', 4: 'Safe timeout'}
    ax.set(xlabel='x (m)', ylabel='y (m)', aspect='equal',
           title=f'{labels[result["outcome_code"]]} at {result["simulation_seconds"]:.2f} s')
    ax.legend(loc='best', ncol=4, fontsize=7, frameon=False)
    finish(fig, directory, 'fig3_vrx', episode / 'result.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--alternating', type=Path)
    parser.add_argument('--vrx', type=Path)
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'figures')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    main_figures(args.study, args.out)
    if args.alternating:
        alternating_figure(args.alternating, args.out)
    if args.vrx:
        vrx_figure(args.vrx, args.out)


if __name__ == '__main__':
    main()
