"""Complete paired analysis and figures for the frozen third confirmation."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np
import matplotlib.pyplot as plt

import gen_fig_predictive_interception as common
from gen_fig_gate_contribution import save, save_csv, read_csv
from evaluate_robust_prediction import ROOT, CONFIDENCE, verify
from source_arboids import sha256


COLORS = dict(predictive='#0072B2', cbf='#009E73', short_value='#E69F00',
              strongest_short='#D55E00', best_fixed='#8C564B', legacy_value='#777777', original='#999999')


def primary_figure(table, labels, methods):
    index = {r['method']:r for r in table}
    visible = [m for m in methods if m != 'original']
    n = index['predictive']['episodes']
    fig, axes = plt.subplots(1,2,figsize=(7.1,3.5),layout='constrained')
    for i, method in enumerate(visible):
        r = index[method]
        p = 100*r['success']/n
        axes[0].errorbar(p,i,xerr=[[p-100*r['success_ci_low']], [100*r['success_ci_high']-p]],
                        color=COLORS[method], fmt='o', capsize=3, markersize=5)
        axes[0].text(101.7,i,f"{r['success']}/{n}",ha='right',va='center',fontsize=8)
        mean = r['capped_capture_time']
        axes[1].barh(i,mean,color=COLORS[method],height=.55,alpha=.85)
        axes[1].errorbar(mean,i,xerr=[[mean-r['time_ci_low']], [r['time_ci_high']-mean]],
                        color='#222222',fmt='none',capsize=2,lw=.8)
        axes[1].text(r['time_ci_high']+.4,i,f'{mean:.2f}',va='center',fontsize=8)
    for ax in axes:
        ax.set_yticks(range(len(visible)),[labels[m] for m in visible])
        ax.invert_yaxis()
        ax.grid(axis='y',visible=False)
    low = min(index[m]['success_ci_low']*100 for m in visible)
    axes[0].set(xlim=(min(95.,np.floor(low)),102.),xlabel='Defense success (%)')
    axes[0].set_title('a  Success',loc='left')
    upper = max(index[m]['time_ci_high'] for m in visible)
    axes[1].set(xlim=(0,upper+4.),xlabel='Capture time (non-captures: 80 s)',yticklabels=[])
    axes[1].set_title('b  Interception efficiency',loc='left')
    fig.suptitle(f'{n} new six-vessel scenes | {100*CONFIDENCE:.3f}% intervals',fontsize=10)
    save(fig,ROOT/'figures','fig_robust_primary')


def paired_figure(rows, contrasts, references, labels):
    index = {(int(r['scene_seed']),r['method']):r for r in rows}
    seeds = sorted({int(r['scene_seed']) for r in rows})
    data = {r['reference']:r for r in contrasts}
    fig, axes = plt.subplots(2,2,figsize=(7.1,6.4),layout='constrained')
    palette = ['#0072B2','#009E73','#E69F00','#CC79A7']
    for k,(ax,reference) in enumerate(zip(axes.flat,references)):
        for agility,color in zip((1.5,2.,2.5,3.),palette):
            chosen = [s for s in seeds if float(index[s,'predictive']['agility'])==agility]
            x = [float(index[s,reference]['failure_capped_capture_time']) for s in chosen]
            y = [float(index[s,'predictive']['failure_capped_capture_time']) for s in chosen]
            ax.scatter(x,y,s=10,alpha=.5,c=color,label=f'{agility:g}',edgecolors='none')
        ax.plot([0,80],[0,80],color='#555555',lw=.7,ls='--')
        r = data[reference]
        ax.set(xlim=(0,82),ylim=(0,82),xlabel=labels[reference]+' (s)',ylabel='New prediction (s)',aspect='equal')
        ax.set_title(f"{chr(97+k)}  {r['improvement_percent']:.1f}% faster\n"
                     f"Delta {r['delta_capture_seconds']:.2f} s [{r['delta_ci_low']:.2f}, {r['delta_ci_high']:.2f}]",
                     loc='left',fontsize=9)
    for ax in list(axes.flat)[len(references):]:
        ax.set_visible(False)
    axes.flat[0].legend(title='Attacker agility',loc='upper left',fontsize=7,title_fontsize=7)
    fig.suptitle(f'Paired scenes | {100*CONFIDENCE:.3f}% stratified bootstrap intervals',fontsize=10)
    save(fig,ROOT/'figures','fig_robust_paired')


def runtime_figure(runtime_path):
    records = read_csv(runtime_path.with_suffix('.csv'))
    runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
    fig, ax = plt.subplots(figsize=(7.1,3.1),layout='constrained')
    for engine,color in (('serial','#777777'),('parallel','#0072B2')):
        values = np.sort([1000*float(r['seconds']) for r in records if r['engine']==engine and int(r['planned'])])
        p95 = np.quantile(values,.95)
        ax.step(values,100*np.arange(1,len(values)+1)/len(values),where='post',color=color,
                label=f'{engine.capitalize()} | p95 {p95:.1f} ms | n={len(values)}')
    ax.axvline(200,color='#D55E00',ls='--',lw=1.,label='200-ms control budget')
    ax.axhline(95,color='#AAAAAA',ls=':',lw=.7)
    ax.set(xlabel='Complete planning call (ms)',ylabel='Cumulative fraction (%)',ylim=(0,101),xscale='log')
    ax.legend(loc='lower right')
    ax.set_title(f"32 complete six-vessel episodes | {runtime['exact_steps']} identical serial/parallel commands",loc='left')
    save(fig,ROOT/'figures','fig_robust_runtime')


def main():
    spec = json.loads((ROOT/'specification.json').read_text(encoding='utf-8'))
    verify(spec)
    if not json.loads((ROOT/'execution.json').read_text(encoding='utf-8'))['passed']:
        raise AssertionError('Confirmation did not pass execution checks.')
    methods = spec['selected']['methods']
    common.METHODS, common.CONFIDENCE = methods, CONFIDENCE
    rows = read_csv(ROOT/'episodes.csv')
    table = [r for r in common.summarize(rows) if r['group']==6]
    contrasts = [r for r in common.comparisons(rows) if r['group']==6]
    for r in contrasts:
        r['meets_strong_criteria'] = bool(r['improvement_percent']>=10. and r['delta_ci_high']<0. and
            r['delta_success_pp']>=0. and r['collisions_new']<=r['collisions_reference'] and
            r['success_noninferior_margin3pp'])
    references = spec['selected']['primary_references']
    primary = [r for r in contrasts if r['reference'] in references]
    strong = len(primary)==len(references) and all(r['meets_strong_criteria'] for r in primary)
    save_csv(ROOT/'outcome_summary.csv',table)
    save_csv(ROOT/'paired_comparisons.csv',contrasts)
    save_csv(ROOT/'cell_summary.csv',common.summarize_cells(rows))
    h = .2*spec['selected']['settings']['predictive']['block_steps']
    other_h = .2*spec['selected']['settings']['strongest_short']['block_steps']
    labels = dict(predictive='New prediction',cbf='CBF',short_value=f'{h:g}-s ablation',
                  strongest_short=f'{other_h:g}-s ablation',best_fixed='Fixed guard',
                  legacy_value='Old value rule',original='Original ARBoids')
    (ROOT/'figures').mkdir(exist_ok=True)
    runtime_path = ROOT.parent/('runtime-'+spec['selected']['predictive']+'.json')
    runtime = json.loads(runtime_path.read_text(encoding='utf-8'))
    planning = next(r for r in runtime['groups'] if r['engine']=='parallel' and r['phase']=='planning')
    primary_figure(table,labels,methods)
    paired_figure(rows,contrasts,references,labels)
    runtime_figure(runtime_path)
    result = dict(passed=True, confidence_level=CONFIDENCE, primary_strong_advantage=strong,
        primary=primary, primary_references=references, selected=spec['selected'],
        runtime_planning=planning, meets_runtime=planning['p95_ms']<200.,
        all_requirements_met=strong and planning['p95_ms']<200.,
        data_sha256=sha256(ROOT/'episodes.csv'), analysis_code_sha256=sha256(Path(__file__)))
    (ROOT/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
