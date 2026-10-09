"""Complete paired analysis; the 6-vessel primary endpoint is predeclared."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta, binomtest, norm

from evaluate_predictive_interception import ROOT, METHODS, frozen_inputs
from gen_fig_gate_contribution import read_csv, save_csv, save


LABELS = dict(original='Original',cbf='CBF',previous='Previous prediction',predictive='Continuation value',
              short_value='Short value',held_prefix='Held-prefix ablation',best_fixed='Best fixed')
COLORS = dict(original='#777777',cbf='#009E73',previous='#56B4E9',predictive='#0072B2',
              short_value='#E69F00',held_prefix='#CC79A7',best_fixed='#8C564B')
CONFIDENCE=.95


def bootstrap_pair(a,b,strata,rng,repeats=10000):
    a,b = np.asarray(a,dtype=float),np.asarray(b,dtype=float)
    sa,sb = np.zeros(repeats),np.zeros(repeats)
    strata=np.asarray(strata)
    for cell in sorted(set(strata)):
        ids=np.flatnonzero(strata==cell)
        chosen=ids[rng.integers(0,len(ids),(repeats,len(ids)))]
        sa+=a[chosen].sum(axis=1)
        sb+=b[chosen].sum(axis=1)
    tail=(1.-CONFIDENCE)/2.
    difference=np.quantile((sa-sb)/len(a),[tail,1.-tail])
    improvement=np.quantile(100*(sb-sa)/sb,[tail,1.-tail])
    return difference,improvement


def summarize(rows):
    table=[]
    rng=np.random.default_rng(19653)
    for group in ('all',*sorted({int(r['defenders']) for r in rows})):
        for method in METHODS:
            r=[x for x in rows if x['method']==method and (group=='all' or int(x['defenders'])==group)]
            counts={k:sum(int(x[k]) for x in r) for k in ('success','capture','collision','source_loss','timeout')}
            p=counts['success']/len(r)
            z=float(norm.ppf((1.+CONFIDENCE)/2.))
            center=(p+z*z/(2*len(r)))/(1.+z*z/len(r))
            half=z*np.sqrt(p*(1.-p)/len(r)+z*z/(4*len(r)**2))/(1.+z*z/len(r))
            lo,hi=max(0.,min(p,center-half)),min(1.,max(p,center+half))
            times=[float(x['failure_capped_capture_time']) for x in r]
            ci,_=bootstrap_pair(times,np.ones(len(r)),[x['cell'] for x in r],rng)
            table.append(dict(group=group,method=method,episodes=len(r),**counts,success_ci_low=lo,
                success_ci_high=hi,capped_capture_time=float(np.mean(times)),time_ci_low=ci[0]+1.,
                time_ci_high=ci[1]+1.,mean_return=float(np.mean([float(x['team_return']) for x in r])),
                successful_capture_time=float(np.mean([float(x['duration']) for x in r if int(x['capture'])]))
                if counts['capture'] else np.nan))
    return table


def comparisons(rows):
    index={(int(r['scene_seed']),r['method']):r for r in rows}
    output=[]
    rng=np.random.default_rng(881276)
    for group in ('all',*sorted({int(r['defenders']) for r in rows})):
        seeds=sorted({int(r['scene_seed']) for r in rows if group=='all' or int(r['defenders'])==group})
        a=[index[s,'predictive'] for s in seeds]
        strata=[r['cell'] for r in a]
        for method in METHODS:
            if method=='predictive':
                continue
            b=[index[s,method] for s in seeds]
            ta=np.array([float(r['failure_capped_capture_time']) for r in a])
            tb=np.array([float(r['failure_capped_capture_time']) for r in b])
            d=np.array([int(x['success'])-int(y['success']) for x,y in zip(a,b)])
            wins,losses=int((d>0).sum()),int((d<0).sum())
            n=len(seeds)
            tail=(1.-CONFIDENCE)/2.
            lower_w=0. if wins==0 else float(beta.ppf(tail,wins,n-wins+1))
            upper_l=1. if losses==n else float(beta.ppf(1.-tail,losses+1,n-losses))
            ci,ratio_ci=bootstrap_pair(ta,tb,strata,rng)
            output.append(dict(group=group,reference=method,episodes=n,wins=wins,losses=losses,
                delta_success_pp=float(100*d.mean()),success_conservative_lower_pp=100*(lower_w-upper_l),
                success_noninferior_margin3pp=lower_w-upper_l>=-.03,
                mcnemar_p=float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.,
                delta_capture_seconds=float((ta-tb).mean()),delta_ci_low=float(ci[0]),delta_ci_high=float(ci[1]),
                improvement_percent=float(100*(tb.mean()-ta.mean())/tb.mean()),
                improvement_ci_low=float(ratio_ci[0]),improvement_ci_high=float(ratio_ci[1]),
                collisions_new=sum(int(r['collision']) for r in a),collisions_reference=sum(int(r['collision']) for r in b)))
    return output


def summarize_cells(rows):
    table=[]
    for cell in sorted({r['cell'] for r in rows}):
        for method in METHODS:
            group=[r for r in rows if r['cell']==cell and r['method']==method]
            table.append(dict(cell=cell,method=method,episodes=len(group),
                **{key:sum(int(r[key]) for r in group) for key in
                   ('success','capture','collision','source_loss','timeout')},
                capped_capture_time=float(np.mean([float(r['failure_capped_capture_time']) for r in group]))))
    return table


def figures(rows,table,contrasts,root):
    folder=root/'figures'
    folder.mkdir(parents=True,exist_ok=True)
    index={(r['group'],r['method']):r for r in table}
    frozen=json.loads((root/'frozen_parameters.json').read_text(encoding='utf-8'))['selected']
    methods=list(METHODS)
    if frozen['fixed']=='cbf':
        methods.remove('best_fixed')
    methods=[m for m in methods if m!='original']+['original']
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.8),layout='constrained')
    y=np.arange(len(methods))
    for ax in axes:
        ax.set_yticks(y,[LABELS[m] for m in methods],fontsize=8)
        ax.invert_yaxis()
        ax.grid(axis='x',alpha=.22)
    data=[index[6,m] for m in methods]
    p=np.array([100*r['success']/r['episodes'] for r in data])
    ci=np.array([[100*r['success_ci_low'],100*r['success_ci_high']] for r in data]).T
    axes[0].barh(y,p,color=[COLORS[m] for m in methods],height=.62,alpha=.85)
    axes[0].errorbar(p,y,xerr=[p-ci[0],ci[1]-p],fmt='none',ecolor='#222222',capsize=2,lw=.8)
    for i,r in enumerate(data):
        axes[0].text(119,i,f"{r['success']}/128",va='center',ha='right',fontsize=7)
    axes[0].set(xlim=(0,121),xlabel='Defense success (%)',xticks=[0,25,50,75,100])
    axes[0].set_title('a  Success',loc='left')
    t=np.array([r['capped_capture_time'] for r in data])
    tci=np.array([[r['time_ci_low'],r['time_ci_high']] for r in data]).T
    axes[1].barh(y,t,color=[COLORS[m] for m in methods],height=.62,alpha=.85)
    axes[1].errorbar(t,y,xerr=[t-tci[0],tci[1]-t],fmt='none',ecolor='#222222',capsize=2,lw=.8)
    axes[1].set(xlabel='Capture time (non-captures: 80 s)',xlim=(0,91),yticklabels=[])
    for i,value in enumerate(t):
        axes[1].text(max(tci[1,i]+1,value+1),i,f'{value:.1f}',va='center',fontsize=7)
    axes[1].set_title('b  Interception efficiency',loc='left')
    fig.suptitle(f'Primary: 128 new six-vessel scenes | {100*CONFIDENCE:g}% intervals',fontsize=10)
    save(fig,folder,'fig_predictive_primary')

    if all((n,'predictive') in index for n in (2,3,4,5,6)):
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.5),layout='constrained')
        for method in ('cbf','previous','predictive','short_value'):
            values=[index[n,method] for n in (2,3,4,5,6)]
            axes[0].plot(range(2,7),[100*r['success']/r['episodes'] for r in values],marker='o',markersize=4,
                         color=COLORS[method],label=LABELS[method])
            axes[1].plot(range(2,7),[r['capped_capture_time'] for r in values],marker='o',markersize=4,
                         color=COLORS[method],label=LABELS[method])
        for ax in axes:
            ax.set(xlabel='Number of defenders',xticks=range(2,7))
        axes[0].set(ylabel='Defense success (%)',ylim=(0,105))
        axes[0].set_title('a  Success across fleet sizes',loc='left')
        axes[1].set(ylabel='Capture time (s; non-captures: 80)',ylim=(0,85))
        axes[1].set_title('b  Efficiency across fleet sizes',loc='left')
        axes[1].legend(fontsize=7,loc='upper right')
        fig.suptitle('256 new scenes | n=32 per fleet size 2–5; n=128 for fleet size 6',fontsize=9)
        save(fig,folder,'fig_predictive_generalization')

    fig,axes=plt.subplots(1,2,figsize=(7.1,3.6),layout='constrained')
    lookup={(int(r['scene_seed']),r['method']):r for r in rows if int(r['defenders'])==6}
    seeds=sorted({s for s,_ in lookup})
    for ax,reference,title in zip(axes,('cbf','short_value'),('a  Compared with CBF','b  Value-horizon ablation')):
        for agility,color in zip((1.5,2.,2.5,3.),('#0072B2','#009E73','#E69F00','#CC79A7')):
            part=[s for s in seeds if float(lookup[s,'predictive']['agility'])==agility]
            ax.scatter([float(lookup[s,reference]['failure_capped_capture_time']) for s in part],
                       [float(lookup[s,'predictive']['failure_capped_capture_time']) for s in part],
                       s=12,alpha=.6,color=color,label=f'{agility:g}')
        ax.plot([0,80],[0,80],color='#777777',ls='--',lw=.8)
        ax.set(xlim=(0,84),ylim=(0,84),xlabel=LABELS[reference]+' time (s)',
               ylabel='Continuation-value time (s)',aspect='equal')
        contrast=next(r for r in contrasts if r['group']==6 and r['reference']==reference)
        ax.set_title(title+f"\nMean difference {contrast['delta_capture_seconds']:+.2f} s; "
                     f"{100*CONFIDENCE:g}% CI [{contrast['delta_ci_low']:+.2f}, {contrast['delta_ci_high']:+.2f}]",
                     loc='left',fontsize=8.5)
    axes[1].legend(title='Attacker agility',fontsize=6.5,title_fontsize=7,loc='lower right')
    fig.suptitle('128 paired scenes | points below the diagonal favor continuation value',fontsize=9)
    save(fig,folder,'fig_predictive_paired')


def main():
    root=ROOT/'validation'
    spec=json.loads((root/'specification.json').read_text(encoding='utf-8'))
    if json.loads(json.dumps(frozen_inputs()))!=spec:
        raise AssertionError('Frozen study differs.')
    for stage in ('develop','confirm'):
        if not json.loads((root/f'{stage}_summary.json').read_text(encoding='utf-8'))['passed']:
            raise AssertionError('A study stage failed.')
    rows=read_csv(root/'confirm.csv')
    table=summarize(rows)
    contrasts=comparisons(rows)
    save_csv(root/'outcome_summary.csv',table)
    save_csv(root/'paired_comparisons.csv',contrasts)
    save_csv(root/'cell_summary.csv',summarize_cells(rows))
    figures(rows,table,contrasts,root)
    primary=[r for r in contrasts if r['group']==6 and r['reference'] in ('cbf','best_fixed')]
    strong=all(r['improvement_percent']>=10 and r['delta_ci_high']<0 and r['delta_success_pp']>=0
               and r['collisions_new']<=r['collisions_reference'] for r in primary)
    output=dict(passed=True,primary_strong_advantage=strong,primary=primary,
                selected=json.loads((root/'frozen_parameters.json').read_text(encoding='utf-8'))['selected'])
    (root/'analysis.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output))


if __name__=='__main__':
    main()
