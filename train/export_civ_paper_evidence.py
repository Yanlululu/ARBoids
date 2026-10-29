"""Render the current unified CIV evidence directly from completed native records."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'train/experiments/author-source-fusion-20261011'
OUT = BASE / 'unified-civ-development-20261027/paper-evidence-20261027'
SELECTED = OUT.parent / 'finalplan-optimization-1045'
COLORS = dict(CIV='#007F80', CI='#DE8F05', **{'without-ci':'#CC79A7'},
              arboids='#315A97', rp='#8C6D31', sac='#8A89A6', boids='#888888',
              mappo='#E56B6F', ordinary='#56B4E9', independent='#999999')
LABELS = dict(CIV='CIV-RL', CI='w/o CV (CI)', **{'without-ci':'w/o CI'},
              arboids='ARBoids', rp='RP', sac='SAC', boids='Boids', mappo='MAPPO',
              ordinary='CI + ordinary value', independent='Independent selection')
METHODS = ('CIV','without-ci','CI','arboids','rp','sac','boids','mappo')
SOURCES = {}
SOURCE_SNAPSHOTS = {}
FIGURES = []
CAPTIONS = []


def sha(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(2**20),b''):result.update(chunk)
    return result.hexdigest()


def record(path):
    path=Path(path).absolute()
    value=torch.load(path,map_location='cpu',weights_only=False)
    if value.get('completed') is False:raise ValueError(f'Incomplete evidence: {path}')
    SOURCES[str(path.relative_to(ROOT))]=sha(path)
    return value


def csv_table(name, rows):
    if not rows:return
    path=OUT/'tables'/f'{name}.csv';path.parent.mkdir(exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def finish(fig,name,caption):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for extension in ('pdf','png'):
        fig.savefig(folder/f'{name}.{extension}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig);FIGURES.append(name);CAPTIONS.append(dict(figure=name,caption=caption))
    print('FIGURE',name,flush=True)


def wilson(k,n):
    if not n:return (float('nan'),float('nan'))
    z=1.959963984540054;rate=k/n;den=1+z*z/n
    center=(rate+z*z/(2*n))/den
    half=z*math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/den
    return 100*(center-half),100*(center+half)


def task_identity(task):
    return json.dumps(dict(scene=list(task['scene']),future=task['future']),sort_keys=True)


def summary(method,rows,cell):
    n=len(rows);success=sum(int(r['success']) for r in rows);lo,hi=wilson(success,n)
    captured=[r for r in rows if r.get('capture')]
    return dict(cell=cell,method=method,episodes=n,success=success,success_percent=100*success/n,
        wilson95_low=lo,wilson95_high=hi,capture=sum(int(r.get('capture',0)) for r in rows),
        collision=sum(int(r.get('collision',0)) for r in rows),
        attacker_win=sum(int(r.get('source_failure',0)) for r in rows),
        physical_breach=sum(int(r.get('physical_breach',0)) for r in rows),
        timeout=sum(int(r.get('timeout',0)) for r in rows),
        mean_capture_seconds=float(np.mean([r['duration'] for r in captured])) if captured else None,
        adopted=sum(int(r.get('selected',0)>0) for r in rows),
        uncertainty='Wilson interval over tasks; one frozen training pipeline, not seed variability')


def paired(a,b,name,cell):
    if len(a)!=len(b):raise ValueError('Unmatched paired evidence')
    for left,right in zip(a,b):
        if task_identity(left)!=task_identity(right):raise ValueError('Paired task identities differ')
    diff=np.asarray([int(x['success'])-int(y['success']) for x,y in zip(a,b)])
    rescue=int((diff>0).sum());harm=int((diff<0).sum());n=len(a)
    rng=np.random.default_rng(20261027)
    boot=diff[rng.integers(n,size=(10000,n))].mean(1)*100
    lo,hi=np.quantile(boot,[.025,.975])
    return dict(cell=cell,comparison=name,episodes=n,success_A=sum(r['success'] for r in a),
        success_B=sum(r['success'] for r in b),rescues=rescue,harms=harm,
        net_percentage_points=100*float(diff.mean()),paired_bootstrap95_low=lo,paired_bootstrap95_high=hi,
        discordant_exact_p=binomtest(rescue,rescue+harm,.5).pvalue if rescue+harm else 1.,
        collisions_A=sum(r.get('collision',0) for r in a),collisions_B=sum(r.get('collision',0) for r in b))


def load_evaluations(design):
    cells={};source_rows=[]
    for path in sorted((OUT/'evaluations').glob('*/*.pth')):
        value=record(path)
        if value['design_sha256']!=sha(OUT/'design.pth'):raise ValueError('Wrong frozen design')
        cell=path.parent.name
        if [task_identity(t) for t in value['tasks']]!=[task_identity(t) for t in design['cells'][cell]]:
            raise ValueError('Evaluation task order differs from frozen design')
        if len(value['rows'])!=len(value['tasks']):raise ValueError('Missing endpoint outcomes')
        policy=Path(value.get('policy') or '')
        if policy.is_file():
            if sha(policy)!=value['policy_sha256']:raise ValueError('Endpoint policy identity changed')
            SOURCES[str(policy.absolute().relative_to(ROOT))]=value['policy_sha256']
        cells.setdefault(cell,{})[path.stem]=value['rows']
        source_rows.append(dict(cell=cell,method=path.stem,path=str(path.relative_to(ROOT)),
            sha256=sha(path),policy_sha256=value.get('policy_sha256'),
            independent_of_selection=cell!='selected-development'))
    selected=record(SELECTED/'selected-calibrated-comparison.pth')
    if [task_identity(t) for t in selected['tasks']]!=[task_identity(t) for t in design['cells']['selected-development']]:
        raise ValueError('Development evidence cohort changed')
    dev=cells.setdefault('selected-development',{})
    for method in ('CIV','CI','ordinary'):dev[method]=selected['rows'][method]
    for method in ('arboids','rp','sac','boids','mappo'):
        path=SELECTED/f'paper-baseline-{method}-fixed100.pth'
        if path.exists():
            value=record(path)
            if value['tasks']!=selected['tasks']:raise ValueError('Development baselines are unpaired')
            dev[method]=value['rows']
    csv_table('evaluation_sources',source_rows)
    csv_table('all_completed_endpoints',[summary(m,rows,cell) for cell,methods in cells.items() for m,rows in methods.items()])
    details=[]
    for cell,methods in cells.items():
        for method,rows in methods.items():
            for r in rows:
                details.append(dict(cell=cell,method=method,scene_seed=r['scene'][0],fleet=r['scene'][1],
                    agility=r['scene'][2],future=r['future'],**{k:r.get(k) for k in
                    ('success','capture','collision','source_failure','physical_breach','timeout','duration','selected')}))
    csv_table('per_task_outcomes',details)
    mechanism=[]
    for cell in ('confirmation','selected-development'):
        group=cells.get(cell,{})
        for a,b,title in [('CI','independent','CI joint vs independent'),('CIV','CI','CIV vs CI'),
                          ('CIV','ordinary','CIV vs ordinary value')]:
            if a in group and b in group:mechanism.append(paired(group[a],group[b],title,cell))
    new_cells=[key for key in design['cells'] if key!='selected-development']
    if all(all(m in cells.get(key,{}) for m in ('CIV','CI','arboids','rp')) for key in new_cells):
        pooled={m:[row for key in new_cells for row in cells[key][m]] for m in ('CIV','CI','arboids','rp')}
        if len({task_identity(r) for r in pooled['CIV']})!=len(pooled['CIV']):raise ValueError('Duplicate new tasks in descriptive pooling')
        csv_table('all_new_tasks_descriptive_pool',[dict(summary(m,rows,'confirmation plus all robustness cells'),
            pooling='Descriptive pooling of all prespecified new-task strata at their actual sample counts; primary 200-task comparison remains separate') for m,rows in pooled.items()])
        mechanism.append(paired(pooled['CIV'],pooled['CI'],'CIV vs CI; all new tasks, descriptive','all-new-strata'))
    csv_table('Table_II_paired_mechanisms',mechanism)
    sensitivities=[]
    for method,rows in dev.items():
        if '-eta' in method or '-lambda' in method:
            sensitivities.append(dict(parameter='eta' if '-eta' in method else 'lambda_multiplier',
                value=float(method.split('-eta')[-1] if '-eta' in method else method.split('-lambda')[-1]),
                **summary(method,rows,'selected-development'),
                **{k:v for k,v in paired(rows,dev['CI'],method+' vs CI','selected-development').items()
                   if k in ('rescues','harms','net_percentage_points','paired_bootstrap95_low','paired_bootstrap95_high')}))
    csv_table('Table_S1_sensitivity',sensitivities)
    return cells


def geometry(ax,boats,limits=None):
    ax.add_patch(Circle((0,0),15,facecolor='#EFF1F4',edgecolor='#8D99A8',lw=1.1))
    defenders=np.asarray(boats['defenders']);attacker=np.asarray(boats['attacker'])
    ax.scatter(defenders[:,0],defenders[:,1],s=40,c=COLORS['CIV'],zorder=4,label='Defenders')
    ax.scatter(*attacker[:2],s=65,c='#D55E00',marker='^',zorder=4,label='Attacker')
    for index,boat in enumerate(defenders):
        ax.annotate(str(index+1),boat[:2],xytext=(4,4),textcoords='offset points',fontsize=8,color=COLORS['CIV'])
        ax.arrow(*boat[:2],3*np.cos(boat[2]),3*np.sin(boat[2]),width=.08,head_width=.8,color=COLORS['CIV'],zorder=3)
    ax.arrow(*attacker[:2],4*np.cos(attacker[2]),4*np.sin(attacker[2]),width=.12,head_width=1.1,color='#D55E00')
    ax.scatter(0,0,c='#58616F',marker='+',s=45)
    ax.set(xlabel='x (m)',ylabel='y (m)',aspect='equal');ax.grid(alpha=.15)
    if limits:ax.set(xlim=limits,ylim=limits)


def figure_1_3():
    value=record(OUT/'illustrations.pth')
    fig,ax=plt.subplots(figsize=(6.3,5.1),layout='constrained')
    geometry(ax,value['initial'],(-70,70))
    ax.add_patch(Circle((0,0),60,fill=False,edgecolor='#8D99A8',ls='--',lw=.9))
    ax.annotate('Protected region\n$R_T=15$ m',xy=(0,-15),xytext=(0,-26),ha='center',va='top',fontsize=9,
                arrowprops=dict(arrowstyle='-',color='#687588',lw=.8))
    ax.text(-65,62,'Six defending vessels',fontweight='bold')
    ax.text(-65,54,'One maneuvering attacker',fontsize=9)
    ax.legend(loc='lower left',frameon=False)
    finish(fig,'Fig_1_task_geometry','Six-vessel target defense. Positions and headings are an actual initialized source-protocol task. The gray disk is the protected region (15 m); the dashed ring is the 60 m sensing radius. Arrows indicate heading, not an invented measured velocity. Numerical capture and collision radii are 5 m; the source termination also counts an attacker closer to the target than every defender as an attacker win.')

    fig,ax=plt.subplots(figsize=(11,5.7));ax.set(xlim=(0,11),ylim=(0,5.7));ax.axis('off')
    def box(x,y,w,h,text,color='#F3F5F7',edge='#B8C3CE',size=10):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.07,rounding_size=.09',fc=color,ec=edge,lw=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=size,linespacing=1.4)
    def arrow(a,b,dashed=False,label=None):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,color='#5B6774',lw=1.2,linestyle='--' if dashed else '-'))
        if label:ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.12,label,ha='center',fontsize=8)
    box(.15,3.75,1.6,1.15,'Public state\nPositions / motion\nLocal observations',size=9)
    box(2.15,4.4,1.65,.62,'SAC policy',color='#E9EFF8')
    box(2.15,3.45,1.65,.62,'Boids control',color='#E9EFF8')
    arrow((1.8,4.35),(2.08,4.7));arrow((1.8,4.1),(2.08,3.76))
    box(4.25,3.5,2.6,1.5,'CI joint candidates\nLocal preference and\ninter-vessel interactions\nReference $z_0$ + alternatives',color='#FFF2D9',edge=COLORS['CI'],size=9)
    arrow((3.85,4.7),(4.18,4.5));arrow((3.85,3.76),(4.18,4.0))
    box(4.25,1.45,2.6,1.3,'CV conditional value\nReference / rescue / harm\nRelational fleet representation\nSame executable joint plans',color='#DDF1EE',edge=COLORS['CIV'],size=8.7)
    arrow((5.55,3.4),(5.55,2.82))
    box(.3,.8,3.1,1.7,'Paired closed-loop training\nSame root and future randomness\nFactual / alternative outcomes\nGradient updates to CV',size=9)
    arrow((3.47,1.7),(4.18,1.95),True)
    box(7.45,2.6,3.15,1.7,'One final fusion choice\n$S(z_0)=0$\n$S(z)=\\widehat{\\Delta}(x,z)-\\eta$\n$z^\\star=\\arg\\max_z S(z)$',color='#E7F4F2',edge=COLORS['CIV'],size=11)
    arrow((6.92,4.25),(7.38,3.95));arrow((6.92,2.1),(7.38,2.9))
    box(7.45,.75,3.15,1.,'Execute fusion coefficients\nCommon actuator interface',color='#EDF1F7',size=9)
    arrow((9.0,2.52),(9.0,1.82));ax.text(7.45,.3,'$\\eta=0.075$, $\\beta=0$; frozen CI.',fontsize=9)
    finish(fig,'Fig_2_unified_method','Unified CIV-RL decision and learning flow. SAC and Boids provide common source actions. CI constructs interaction-conditioned joint candidates and the reference. The relational CV network is trained from paired terminal outcomes and scores those same executable candidates. A single maximization chooses the final fusion plan. CV gradients update its value networks; they do not update the frozen CI parameters. The deployed uncertainty penalty is zero.')

    fig,axes=plt.subplots(1,3,figsize=(12.4,3.9),layout='constrained')
    catalog=value['catalog'];choice=value['choice'];index=1+(choice-1)//2
    state=value['decision_state'];start=state['defenders'][:,:2]
    ax=axes[0];geometry(ax,state)
    for endpoints,color,label,style in [(catalog['endpoints'][0],COLORS['CI'],'CI reference','--'),
                                        (catalog['endpoints'][index],COLORS['CIV'],'CV-selected plan','-')]:
        for i,(a,b) in enumerate(zip(start,endpoints)):
            ax.plot([a[0],b[0]],[a[1],b[1]],style,color=color,lw=1.4,label=label if i==0 else None)
            ax.scatter(*b,s=15,color=color,zorder=5)
    ax.set_title('(a) Predicted joint motion',loc='left',fontweight='bold');ax.legend(fontsize=7,frameon=False)
    for i in range(6):
        for j in range(i):ax.plot(start[[i,j],0],start[[i,j],1],color='#AAB4BE',lw=.5,alpha=.35,zorder=1)
    ax=axes[1]
    ax.bar([0,1],[value['endpoints'][m]['success'] for m in ('CI','CIV')],color=[COLORS['CI'],COLORS['CIV']],width=.6)
    ax.set(xticks=[0,1],xticklabels=['Factual: CI','Alternative: CIV'],ylabel='Observed terminal success',ylim=(-.05,1.22),yticks=[0,1])
    ax.text(0,.1,'Attacker win\n50.8 s',ha='center',fontsize=9)
    ax.text(1,1.05,'Capture\n9.2 s',ha='center',fontsize=9)
    ax.set_title('(b) Paired closed-loop consequences',loc='left',fontweight='bold')
    gain=np.asarray(value['decision']['gain']);valid=np.asarray(value['features']['valid']).astype(bool)
    x=np.arange(len(gain));colors=[COLORS['CIV'] if i==choice else '#BAC8D1' for i in x]
    axes[2].bar(x[valid],gain[valid],color=np.asarray(colors)[valid]);axes[2].axhline(.075,color='#A74D35',ls='--',lw=1,label='$\\eta=0.075$')
    axes[2].set(xticks=x,xticklabels=[str(i) for i in x],xlabel='Executable joint option (0 = reference)',ylabel='Predicted net success gain')
    axes[2].set_title('(c) Conditional value choice',loc='left',fontweight='bold');axes[2].legend(frameon=False,fontsize=8)
    finish(fig,'Fig_3_candidate_geometry','Actual decision at t=4 s in the first rescued task in the frozen order of the selected development cohort. (a) CI-predicted endpoints for the reference and selected joint plan; these segments are predictions, not simulated trajectories. Light links show the inter-vessel relations evaluated by CI. This case has zero predicted collision-risk cost. (b) Observed factual and alternative terminal outcomes from the same initialized task and future. (c) Conditional value estimates over supported joint options; option 7 exceeds the 0.075 gain threshold. This selected case illustrates mechanism geometry and is not used as aggregate performance evidence.')


def figure_5(cells):
    group=cells.get('confirmation',{})
    if not all(m in group for m in METHODS):return
    stages=[0,1500,3000,4500]
    if not all(f'learning-{u}-eta0.075' in group for u in stages[1:]):return
    reproduction=record(OUT/'learning/reproduction.pth')
    if reproduction['max_parameter_difference']!=0 or not reproduction['selected_update_match']:
        raise ValueError('The CV learning reproduction differs from the selected fit')
    for update in stages[1:]:record(OUT/'learning'/f'conditional-update{update}.pth')
    fig=plt.figure(figsize=(12.4,7.4),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1.05])
    ax=fig.add_subplot(gs[0,0]);learning_rows=[]
    for family in ('arboids','rp','sac'):
        path=BASE/'independent-training'/f'source-six-{family}-seed42-20261021'/'metrics1.csv'
        rows=list(csv.DictReader(path.open()));SOURCES[str(path.relative_to(ROOT))]=sha(path)
        x=np.asarray([int(r['step']) for r in rows]);y=np.asarray([float(r['def_sr']) for r in rows])*100
        for step,rate in zip(x,y):learning_rows.append(dict(stage='base control',method=family,training_seed=42,updates_or_steps=step,success_percent=rate))
        ax.plot(x/1e6,y,color=COLORS[family],alpha=.2,lw=.65)
        label=LABELS[family]+(' (CIV source)' if family=='arboids' else '')
        smoothed=np.convolve(y,np.ones(5)/5,'valid');ax.plot(x[4:]/1e6,smoothed,color=COLORS[family],label=label,lw=1.7)
    ax.set(xlabel='Environment interactions (million)',ylabel='Development success (%)',ylim=(0,105))
    ax.set_title('(a-i) Base control learning',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8)
    ax.text(.02,.02,'Seed 42; faint: raw, solid: 5-checkpoint average',transform=ax.transAxes,fontsize=8)
    ax=fig.add_subplot(gs[0,1]);values=[summary('CI',group['CI'],'confirmation')]
    values.extend(summary('CIV',group[f'learning-{u}-eta0.075'],'confirmation') for u in stages[1:])
    y=np.asarray([v['success_percent'] for v in values]);low=np.asarray([v['wilson95_low'] for v in values]);high=np.asarray([v['wilson95_high'] for v in values])
    ax.errorbar(stages,y,yerr=[y-low,high-y],color=COLORS['CIV'],marker='o',capsize=3,label='Complete CIV policy')
    ax.axhline(y[0],color=COLORS['CI'],ls='--',label='Frozen CI reference')
    ax.axvline(1500,color='#B5B5B5',ls=':',lw=1)
    ax.set(xlabel='CV gradient updates per member',ylabel='Independent task success (%)',xticks=stages,ylim=(min(low)-3,min(102,max(high)+4)))
    ax.set_title('(a-ii) CV learning: full closed-loop policy',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8)
    for u,v in zip(stages,values):learning_rows.append(dict(v,stage='CV full policy',method='CI' if u==0 else 'CIV',updates_or_steps=u))
    ax=fig.add_subplot(gs[1,:]);values=[summary(m,group[m],'confirmation') for m in METHODS]
    y=np.asarray([v['success_percent'] for v in values]);lo=np.asarray([v['wilson95_low'] for v in values]);hi=np.asarray([v['wilson95_high'] for v in values])
    ax.bar(np.arange(8),y,color=[COLORS[m] for m in METHODS],width=.68)
    ax.errorbar(np.arange(8),y,yerr=[y-lo,hi-y],fmt='none',ecolor='#444444',capsize=3,lw=1)
    for i,v in enumerate(values):ax.text(i,min(hi[i]+1,103),f"{v['success']}/{v['episodes']}",ha='center',fontsize=9)
    ax.set(xticks=range(8),xticklabels=[LABELS[m] for m in METHODS],ylabel='Defense success (%)',ylim=(0,108))
    ax.set_title('(b) Frozen policies on 200 new paired tasks',loc='left',fontweight='bold');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    csv_table('Fig_5_learning',learning_rows);csv_table('Fig_5_eight_methods',values)
    finish(fig,'Fig_5_learning_and_comparison','(a-i) Actual seed-42 source ARBoids, RP and SAC development learning histories. Raw checkpoints are faint; solid curves are five-checkpoint moving averages. These are source controllers, not complete CIV-RL. (a-ii) Full CIV closed-loop evaluations at actual reproduced CV updates (three members, each with 1,500/3,000/4,500 updates); 0 denotes CI without CV. The 1,500-update model was selected on development data before confirmation; later points describe learning and are not endpoint-based selection. (b) Eight frozen methods on the same 200 new task/future pairs. Error bars are task-level Wilson 95% intervals for one frozen pipeline, not standard deviations across independent training runs. All methods use numerical source termination. The separate selected-development cohort is tabulated in full and is not pooled with this independent evaluation.')


def figure_6(cells):
    methods=('CIV','CI','arboids','rp');ags=(1,1.5,2,2.5,3,3.5);fleets=(3,4,5,6,7,8)
    needed=[f'agility-{a:g}' for a in ags]+[f'fleet-{n}' for n in fleets if n!=6]
    if not all(all(m in cells.get(k,{}) for m in methods) for k in needed):return
    fig,axes=plt.subplots(1,2,figsize=(10.8,4),layout='constrained');rows=[]
    for panel,xs,label in [(0,ags,'Attacker agility'),(1,fleets,'Number of defenders')]:
        for method in methods:
            y=[]
            for x in xs:
                key=f'agility-{x:g}' if panel==0 else f'fleet-{x}'
                data=cells['agility-2'][method]+cells['agility-2.5'][method] if panel==1 and x==6 else cells[key][method]
                value=summary(method,data,key);rows.append(value);y.append(value['success_percent'])
            axes[panel].plot(xs,y,marker='o',ms=4,lw=1.8,color=COLORS[method],label=LABELS[method])
        axes[panel].set(xlabel=label,ylabel='Defense success (%)',ylim=(0,105),xticks=xs);axes[panel].grid(alpha=.18)
        axes[panel].legend(frameon=False,fontsize=8,loc='lower left')
    axes[0].set_title('(a) Agility robustness',loc='left',fontweight='bold')
    axes[1].set_title('(b) Frozen six-vessel model: team transfer',loc='left',fontweight='bold')
    axes[1].axvline(6,color='#888888',ls=':',lw=1);axes[1].text(6.08,5,'training size',fontsize=8,rotation=90)
    csv_table('Fig_6_robustness_and_fleet',rows)
    finish(fig,'Fig_6_robustness_and_generalization','Frozen-policy transfer with the same CIV checkpoint, CI settings and relational CV architecture. (a) Six agility levels, 24 paired tasks each. (b) Unseen teams of 3, 4, 5, 7 and 8 defenders, 24 paired tasks each; the six-defender reference combines the 48 agility-2/2.5 tasks. No retraining or parameter changes occur across these tests. Every listed task is retained. Small samples show descriptive performance; task counts and Wilson intervals, collisions, capture, attacker wins and physical breaches are provided in the companion table.')


def figure_7():
    folder=OUT/'adaptation';names=['attacker-0','attacker-8000','attacker-16000','attacker-final','defender-updated']
    retention=(folder/'retention-design.pth').exists()
    apf_names=['original-apf-before','original-apf-after']
    if retention:names.append('defender-retained');apf_names.append('original-apf-retained')
    def path(name):
        old=folder/f'{name}.pth'
        return old if old.exists() and 'rows' in record(old) else folder/f'evaluation-{name}.pth'
    paths=[path(n) for n in names+apf_names]
    if not all(p.exists() for p in paths):return
    values=[record(p) for p in paths];design=record(OUT/'adaptation-design.pth')
    for v in values:
        if v['tasks']!=design['evaluation_tasks']:raise ValueError('Adaptation evaluation tasks changed')
        if v.get('attacker') and v.get('opponent_action_calls',0)<=0:raise ValueError('Inactive learned attacker')
    rows=[dict(stage=n,**summary('CIV',v['rows'],'learning-attacker')) for n,v in zip(names+apf_names,values)]
    stages=[record(folder/f'attacker-{n}.pth') for n in ('0','8000','16000','final')]
    fig,axes=plt.subplots(1,2,figsize=(10.9,4),layout='constrained');ax=axes[0]
    for field,label,color in [('success','Defense success',COLORS['CIV']),('attacker_win','Attacker success','#D55E00'),('collision','Defender collision','#7D6C9D')]:
        ax.plot(range(len(names)),[100*r[field]/r['episodes'] for r in rows[:len(names)]],marker='o',color=color,lw=1.8,label=label)
    ax.axvspan(3.5,len(names)-.6,color='#DDF1EE',alpha=.7,zorder=0)
    labels=['Initial']+[f"+{stages[i]['additional_steps']/1000:g}k" for i in range(1,4)]+['CV\nupdate 1']+(['CV +\nreplay'] if retention else [])
    ax.set(xticks=range(len(names)),xticklabels=labels,ylabel='Outcome rate (%)',xlabel='Attacker interactions, then CV continuation',ylim=(0,105),xlim=(-.2,len(names)-.6))
    ax.set_title('(a) Actual alternating continuation',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8,loc='center left')
    ax=axes[1];x=np.arange(2);width=.75/len(apf_names)
    apf_labels=['Before adaptation','After CV update 1']+(['After CV + replay'] if retention else [])
    for j,(v,label,color) in enumerate(zip(values[len(names):],apf_labels,[COLORS['CI'],'#80B1D3',COLORS['CIV']])):
        counts=[sum(r[k] for r in v['rows']) for k in ('success','collision')]
        offset=x+(j-(len(apf_names)-1)/2)*width
        ax.bar(offset,[100*k/len(v['rows']) for k in counts],width,color=color,label=label)
        for location,k in zip(offset,counts):ax.text(location,100*k/len(v['rows'])+1.5,str(k),ha='center',fontsize=8)
    ax.set(xticks=x,xticklabels=['Defense success','Collision'],ylabel='Outcome rate (%)',ylim=(0,105))
    ax.set_title('(b) Original APF-attacker retention',loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8)
    csv_table('Fig_7_learning_attacker',rows)
    csv_table('Fig_7_actual_training_stages',[dict(stage=n,original_plus_new_steps=v['step'],new_steps=v['additional_steps'],gradient_updates=v['gradient_updates'],episodes=v['episodes']) for n,v in zip(names,stages)])
    bank=record(folder/'paired-data.pth');updated=record(folder/'defender-updated.pth')
    updates=[dict(stage='CV update 1',initial_tasks=len(bank['tasks']),actionable_roots=len(bank['roots']),
        branches=bank['branch_count'],simulated_steps=bank['simulation_steps'],opponent_action_calls=bank['opponent_action_calls'],
        training_roots=updated['fit']['training_roots'],validation_roots=updated['fit']['validation_roots'],
        updates=updated['fit']['completed_updates'],selected_update=updated['fit']['updates'],
        acceptance_margin=updated['config']['acceptance_margin'],resume_sha256=updated['fit']['resume_sha256'])]
    if retention:
        protocol=record(folder/'retention-design.pth');second=record(folder/'defender-retained.pth')
        updates.append(dict(stage='CV + original-task replay',new_attacker_roots=len(bank['roots']),
            original_apf_task_prefix=protocol['original_task_prefix'],training_roots=second['fit']['training_roots'],
            validation_roots=second['fit']['validation_roots'],updates=second['fit']['completed_updates'],
            selected_update=second['fit']['updates'],acceptance_margin=second['config']['acceptance_margin'],
            resume_sha256=second['fit']['resume_sha256']))
    csv_table('Fig_7_defender_update',updates)
    for item in (design['initial_training'],design['initial_attacker']):
        source=Path(item['path']).absolute()
        if sha(source)!=item['sha256']:raise ValueError('Attacker continuation source changed')
        SOURCES[str(source.relative_to(ROOT))]=item['sha256']
    final_training=folder/'attacker-training.pth';SOURCES[str(final_training.relative_to(ROOT))]=sha(final_training)
    extra=' A second continuation adds the first 192 original-APF tasks as an explicitly labeled retention objective alongside current-attacker data; original labels are not represented as new-attacker outcomes. Both continuation stages, including the initial retention loss, remain visible.' if retention else ''
    finish(fig,'Fig_7_learning_attacker',f'Actual continuation of a previously trained SAC attacker against frozen current CIV, followed by conditional-value continuation against the final attacker. Actor, critic, target critic, optimizers and entropy are resumed; fresh replay is collected against current CIV. Horizontal labels show actual additional interactions. CV uses newly collected factual/alternative outcomes for {len(bank["roots"])} actionable roots from {len(bank["tasks"])} initial tasks, two paired futures per root, with CI fixed. All stages use the same 64 evaluation tasks. Attacker success is the source failure event; collisions are reported separately. APF bar labels give counts out of 64. All continuation stages are retained, including non-monotonic attacker progress.'+extra)


def figure_4(design):
    folder=OUT/'vrx-cuda';trials=[];valid={};timings={m:[] for m in ('CIV','CI','arboids')}
    for path in sorted(folder.glob('*/result.json')):
        value=json.loads(path.read_text());SOURCES[str(path.relative_to(ROOT))]=sha(path)
        index=value.get('paired_scene_index')
        if index is None:continue
        method=path.parent.name.split('-')[1]
        row=dict(case=index,method=method,attempt=path.parent.name,valid_transport=bool(value.get('passed')),
            success=value.get('success'),outcome=value.get('outcome'),collision=value.get('defender_collision'),
            error=value.get('error'),policy_sha256=value.get('policy_sha256'),capture_radius=value.get('capture_radius'),
            target_radius=value.get('target_radius'),collision_radius=value.get('collision_radius'))
        trials.append(row)
        if not value.get('passed'):continue
        if (index,method) in valid:raise ValueError('Duplicate valid mission evidence')
        valid[index,method]=value
        native=path.parent/'trajectory.npz';trace=np.load(native)
        if len(trace['ControlWallSeconds'])!=value['control_steps']:raise ValueError('VRX latency sample mismatch')
        timings[method].extend(trace['ControlWallSeconds'].tolist());SOURCES[str(native.relative_to(ROOT))]=sha(native)
    csv_table('VRX_all_attempts',trials)
    if len(valid)!=3*len(design['vrx_tasks']):return
    for index in range(len(design['vrx_tasks'])):
        cases=[valid[index,m] for m in timings]
        for key in ('initial_poses','capture_radius','target_radius','collision_radius','termination_rule','duration_limit','action_period','agility','setting','seed'):
            if len({json.dumps(v[key],sort_keys=True) for v in cases})!=1:raise ValueError(f'Unmatched VRX {key}')
        for v in cases[:2]:
            if v['policy_sha256']!=design['models']['CIV']['sha256']:raise ValueError('Wrong current VRX checkpoint')
    latency=[];rates=[]
    for m,times in timings.items():
        array=np.asarray(times);cases=[valid[i,m] for i in range(len(design['vrx_tasks']))]
        latency.append(dict(method=m,control_samples=len(array),mean_ms=array.mean()*1000,p95_ms=np.quantile(array,.95)*1000,
            maximum_ms=array.max()*1000,deadline_ms=200,deadline_misses=int((array>.2).sum()),
            deadline_miss_percent=100*float((array>.2).mean()),backend='CUDA; measured during shared server load'))
        rates.append(dict(method=m,trials=len(cases),success=sum(bool(c['success']) for c in cases),
            captures=sum(c['outcome']=='defender_capture' for c in cases),collision=sum(bool(c['defender_collision']) for c in cases)))
    csv_table('VRX_latency',latency);csv_table('VRX_paired_summary',rates)
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.7),layout='constrained')
    frames=sorted((folder/'case000-CIV').glob('frame-*.png'))
    if frames:
        axes[0].imshow(plt.imread(frames[-1]));axes[0].axis('off')
        SOURCES[str(frames[-1].relative_to(ROOT))]=sha(frames[-1])
    axes[0].set_title('(a) Gazebo / VRX execution',loc='left',fontweight='bold')
    trace=np.load(folder/'case000-CIV/trajectory.npz');defenders=trace['DefPos'].reshape(len(trace['DefPos']),-1,2)
    ax=axes[1];ax.add_patch(Circle((0,0),15,color='#EFF1F4',ec='#A0A8B0'))
    for i in range(defenders.shape[1]):ax.plot(defenders[:,i,0],defenders[:,i,1],color=COLORS['CIV'],lw=1,alpha=.75,label='Defenders' if i==0 else None)
    ax.plot(trace['AttPos'][:,0],trace['AttPos'][:,1],color='#D55E00',lw=1.6,label='Attacker')
    ax.scatter(trace['AttPos'][-1,0],trace['AttPos'][-1,1],marker='x',color='#D55E00',s=30)
    ax.set(xlabel='x (m)',ylabel='y (m)',aspect='equal');ax.legend(frameon=False,fontsize=8)
    ax.set_title('(b) Recorded paired case 0',loc='left',fontweight='bold')
    ax=axes[2]
    for i,r in enumerate(rates):
        rate=100*r['success']/r['trials'];lo,hi=wilson(r['success'],r['trials'])
        ax.bar(i,rate,color=COLORS[r['method']],width=.6);ax.errorbar(i,rate,yerr=[[rate-lo],[hi-rate]],fmt='none',color='#444444',capsize=3)
        ax.text(i,hi+2,f"{r['success']}/{r['trials']}",ha='center',fontsize=9)
    ax.set(xticks=range(3),xticklabels=['CIV-RL','CI','ARBoids'],ylabel='Defense success (%)',ylim=(0,111))
    ax.set_title('(c) Paired high-fidelity outcomes',loc='left',fontweight='bold')
    finish(fig,'Fig_4_vrx_execution','Actual high-fidelity Gazebo/VRX execution of the current unified policy, frozen CI and original ARBoids on 12 common tasks (36 valid missions). Initial poses, seed, agility, control interval and termination settings match within each triplet. Capture uses the original VRX radius 5.5 m, versus 5 m in numerical tests; the two domains are not pooled. All valid mission outcomes are retained. Startup/transport failures and their retries are tabulated separately. Case 0 is the first case in the frozen order, not a performance-selected example. Error bars are task-level Wilson intervals. The latency table aggregates every valid recorded control step, including deadline overruns during shared GPU use; it is not a claim of hard real-time execution.')


def parameters(design):
    value=record(Path(design['models']['CIV']['path']));ordinary=record(Path(design['models']['ordinary']['path']))
    source=record(Path(value['source_path']))
    data=[dict(category='execution',parameter=k,value=json.dumps(v)) for k,v in value['config'].items()]
    data.extend(dict(category='task',parameter=k,value=v) for k,v in dict(
        numerical_protocol='source',main_fleet=6,main_agility='2.0 and 2.5',control_period_seconds=.2,
        horizon_seconds=80,target_radius_m=15,numerical_capture_radius_m=5,collision_radius_m=5,
        vrx_capture_radius_m=5.5,attacker_win='physical breach OR closer to target than every defender',
        defense_success='capture OR timeout; collision and attacker win are failures').items())
    for kind,model in [('conditional',value),('ordinary',ordinary)]:
        fit=model['fit']
        data.extend(dict(category=kind,parameter=k,value=json.dumps(v,default=str)) for k,v in fit.items()
            if k in ('updates','width','members','member_seeds','planned_updates','learning_rate','dropout','weight_decay','training_roots','validation_roots','batch_size','budget','completed_updates','loss_mode','search_completed','data_sources','parameter_count','planned_total_parameter_updates','split_protocol','bootstrap_roots'))
        data.append(dict(category=kind,parameter='trainable_parameters_per_member',value=sum(t.numel() for k,t in model['networks'][0].items() if k.endswith(('weight','bias')))))
        data.append(dict(category=kind,parameter='members',value=len(model['networks'])))
        data.append(dict(category=kind,parameter='checkpoint_sha256',value=design['models']['CIV' if kind=='conditional' else 'ordinary']['sha256']))
    data.extend(dict(category='CI',parameter=k,value=json.dumps(v,default=str)) for k,v in source['actuator_fusion_refinement'].items())
    data.extend(dict(category='source',parameter=k,value=json.dumps(v,default=str)) for k,v in source.items()
        if k in ('source_sha256','source_commit','seed','initialization','simulation_steps','iteration','evidence_role'))
    for k in ('width','dropout','normalize','value_architecture'):
        data.append(dict(category='conditional',parameter=k,value=value.get(k)))
    ablation=record(OUT/'without-ci.pth')
    for k in ('training_roots','validation_roots','member_seeds','updates','completed_updates','loss_mode','parameter_count','data_sources'):
        data.append(dict(category='without_CI',parameter=k,value=json.dumps(ablation['fit'].get(k),default=str)))
    data.extend([dict(category='data',parameter='confirmation_tasks',value=len(design['cells']['confirmation'])),
        dict(category='data',parameter='main_training_repetitions',value=1),
        dict(category='data',parameter='uncertainty_reporting',value='Task Wilson intervals; no claim of five independently retrained complete pipelines'),
        dict(category='ablation',parameter='without_CI_data',value='Recollected local-reference paired roots; same relational architecture and 4500 updates/member; root counts differ from CI-conditioned bank')])
    csv_table('Table_I_parameters',data)


def latency_tables(design):
    path=OUT/'controller-latency.pth'
    if not path.exists():return False
    value=record(path)
    if value['policy_sha256']!=design['models']['CIV']['sha256']:raise ValueError('Different timed controller')
    groups={};raw=[]
    for row in value['rows']:
        key=row['kind'],row['setting'],row['event'];groups.setdefault(key,[]).extend(row['seconds'])
        raw.extend(dict(kind=row['kind'],setting=row['setting'],event=row['event'],scene=json.dumps(row['scene']),
            repeat=i,seconds=t) for i,t in enumerate(row['seconds']))
    rows=[]
    for (kind,setting,event),samples in groups.items():
        array=np.asarray(samples)
        rows.append(dict(kind=kind,setting=setting,event=event,samples=len(array),mean_ms=1000*array.mean(),
            p95_ms=1000*np.quantile(array,.95),maximum_ms=1000*array.max(),
            deadline_exceedance_percent=100*float((array>.2).mean()),device=value['device'],gpu=value['gpu'],scope=value['scope']))
    csv_table('controller_latency_by_fleet_and_lambda',rows);csv_table('controller_latency_raw',raw)
    table=OUT/'tables/Table_S1_sensitivity.csv'
    if table.exists():
        sensitivity=list(csv.DictReader(table.open(encoding='utf-8-sig')))
        for row in sensitivity:
            if row['parameter']=='lambda_multiplier':
                item=next(r for r in rows if r['kind']=='lambda' and r['setting']==float(row['value']))
                row.update(controller_mean_ms=item['mean_ms'],controller_p95_ms=item['p95_ms'],
                    timing_scope='GPU stateful decision benchmark on common roots; no Gazebo transport; actual mission timing reported separately')
        csv_table('Table_S1_sensitivity',sensitivity)
    return True


def integrity(design,cells):
    models={name:record(Path(item['path'])) for name,item in design['models'].items()}
    for name,item in design['models'].items():
        if sha(item['path'])!=item['sha256']:raise ValueError('Frozen policy bytes changed')
        for data in models[name]['fit']['data_sources']:
            path=Path(data['path']).absolute()
            if sha(path)!=data['sha256']:raise ValueError('Fitting bank changed')
            SOURCES[str(path.relative_to(ROOT))]=data['sha256']
    for name,digest in design['runtime'].items():
        path=ROOT/'train'/name
        if sha(path)!=digest:raise ValueError('Frozen execution runtime changed: '+name)
        SOURCES[str(path.relative_to(ROOT))]=digest
    for name in ('civ_generalization.py','civ_rl_experiment.py','civ_paper_evidence.py','civ_deployment.py','civ_baseline_evaluation.py','measure_civ_paper_latency.py'):
        original=ROOT/'train'/name
        frozen=OUT/'source'/'train'/name
        path=frozen if frozen.exists() else original
        SOURCES[str(path.relative_to(ROOT))]=sha(path)
        if path==frozen:
            SOURCE_SNAPSHOTS[str(original.relative_to(ROOT))]=str(frozen.relative_to(ROOT))
    for path in (ROOT/'vrx/run_r20_evidence.py',ROOT/'vrx/run_gate_contribution.py'):
        if path.exists():SOURCES[str(path.relative_to(ROOT))]=sha(path)
    base_training=[]
    for family in ('arboids','rp','sac'):
        path=BASE/'independent-training'/f'source-six-{family}-seed42-20261021'/'source-repeat-result.pth'
        value=record(path);selected=Path(value['selected_checkpoint']).absolute()
        if not value['completed'] or value['training_steps']!=1000000 or sha(selected)!=value['selected_checkpoint_sha256']:
            raise ValueError('Incomplete or changed source baseline')
        SOURCES[str(selected.relative_to(ROOT))]=sha(selected)
        base_training.append(dict(method=family,seed=value['seed'],training_defenders=value['training_defenders'],
            environment_steps=value['training_steps'],checkpoint=str(selected.relative_to(ROOT)),sha256=sha(selected),
            source_commit=value['source_commit'],fresh_initialization=value['fresh_initialization']))
    path=BASE/'r20-six-baseline-comparison/mappo-seed42/training-result.pth';value=record(path)
    if value['steps']!=value['budget']:raise ValueError('Incomplete MAPPO training')
    base_training.append(dict(method='mappo',seed=42,training_defenders=6,environment_steps=value['steps'],
        checkpoint=str((path.parent/'best.pth').relative_to(ROOT)),sha256=value['best_sha256']))
    csv_table('baseline_training_provenance',base_training)
    seen={tuple(t) for m in models.values() for t in m['fit']['training_scenes']+m['fit']['validation_scenes']}
    for cell,tasks in design['cells'].items():
        if cell!='selected-development' and any(tuple(t['scene']) in seen for t in tasks):raise ValueError('Endpoint fitting overlap')
    a,b=models['CIV'],models['ordinary']
    if a['source_sha256']!=b['source_sha256'] or a['fit']['data_sources']!=b['fit']['data_sources']:
        raise ValueError('Ordinary control has different source or paired bank')
    for key in ('parameter_count','completed_updates','training_scenes','validation_scenes'):
        if a['fit'][key]!=b['fit'][key]:raise ValueError('Unmatched ordinary training: '+key)
    checks=dict(frozen_policy_hashes=True,frozen_runtime=True,disjoint_confirmation_and_fit=True,
        ordinary_same_data_source_capacity_and_budget=True,independent_full_pipelines=1)
    complete=all(m in cells.get('confirmation',{}) for m in METHODS+('ordinary','independent'))
    complete=complete and all(f'CIV-eta{v:g}' in cells.get('selected-development',{}) for v in (0,.025,.075,.15))
    complete=complete and all(f'CI-lambda{v:g}' in cells.get('selected-development',{}) for v in (0,.5,1,2))
    checks['main_and_sensitivity_complete']=complete
    preflight=[]
    for path in sorted((OUT/'vrx').glob('*/result.json')):
        value=json.loads(path.read_text());SOURCES[str(path.relative_to(ROOT))]=sha(path)
        preflight.append(dict(attempt=path.parent.name,backend='CPU diagnostic preflight',
            passed=value.get('passed'),success=value.get('success'),outcome=value.get('outcome'),
            error=value.get('error'),mean_control_seconds=value.get('control_seconds',{}).get('mean'),
            included_in_GPU_mission_comparison=False))
    csv_table('VRX_CPU_preflight',preflight)
    return checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    torch.set_num_threads(1)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    design=record(OUT/'design.pth');cells=load_evaluations(design);parameters(design)
    checks=integrity(design,cells);checks['controller_latency_complete']=latency_tables(design)
    figure_1_3();figure_5(cells);figure_6(cells);figure_7();figure_4(design)
    csv_table('figure_captions',CAPTIONS)
    expected={f'Fig_{i}_' for i in range(1,8)}
    missing=[prefix for prefix in sorted(expected) if not any(n.startswith(prefix) for n in FIGURES)]
    complete=not missing and checks['main_and_sensitivity_complete'] and checks['controller_latency_complete']
    manifest=dict(completed=complete,algorithm='CIV-RL-unified-v1',design_sha256=sha(OUT/'design.pth'),
        model_sha256=design['models']['CIV']['sha256'],figures=FIGURES,missing_figures=missing,
        main_inference='CPU fixed width 64',vrx_and_adaptation_inference='CUDA fixed width 64',
        independent_selection_scope=design['confirmation_scope'],development_selection_scope=design['selection_scope'],
        complete_pipeline_repetitions=1,integrity=checks,sources=SOURCES,source_snapshots=SOURCE_SNAPSHOTS,exporter_sha256=sha(__file__),
        inference='Task uncertainty only; retained adverse and null outcomes; no relabeling of legacy F11 records')
    artifacts=list((OUT/'figures').glob('*'))+list((OUT/'tables').glob('*.csv'))
    manifest['artifacts']={str(p.relative_to(OUT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(artifacts)}
    (OUT/'evidence-index.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('EXPORT',json.dumps(dict(completed=complete,figures=FIGURES,missing=missing,integrity=checks)),flush=True)
    if args.require_complete and not complete:raise RuntimeError('Required evidence remains incomplete: '+str(missing)+str(checks))


if __name__=='__main__':main()
