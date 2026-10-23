"""Export the paper figure groups from verified, fixed-endpoint artifacts."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'train'))
import study_runtime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
from interaction_evaluation import hierarchical_interval, SEEDS, METRICS
from interaction_review import digest

ARMS=('arboids_cbf','same_info','model_value','short','no_peer','full')
LABELS=dict(arboids_cbf='ARBoids+CBF',same_info='Same-info',model_value='Model-return',
            short='Short (2 s)',full='Full (IA-CRRL)',boids='Boids',
            original='ARBoids (unfiltered)',long_reference='Long selector',no_peer='No candidate interaction')
COLORS=dict(arboids_cbf='#6C757D',same_info='#427AB3',model_value='#BC863D',short='#8B6AA8',
            full='#00857D',boids='#A4AAAD',original='#C35B50',long_reference='#204A68',no_peer='#B65E84')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,
    'axes.spines.right':False,'axes.labelsize':8,'legend.fontsize':6,'pdf.fonttype':42,'svg.fonttype':'none'})


def read_csv(path):
    with Path(path).open(encoding='utf-8') as f: return list(csv.DictReader(f))


def stage_review_costs(study, version):
    rows = []
    if version in ('interaction-study-v6','interaction-study-v7'):
        study=Path(study)
        paths=[study/'reviews/signal/completed.json']
        paths.extend((study/'reviews/screens').glob('*/completed.json'))
        paths.extend((study/'reviews/confirmation').glob('*/completed.json'))
        paths.extend((study/'reviews/target-audit').glob('*/completed.json'))
        paths.extend((study/'reviews/bootstrap-check').glob('*/completed.json'))
        paths.extend((study/'reviews/precision').glob('*/completed.json'))
        paths.extend((study/'reviews/development-mechanism').glob('*/completed.json'))
        paths.extend((study/'mechanism').glob('*/completed.json'))
        for path in paths:
            result=json.loads(path.read_text())
            if not result.get('complete'): raise ValueError('Incomplete signal-first diagnostic cost')
            rows.append(dict(stage=str(path.parent.relative_to(study)),seed=result['input'].get('seed'),
                validation_steps=result.get('task_steps',0),
                calibration_steps=result.get('simulated_steps',0)+result.get('state_collection_steps',0),
                wall_seconds=result['wall_seconds']))
        for path in (study/'reviews/bootstrap-repair').glob('*/completed.json'):
            result=json.loads(path.read_text())
            calibration=result['calibration']
            for attempt in result.get('prior_attempts',[]):
                audit=attempt['calibration']
                rows.append(dict(stage=str(path.parent.relative_to(study))+'/failed-reconditioning',
                    seed=int(path.parent.name.split('-')[0]),validation_steps=0,
                    calibration_steps=audit['simulated_steps'],
                    wall_seconds=attempt['learning_wall_seconds']+audit['wall_seconds'],
                    additional_critic_updates=attempt['bootstrap_updates']))
            # The successful evaluator fitting time is already in training.elapsed.
            rows.append(dict(stage=str(path.parent.relative_to(study)),seed=int(path.parent.name.split('-')[0]),
                validation_steps=0,calibration_steps=calibration['simulated_steps'],
                wall_seconds=calibration['wall_seconds']))
            investigation=result.get('discarded_investigation')
            if investigation:
                rows.append(dict(stage=str(path.parent.relative_to(study))+'/discarded-investigation',
                    seed=int(path.parent.name.split('-')[0]),validation_steps=0,
                    calibration_steps=investigation['model_steps'],wall_seconds=investigation['wall_seconds'],
                    additional_critic_updates=investigation['monte_carlo_fit_updates']+investigation['failed_td_updates'],
                    unmeasured_wall_time=investigation['failed_td_wall_seconds'] is None))
        # Retain the CPU baseline evaluation already spent under the earlier graph.
        for path in (study/'reviews').glob('data-gate-*/initial_cbf.json'):
            result=json.loads(path.read_text())
            rows.append(dict(stage=str(path.parent.relative_to(study)),seed=result['seed'],
                validation_steps=result['validation_steps'],calibration_steps=result['calibration_steps'],
                wall_seconds=result['wall_seconds']))
        for row in rows:
            row.setdefault('additional_critic_updates',0)
            row.setdefault('unmeasured_wall_time',False)
        return rows
    for stage in ('gate', 'joint'):
        for seed in SEEDS:
            directory = Path(study)/f'reviews/data-{stage}-{seed}'
            path = directory/'completed.json'
            if version in ('interaction-study-v4', 'interaction-study-v5'):
                # Core, full and peer reviews share immutable arm artifacts; count each once.
                artifacts = {}
                for name in ('completed.json', 'core-completed.json', 'peer-completed.json'):
                    completion = directory/name
                    if not completion.exists(): continue
                    result = json.loads(completion.read_text())
                    if not result.get('complete'): raise ValueError('Incomplete stage cost artifact')
                    for arm, fingerprint in result['artifacts'].items():
                        if arm in artifacts and artifacts[arm] != fingerprint:
                            raise ValueError('Conflicting cached validation cost artifacts')
                        artifacts[arm] = fingerprint
                if not artifacts and (stage == 'joint' or seed == SEEDS[0]):
                    raise ValueError('Missing evidence-chain validation cost')
                if not artifacts: continue
                totals = dict(validation_steps=0, calibration_steps=0, candidate_response_steps=0,
                              candidate_response_cbf_evaluations=0, wall_seconds=0.)
                for arm, fingerprint in artifacts.items():
                    path = directory/f'{arm}.json'
                    if digest(path) != fingerprint: raise ValueError('Validation cost artifact changed')
                    result = json.loads(path.read_text())
                    for key in ('validation_steps', 'calibration_steps', 'candidate_response_steps', 'wall_seconds'):
                        totals[key] += result[key]
                    totals['candidate_response_cbf_evaluations'] += (result.get('candidate_response') or {}).get('cbf_evaluations', 0)
                rows.append(dict(stage=stage, seed=seed, **totals))
                continue
            if version == 'interaction-study-v2' and not path.exists():
                raise ValueError('Missing staged validation cost')
            if version == 'interaction-study-v3':
                if not path.exists(): path = directory/'core-completed.json'
                if not path.exists() and (stage == 'joint' or seed == SEEDS[0]):
                    raise ValueError('Missing evidence-chain validation cost')
            if path.exists():
                result = json.loads(path.read_text())
                rows.append(dict(stage=stage, seed=seed, validation_steps=result['validation_steps'],
                    calibration_steps=result['calibration_steps'], wall_seconds=result['wall_seconds']))
    return rows


def export(fig, directory, name):
    directory.mkdir(parents=True,exist_ok=True)
    for extension in ('pdf','svg','png'):
        fig.savefig(directory/f'{name}.{extension}',bbox_inches='tight',dpi=180)
    plt.close(fig)


def box(ax,x,y,text,width=2.1,height=.8):
    ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,boxstyle='round,pad=.03',
        facecolor='#F0F5F8',edgecolor='#204A68',lw=.9))
    ax.text(x,y,text,ha='center',va='center',fontsize=8)


def arrow(ax,start,end,color='#204A68',style='-'):
    ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1,color=color,linestyle=style))


def schematics(directory):
    fig,ax=plt.subplots(figsize=(3.5,2.6))
    ax.set_aspect('equal');ax.set_xlim(-28,67);ax.set_ylim(-31,31)
    ax.add_patch(Circle((0,0),15,color='#204A68',alpha=.15));ax.text(0,0,'Protected\nregion',ha='center',va='center')
    points=np.array([[23,18],[31,0],[21,-18]])
    ax.plot(*np.vstack((points,points[0])).T,'--',lw=1,color='#00857D')
    for i,(x,y) in enumerate(points):
        ax.add_patch(Polygon([[x+3,y],[x-2,y+1.5],[x-2,y-1.5]],color='#00857D'))
        ax.text(x,y+4,f'D{i+1}',ha='center')
    ax.add_patch(Polygon([[52,4],[58,6],[58,2]],color='#D27A32'))
    arrow(ax,(51,4),(35,8),color='#D27A32');ax.text(53,12,'Intruder',ha='center')
    ax.set_xlabel('Schematic geometry (not an experimental trajectory)');ax.set_xticks([]);ax.set_yticks([])
    export(fig,directory,'fig1-task')
    fig,ax=plt.subplots(figsize=(7.1,2.9));ax.set_xlim(-.3,12);ax.set_ylim(-.5,3.6);ax.axis('off')
    titles=['Public\nobservation','Learned\ncandidates','Candidate\nrelations','Conditional\nblending','Nominal blend\ncommon CBF']
    for i,t in enumerate(titles):
        box(ax,.9+2.45*i,2.6,t)
        if i: arrow(ax,(-.15+2.45*i-.35,2.6),(-.15+2.45*i,2.6))
    box(ax,5.8,1.35,'Peer motion +\ncontrol candidates',3.2)
    arrow(ax,(5.8,1.75),(5.8,2.18))
    box(ax,3.3,0,'Paired composition\ninterventions',2.8);box(ax,7.2,0,'Joint configuration\nvalue',2.8)
    arrow(ax,(4.75,0),(5.75,0));ax.text(5.25,.2,'ΔG',ha='center')
    arrow(ax,(7.6,.45),(8.2,2.15),style='--');ax.text(.05,0,'Training\nonly',color='#D27A32',va='center')
    export(fig,directory,'fig2-architecture')
    fig,ax=plt.subplots(figsize=(3.5,3));ax.set_xlim(-3.2,3.2);ax.set_ylim(-.65,3.5);ax.axis('off')
    box(ax,0,3,'Fixed state + candidate controls',4.8)
    box(ax,-1.55,1.9,'Original\ncomposition',2.4);box(ax,1.55,1.9,'One coefficient\nreplaced',2.4)
    box(ax,-1.55,.65,'Closed-loop\nreturn G',2.4)
    box(ax,1.55,.65,'Closed-loop\nreturn G′',2.4)
    for x in (-1.55,1.55): arrow(ax,(x,2.55),(x,2.32));arrow(ax,(x,1.48),(x,1.07))
    ax.text(0,-.25,'Shared noise; label ΔG = G − G′',ha='center',color='#00857D')
    export(fig,directory,'fig3-intervention')


def series(x,y,label,color,lo=None,hi=None):
    return dict(x=np.asarray(x,float),y=np.asarray(y,float),label=label,color=color,
                lo=None if lo is None else np.asarray(lo,float),hi=None if hi is None else np.asarray(hi,float))


def panel(title,xlabel,ylabel,lines,**options):
    return dict(title=title,xlabel=xlabel,ylabel=ylabel,lines=lines,**options)


def escape(text):
    return str(text).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def coordinates(x,y):
    return ' '.join(f'({a:.6g},{b:.6g})' for a,b in zip(x,y))


def chart(directory,name,panels,caption,label):
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.1),layout='constrained')
    latex=[r'\begin{figure*}[t]\centering',r'\begin{tikzpicture}',
        r'\begin{groupplot}[group style={group size=2 by 2,horizontal sep=1.2cm,vertical sep=1.1cm},',
        r'width=.44\textwidth,height=4.4cm,scale only axis=false,tick label style={font=\scriptsize},',
        r'label style={font=\scriptsize},title style={font=\scriptsize},legend style={font=\tiny,draw=none},',
        r'legend pos=north east,grid=major,grid style={gray!15}]']
    data=[]
    for pi,(ax,p) in enumerate(zip(axes.flat,panels)):
        options=[f'title={{{escape(p["title"])}}}',f'xlabel={{{escape(p["xlabel"])}}}',f'ylabel={{{escape(p["ylabel"])}}}']
        if p.get('equal'): ax.set_aspect('equal');options.append('axis equal image')
        if 'ylim' in p: ax.set_ylim(p['ylim']);options.extend([f'ymin={p["ylim"][0]}',f'ymax={p["ylim"][1]}'])
        latex.append(r'\nextgroupplot['+','.join(options)+']')
        for li,s in enumerate(p['lines']):
            ax.plot(s['x'],s['y'],label=s['label'],color=s['color'],lw=1.2)
            color=f'color{pi}{li}';latex.append(r'\definecolor{'+color+'}{HTML}{'+s['color'].lstrip('#')+'}')
            if s['lo'] is not None:
                ax.fill_between(s['x'],s['lo'],s['hi'],color=s['color'],alpha=.12,lw=0)
                latex.extend([r'\addplot[name path=lower'+str(li)+',draw=none,forget plot] coordinates {'+coordinates(s['x'],s['lo'])+'};',
                    r'\addplot[name path=upper'+str(li)+',draw=none,forget plot] coordinates {'+coordinates(s['x'],s['hi'])+'};',
                    r'\addplot['+color+',fill opacity=.12,forget plot] fill between[of=lower'+str(li)+' and upper'+str(li)+'];'])
            latex.append(r'\addplot['+color+',thick,no marks] coordinates {'+coordinates(s['x'],s['y'])+'};')
            latex.append(r'\addlegendentry{'+escape(s['label'])+'}')
            for k,(x,y) in enumerate(zip(s['x'],s['y'])):
                data.append(dict(panel=pi,series=s['label'],x=x,y=y,
                    lower='' if s['lo'] is None else s['lo'][k],upper='' if s['hi'] is None else s['hi'][k]))
        ax.set(title=p['title'],xlabel=p['xlabel'],ylabel=p['ylabel']);ax.grid(alpha=.15);ax.legend(loc='best')
    export(fig,directory,name)
    with (directory/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=data[0].keys());writer.writeheader();writer.writerows(data)
    latex += [r'\end{groupplot}\end{tikzpicture}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\end{figure*}']
    return '\n'.join(latex)


def seed_band(values):
    values=np.asarray(values,float)
    rng=np.random.default_rng(7402026)
    draws=rng.integers(len(values),size=(10000,len(values)))
    bootstrap=values[draws].mean(axis=1)
    return values.mean(axis=0),*np.quantile(bootstrap,[.025,.975],axis=0)


def trajectories(study):
    panels=[]
    for n,setting in ((3,0),(6,1)):
        path=study/'vrx'/f'vrx-{SEEDS[0]}-full-n{n}-s{setting}-00'/'trajectory.npz'
        with np.load(path) as a:
            position=a['DefPos'].reshape(-1,n,2);attacker=a['AttPos'];time=a['timestamp'];gate=a['AdapterGate']
        lines=[];gates=[]
        colors=['#00857D','#427AB3','#D27A32','#8B6AA8','#C35B50','#706F6F']
        for i in range(n):
            lines.append(series(position[:,i,0],position[:,i,1],f'D{i+1}',colors[i]))
            gates.append(series(time,gate[:,i],f'D{i+1}',colors[i]))
        lines.append(series(attacker[:,0],attacker[:,1],'Intruder','#333333'))
        angle=np.linspace(0,2*np.pi,101)
        lines.append(series(15*np.cos(angle),15*np.sin(angle),'Target boundary','#AAAAAA'))
        name=f'{n} defenders, '+('open water' if setting==0 else 'dock')
        panels.extend([panel(name,'x (m)','y (m)',lines,equal=True),panel(name,'Time (s)','Blending coefficient',gates,ylim=(0,1))])
    return panels


def learning(study):
    panels=[]
    for metric,title in (('success','Defense success'),('capture','Capture'),('collision','Collision'),('capture_time','Capped capture time')):
        lines=[]
        for arm in ARMS:
            rows=[read_csv(study/f'training/seed-{s}/{arm}/metrics.csv') for s in SEEDS]
            steps=[int(r['step']) for r in rows[0]]
            if steps!=list(range(5000,1000001,5000)): raise ValueError('Incomplete learning curve: '+arm)
            if any([int(r['step']) for r in group]!=steps for group in rows): raise ValueError('Unpaired learning checkpoints')
            mean,lo,hi=seed_band([[float(r[metric]) for r in g] for g in rows])
            lines.append(series(np.array(steps)/1e6,mean,LABELS[arm],COLORS[arm],lo,hi))
        panels.append(panel(title,'Additional real steps (million)','Seconds' if metric=='capture_time' else 'Rate',lines,
            ylim=(0,60) if metric=='capture_time' else (0,1)))
    return panels


def generalization(analysis):
    lookup={(r['arm'],r['cell']):r for r in analysis['summaries']}
    panels=[]
    for mode in ('agility','fleet'):
        x=[1.5,2,2.5,3] if mode=='agility' else list(range(2,8))
        cells=[f'n3-a{a:g}' for a in x] if mode=='agility' else [f'n{n}-a2.25' for n in x]
        for metric in ('success','capture_time'):
            lines=[]
            for arm in ARMS:
                rows=[lookup[arm,c] for c in cells]
                lines.append(series(x,[r[metric] for r in rows],LABELS[arm],COLORS[arm],
                    [r['ci95'][metric][0] for r in rows],[r['ci95'][metric][1] for r in rows]))
            panels.append(panel(('Agility' if mode=='agility' else 'Fleet size')+' transfer',
                'Intruder agility' if mode=='agility' else 'Defenders','Defense success' if metric=='success' else 'Capped capture time (s)',
                lines,ylim=(0,1) if metric=='success' else (0,60)))
    return panels


def alternating(study):
    panels=[]
    for metric,title in (('success','Defense success'),('capture','Capture'),('collision','Collision'),('breach','Actual breach')):
        lines=[]
        for arm in ('arboids_cbf','same_info','full'):
            groups=[]
            for seed in SEEDS:
                row=[]
                for stage in range(1,6):
                    part=read_csv(study/f'adversarial/seed-{seed}/{arm}/phase-{stage}/metrics.csv')
                    if [int(r['step']) for r in part]!=list(range(5000,500001,5000)):
                        raise ValueError('Incomplete alternating phase curve')
                    row.extend(float(r[metric]) for r in part)
                groups.append(row)
            mean,lo,hi=seed_band(groups)
            lines.append(series(np.arange(5000,2500001,5000)/1e6,mean,LABELS[arm],COLORS[arm],lo,hi))
        panels.append(panel(title,'Alternating training steps (million)','Rate',lines,ylim=(0,1)))
    return panels


def vrx_rows(study):
    rows=[]
    for seed in SEEDS:
        for arm in ('arboids_cbf','same_info','full'):
            for n in (3,6):
                for setting in (0,1):
                    for scene in range(20):
                        r=json.loads((study/'vrx'/f'vrx-{seed}-{arm}-n{n}-s{setting}-{scene:02d}'/'result.json').read_text())
                        if not r.get('passed'): raise ValueError('Unresolved VRX infrastructure failure')
                        code=r['outcome_code'];duration=r['simulation_seconds']
                        rows.append(dict(arm=arm,training_seed=seed,cell=f'n{n}-s{setting}',scene=scene,
                            success=int(code in (3,4)),capture=int(code==3),collision=int(r['defender_collision']),
                            breach=int(np.linalg.norm(r['terminal_positions'][0])<=r['target_radius']),timeout=int(code==4),
                            capture_time=min(60.,duration) if code==3 else 60.))
    return rows


def aggregate(rows,scenes):
    output=[];rng=np.random.default_rng(7502026)
    for arm,cell in sorted({(r['arm'],r['cell']) for r in rows}):
        groups=[[r for r in rows if r['arm']==arm and r['cell']==cell and int(r['training_seed'])==s] for s in SEEDS]
        if any(len(g)!=scenes for g in groups): raise ValueError('Incomplete grouped evaluation')
        values={m:np.array([[float(r[m]) for r in g] for g in groups]) for m in METRICS}
        output.append(dict(arm=arm,cell=cell,**{m:float(x.mean()) for m,x in values.items()},
            ci95={m:hierarchical_interval(x,rng) for m,x in values.items()}))
    return output


def paired_groups(rows,scenes):
    lookup={(r['arm'],int(r['training_seed']),r['cell'],int(r['scene'])):r for r in rows}
    output=[];rng=np.random.default_rng(7702026)
    for arm,cell in sorted({(r['arm'],r['cell']) for r in rows if r['arm']!='full'}):
        for metric in METRICS:
            differences=[]
            for seed in SEEDS:
                group=[]
                for scene in range(scenes):
                    a=lookup[arm,seed,cell,scene];b=lookup['full',seed,cell,scene]
                    group.append(float(b[metric])-float(a[metric]))
                differences.append(group)
            output.append(dict(reference=arm,cell=cell,metric=metric,difference=float(np.mean(differences)),
                ci95=hierarchical_interval(differences,rng)))
    return output


def table_metrics(rows,caption,label):
    lines=[r'\begin{table*}[t]\centering\scriptsize',r'\caption{'+caption+'}',r'\label{'+label+'}',
        r'\begin{tabular}{@{}llrrrrrr@{}}\toprule',r'Method & Condition & Success & Capture & Collision & Breach & Timeout & $T_{60}$ (s)\\\midrule']
    for r in rows:
        lines.append(escape(LABELS[r['arm']])+' & '+escape(r['cell'])+' & '+
            ' & '.join(f'{100*r[m]:.1f}' for m in METRICS[:-1])+f' & {r["capture_time"]:.2f}'+r'\\')
    lines += [r'\bottomrule\end{tabular}',r'\end{table*}']
    return '\n'.join(lines)


def replace_results(path,block):
    text=path.read_text(encoding='utf-8')
    start,end='%% BEGIN GENERATED RESULTS','%% END GENERATED RESULTS'
    if text.count(start)!=1 or text.count(end)!=1: raise ValueError('Manuscript result markers missing')
    before,rest=text.split(start);_,after=rest.split(end)
    path.write_text(before+start+'\n'+block+'\n'+end+after,encoding='utf-8')


def primary_task_contrast(analysis):
    """Use the recorded reference; never silently relabel historical CBF results."""
    formal=analysis.get('formal_evidence')
    if formal:
        claim=formal['A']
        metric=formal.get('standard',{}).get('primary_metric','capture_time')
        return claim['reference'],claim['cell'],claim['effects'][metric]
    reference,cell='original','n3-a2.25'
    paired=[r for r in analysis['paired_comparisons']
            if r['reference']==reference and r['cell']==cell and r['metric']=='capture_time']
    if len(paired)!=1:
        raise ValueError(f'Missing paired primary capped-time contrast against {LABELS[reference]}')
    return reference,cell,paired[0]


def finish(study,directory,manuscript,include_extensions=True):
    analysis=json.loads((study/'analysis.json').read_text())
    if not analysis['complete']: raise ValueError('The 90000-row fixed numerical matrix is incomplete')
    formal=analysis.get('formal_evidence')
    primary_reference,primary_cell,contrast=primary_task_contrast(analysis)
    primary_label=LABELS[primary_reference]
    primary_metric=(formal or {}).get('standard',{}).get('primary_metric','capture_time')
    primary_name='defense-success-rate' if primary_metric=='success' else 'capped-capture-time'
    primary_unit='percentage points' if primary_metric=='success' else 's'
    primary_scale=100. if primary_metric=='success' else 1.
    primary_effect=contrast['difference']*primary_scale
    primary_ci=[v*primary_scale for v in contrast['ci95']]
    blocks=[]
    blocks.append(chart(directory,'fig4-vrx',trajectories(study),
        'Recorded complete-method trajectories and blending coefficients. The first prescribed formal-seed trial is shown for three defenders in open water and six at the dock; scenarios are fixed independently of outcome. Coefficients parameterize nominal composition and do not assign maneuver roles.','fig:vrx'))
    blocks.append(chart(directory,'fig5-learning',learning(study),
        r'Five-seed validation curves with seed-bootstrap 95\% intervals. The first 0.25 million additional transitions freeze the proposals; subsequent transitions update both policy stages.','fig:learning'))
    blocks.append(chart(directory,'fig6-generalization',generalization(analysis),
        r'Fixed-endpoint agility and fleet-size tests. Each point uses 200 independent paired scenarios per training seed. Bands are hierarchical 95\% bootstrap intervals.','fig:generalization'))
    if include_extensions:
        blocks.append(chart(directory,'fig7-alternating',alternating(study),
            'Alternating defender/attacker training. Defender phases 1, 3, and 5 alternate with attacker phases 2 and 4, each lasting 0.5 million transitions. Curves evaluate the current training opponent; the common frozen-opponent results are reported separately.','fig:alternating'))
    standard=([dict(arm=r['arm'],cell=r['cell'],**r['mean']) for r in formal['arm_summaries']]
              if formal else [r for r in analysis['summaries'] if r['cell']=='n3-a2.25'])
    blocks.append(table_metrics(standard,'Prespecified core conditions at the fixed endpoint. Rates are percentages; all 1000 scenarios per condition enter the capped-time mean.','tab:numerical'))
    vrx_raw=vrx_rows(study)
    vrx=aggregate(vrx_raw,20)
    vrx_paired=paired_groups(vrx_raw,20)
    blocks.append(table_metrics(vrx,'VRX results: 100 episodes per method and condition. s0 denotes open water and s1 the dock. Rates are percentages.','tab:vrx'))
    calibration=[];rng=np.random.default_rng(7602026)
    for arm in ('same_info','model_value','short','no_peer','full'):
        groups=[read_csv(study/f'calibration/seed-{s}/{arm}/calibration.csv') for s in SEEDS]
        if any(len(g)!=64 for g in groups): raise ValueError('Incomplete intervention calibration')
        values=np.array([[float(r['absolute_error']) for r in g] for g in groups])
        calibration.append(dict(arm=arm,E_delta=float(values.mean()),ci95=hierarchical_interval(values,rng)))
    blocks.append(r'\begin{table}[t]\centering\small\caption{Conditional-value error on independent intervention states.}\begin{tabular}{@{}lrr@{}}\toprule Method & $E_\Delta$ & 95\% CI\\\midrule'+'\n'+
        '\n'.join(escape(LABELS[r['arm']])+f' & {r["E_delta"]:.3f} & [{r["ci95"][0]:.3f}, {r["ci95"][1]:.3f}]'+r'\\' for r in calibration)+
        '\n'+r'\bottomrule\end{tabular}\end{table}')
    cost=[]
    for seed in SEEDS:
        for arm in ARMS:
            r=json.loads((study/f'training/seed-{seed}/{arm}/progress.json').read_text())
            if r['step']!=1000000 or not r['complete']: raise ValueError('Incomplete fixed training endpoint')
            cost.append(dict(seed=seed,arm=arm,real_steps=r['step'],auxiliary_steps=r['simulated_steps'],
                             updates=r['updates'],bootstrap_updates=r.get('bootstrap_updates',0),
                             critic_warmup_updates=r.get('critic_warmup_updates',0),wall_seconds=r['elapsed'],phase='formal'))
    blocks.append(r'\begin{table}[t]\centering\scriptsize\caption{Additional training cost, mean per seed. The common pretraining costs one million real steps per seed.}\begin{tabular}{@{}lrrr@{}}\toprule Method & Real steps & Auxiliary steps & Hours\\\midrule'+'\n'+
        '\n'.join(escape(LABELS[a])+f' & 1000000 & {np.mean([r["auxiliary_steps"] for r in cost if r["arm"]==a]):.0f} & {np.mean([r["wall_seconds"] for r in cost if r["arm"]==a])/3600:.2f}'+r'\\' for a in ARMS)+
        '\n'+r'\bottomrule\end{tabular}\end{table}')
    latency=json.loads((study/'runtime/completed.json').read_text())
    if not latency['complete']: raise ValueError('Incomplete isolated latency evaluation')
    blocks.append(r'\begin{table}[t]\centering\scriptsize\caption{Isolated CPU inference including the applicable safety layer. Times are milliseconds.}\begin{tabular}{@{}lrrr@{}}\toprule Method & Defenders & Median & 95th pct.\\\midrule'+'\n'+
        '\n'.join(escape(LABELS[r['arm']])+f' & {r["defenders"]} & {r["median"]*1000:.2f} & {r["p95"]*1000:.2f}'+r'\\' for r in latency['summaries'])+
        '\n'+r'\bottomrule\end{tabular}\end{table}')
    common_summary,common_paired=[],[]
    if include_extensions:
        common=[]
        for arm in ('full','arboids_cbf','same_info'):
            for seed in SEEDS:
                rows=read_csv(study/f'adversarial/seed-{seed}/{arm}/common-opponents.csv')
                if len(rows)!=900: raise ValueError('Incomplete common-opponent evaluation')
                common.extend(dict(r,cell=f'phase{r["phase"]}-{r["opponent"]}') for r in rows)
        common_summary=aggregate(common,100)
        common_paired=paired_groups(common,100)
        blocks.append(table_metrics(common_summary,'Common frozen-opponent tests after defender phases. APF and the ARBoids+CBF attacker checkpoints are shared across methods. Rates are percentages.','tab:opponents'))
    if formal:
        contrasts=[('A: task','probability' if primary_metric=='success' else 's',contrast),('B: conditional value','return',formal['B']['E_env']['all']),
            ('C: six-defender task','probability' if primary_metric=='success' else 's',formal['C']['strong'][primary_metric]),
            ('D: configuration matching','return',formal['D']['configuration_matching']),
            ('D: signed interaction','return',formal['D']['conditional_interaction']),
            ('D: noise-corrected interaction','return squared',formal['D']['noise_corrected_interaction_second_moment'])]
        blocks.append(r'\begin{table*}[t]\centering\small\caption{Prespecified A--D paired effects across all five training seeds. Task and value-error contrasts are Full minus reference. Higher success and lower time or value error favor Full. Matching contrasts are original minus permuted; interaction signs have a different meaning.}\begin{tabular}{@{}llrr@{}}\toprule Evidence & Unit & Effect & 95\% CI\\\midrule'+'\n'+
            '\n'.join(escape(label)+f' & {escape(unit)} & {effect["difference"]:.3f} & [{effect["ci95"][0]:.3f}, {effect["ci95"][1]:.3f}]'+r'\\' for label,unit,effect in contrasts)+
            '\n'+r'\bottomrule\end{tabular}\end{table*}')
        with (directory/'table-formal-seed-effects.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['evidence','unit','seed','effect']);writer.writeheader()
            writer.writerows(dict(evidence=label,unit=unit,seed=seed,effect=value)
                for label,unit,effect in contrasts for seed,value in effect['seed_effects'].items())
        with (directory/'table-core-means.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['arm','cell','metric','mean','ci95_low','ci95_high','seed_standard_deviation']);writer.writeheader()
            writer.writerows(dict(arm=r['arm'],cell=r['cell'],metric=m,mean=r['mean'][m],
                ci95_low=r['ci95'][m][0],ci95_high=r['ci95'][m][1],seed_standard_deviation=r['seed_standard_deviation'][m])
                for r in formal['arm_summaries'] for m in r['mean'])
        with (directory/'table-core-seed-means.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['arm','cell','metric','seed','mean']);writer.writeheader()
            writer.writerows(dict(arm=r['arm'],cell=r['cell'],metric=m,seed=seed,mean=value)
                for r in formal['arm_summaries'] for m,means in r['seed_means'].items() for seed,value in means.items())
    headline=(r'\subsection{Fixed-endpoint evidence}'+'\n'+
        f'For {escape(primary_cell)}, the Full minus {escape(primary_label)} {primary_name} difference is {primary_effect:.2f} {primary_unit} '
        f'(hierarchical paired 95\\% CI [{primary_ci[0]:.2f}, {primary_ci[1]:.2f}] {primary_unit}). '
        'Tables report all task outcomes and capped capture time; the exported analysis retains condition-specific paired intervals for every metric. '
        'This is a complete-method comparison; safety-filter and training-budget differences are disclosed separately. '
        'Matched ablations, rather than this contrast alone, establish the contributions of candidate interaction and conditional-value supervision.\n')
    replace_results(manuscript,headline+'\n\n'.join(blocks))
    text=manuscript.read_text(encoding='utf-8')
    evidence=(f'At the fixed endpoint for {escape(primary_cell)}, the complete-method minus {escape(primary_label)} difference in {primary_name} is '
        f'{primary_effect:.2f} {primary_unit} '
        f'(paired 95\\% confidence interval [{primary_ci[0]:.2f}, {primary_ci[1]:.2f}] {primary_unit}).')
    text=text.replace('This manuscript currently specifies the implemented method and evaluation design; numerical claims about the new policy await completion of the fixed experiment matrix.',evidence)
    text=text.replace('Final empirical conclusions await the corresponding completed evaluations.',evidence+' The condition-specific results, auxiliary controls, and compute measurements define the scope of this empirical finding.')
    manuscript.write_text(text,encoding='utf-8')
    # All empirical claims remain numeric; subsequent editorial synthesis must
    # inspect the complete comparisons before claiming mechanism or superiority.
    manifest=json.loads((study/'manifest.json').read_text())
    pretraining=[]
    for seed in dict.fromkeys([*SEEDS,*manifest.get('development_seeds',[])]):
        reused=manifest['inherited_pretraining'].get(f'pretrain-{seed}')
        if reused:
            pretraining.append(dict(seed=seed,real_steps=1000000,inherited=True,wall_seconds=None,source=reused['provenance']))
        else:
            source=Path(manifest.get('pretraining_outputs',{}).get(str(seed),str(study/f'pretrain/seed-{seed}/actor.pth'))).with_name('completed.json')
            r=json.loads(source.read_text())
            pretraining.append(dict(seed=seed,real_steps=r['actual_phase_steps'],inherited=False,wall_seconds=r['elapsed_seconds'],source=str(source)))
    review_cost=stage_review_costs(study,manifest['format'])
    for path in sorted((study/'training').glob('seed-*/*/progress.json')):
        seed=int(path.parents[1].name.split('-')[1])
        if seed in SEEDS: continue
        r=json.loads(path.read_text())
        cost.append(dict(seed=seed,arm=path.parent.name,real_steps=r['step'],auxiliary_steps=r['simulated_steps'],
                         updates=r['updates'],bootstrap_updates=r.get('bootstrap_updates',0),
                         critic_warmup_updates=r.get('critic_warmup_updates',0),wall_seconds=r['elapsed'],phase='development'))
    history=manifest.get('previous_development');seen={study.resolve()}
    while history:
        previous=Path(history['study']).resolve()
        if previous in seen: raise ValueError('Cyclic development provenance.')
        seen.add(previous);prior=json.loads((previous/'manifest.json').read_text())
        review_cost.extend(dict(row,stage=f'{previous.name}/{row["stage"]}')
                           for row in stage_review_costs(previous,prior['format']))
        for path in sorted((previous/'training').glob('seed-*/*/progress.json')):
            r=json.loads(path.read_text())
            cost.append(dict(seed=int(path.parents[1].name.split('-')[1]),arm=path.parent.name,
                real_steps=r['step'],auxiliary_steps=r['simulated_steps'],updates=r['updates'],
                bootstrap_updates=r.get('bootstrap_updates',0),critic_warmup_updates=r.get('critic_warmup_updates',0),
                wall_seconds=r['elapsed'],phase='development-retired:'+previous.name))
        history=prior.get('previous_development')
    deployment_jobs=[j for j in manifest['jobs'] if j.get('phase')=='deployment-validation']
    deployment_cost=dict(episodes=len(deployment_jobs),
        simulation_seconds=sum(json.loads(Path(j['result']).read_text())['simulation_seconds'] for j in deployment_jobs),
        wall_seconds=sum(json.loads((study/'jobs'/f'{j["name"]}.json').read_text())['wall_seconds'] for j in deployment_jobs))
    summary=dict(numerical=analysis,vrx=vrx,vrx_paired=vrx_paired,calibration=calibration,stage_review_cost=review_cost,
        deployment_validation_cost=deployment_cost,extensions_complete=include_extensions,
        common_pretraining=pretraining,training_cost=cost,latency=latency,common_opponents=common_summary,common_opponents_paired=common_paired)
    (directory/'tables.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    for name,rows in (('numerical',analysis['summaries']),('vrx',vrx),('vrx-paired',vrx_paired),('calibration',calibration),('common-pretraining',pretraining),('training-cost',cost),('latency',latency['summaries']),('common-opponents',common_summary),('common-opponents-paired',common_paired)):
        if not rows: continue
        with (directory/f'table-{name}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    if review_cost:
        with (directory/'table-stage-review-cost.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=review_cost[0].keys());writer.writeheader();writer.writerows(review_cost)
    path=ROOT/'results.md';text=path.read_text(encoding='utf-8')
    start,end='<!-- IA-CRRL RESULTS BEGIN -->','<!-- IA-CRRL RESULTS END -->'
    content=('## 新论文：协同控制组成的 60 秒协议正式结果\n\n'+
        '研究对象为协同控制组成；核心机制为候选控制交互条件化的融合决策与面向组成变量的条件价值学习。'
        '融合系数表征名义控制组成，行为解释结合候选控制、执行推力与实际轨迹。\n\n'+
        f'{primary_cell}：完整方法相对 {primary_label} 的平均封顶捕获耗时差为 {contrast["difference"]:.2f} 秒，'
        f'分层配对 95% 区间 [{contrast["ci95"][0]:.2f}, {contrast["ci95"][1]:.2f}] 秒。\n\n'+
        '原始 ARBoids 不含 CBF。带 CBF 的继续训练对照单独列示；两项创新的收益由保持相同执行层的消融识别，不将完整方法对原法的全部差值归给创新机制。\n\n'+
        '固定训练终点、五个独立正式种子、90,000 个数值测试回合和 1,200 个 VRX 回合已汇总。'
        '旧 80 秒实验保留在下文，独立归属原方法及原协议。\n')
    if start in text:
        prefix,rest=text.split(start);_,suffix=rest.split(end)
        text=prefix+start+'\n'+content+end+suffix
    else: text=start+'\n'+content+end+'\n\n'+text
    path.write_text(text,encoding='utf-8')


# Current source-environment study. Legacy 60-s exporters above remain callable
# only for their own recorded study inputs.
SOURCE_COLORS = {'F00':'#8B9198','F10':'#4477AA','F01':'#AA66AA',
    'F11':'#00857D','ordinary':'#D17B27','M1-strong':'#204A68',
    'M1-independent':'#C35B50','state-interaction':'#A3B7CC',
    'candidate-interaction':'#819AB4'}
SOURCE_LABELS = {'F00':'Transferred source','F10':'Joint fusion','F01':'Value only',
    'F11':'Complete','ordinary':'Ordinary MLP','M1-strong':'Refined joint',
    'M1-independent':'Independent','state-interaction':'State interaction',
    'candidate-interaction':'Candidate interaction'}


def source_rows(evaluations):
    return [row for entry in evaluations.values() if entry for row in entry['rows']]


def source_paired(left, right, repetitions=4000):
    """Paired layout-cluster bootstrap; repeated futures never split a layout."""
    groups = {}
    differences = []
    rescue = harm = 0
    for suite in left:
        a, b = left[suite], right[suite]
        if not a or not b:
            raise ValueError('A paired comparison needs both completed evaluations.')
        if a['tasks'] != b['tasks']:
            raise ValueError('Paired evidence differs in layout, fleet, agility, future or row order.')
        for task, x, y in zip(a['tasks'], a['rows'], b['rows']):
            delta = x['success']-y['success']
            key = (suite, *task[0])
            groups.setdefault(key, []).append(delta)
            differences.append(delta)
            rescue += int(delta > 0); harm += int(delta < 0)
    strata = {}
    for (suite, seed, defenders, agility), values in groups.items():
        strata.setdefault((suite, defenders, agility), []).append(values)
    rng = np.random.default_rng(432201)
    total = np.zeros(repetitions); count = np.zeros(repetitions)
    for values in strata.values():
        sums = np.asarray([sum(x) for x in values], float)
        sizes = np.asarray([len(x) for x in values], int)
        chosen = rng.integers(len(values), size=(repetitions, len(values)))
        total += sums[chosen].sum(1); count += sizes[chosen].sum(1)
    interval = np.quantile(100*total/count, [.025, .975])
    return dict(episodes=len(differences),layouts=len(groups),rescue=rescue,harm=harm,
        difference_pp=100*float(np.mean(differences)),ci95_pp=interval.tolist(),
        interval_scope='development layout-cluster bootstrap; no correction for adaptive model selection')


def source_task_consequences(left, right, repetitions=4000):
    """Separate capture, failure type and time on jointly captured episodes."""
    metrics=('capture','collision','source_failure','timeout','physical_breach')
    groups={}; counts={side:dict.fromkeys(metrics,0) for side in ('left','right')}
    for suite in left:
        a,b=left[suite],right[suite]
        if not a or not b or a['tasks']!=b['tasks']:
            raise ValueError('Task consequences require complete, identical paired tasks.')
        for task,x,y in zip(a['tasks'],a['rows'],b['rows']):
            for side,row in (('left',x),('right',y)):
                for metric in metrics:counts[side][metric]+=row[metric]
            common=int(bool(x['capture'] and y['capture']))
            observation=[x[metric]-y[metric] for metric in metrics]
            observation.extend((common*(x['duration']-y['duration']),common,1))
            key=(suite,*task[0])
            groups[key]=groups.get(key,np.zeros(len(observation)))+observation
    strata={}
    for (suite,seed,defenders,agility),values in groups.items():
        strata.setdefault((suite,defenders,agility),[]).append(values)
    rng=np.random.default_rng(432202);boot=np.zeros((repetitions,len(metrics)+3))
    for values in strata.values():
        values=np.asarray(values)
        sampled=rng.integers(len(values),size=(repetitions,len(values)))
        boot+=values[sampled].sum(1)
    total=np.asarray(list(groups.values())).sum(0)
    result=dict(episodes=int(total[-1]),counts=counts,rate_differences={})
    for index,metric in enumerate(metrics):
        result['rate_differences'][metric]=dict(difference_pp=100*total[index]/total[-1],
            ci95_pp=np.quantile(100*boot[:,index]/boot[:,-1],[.025,.975]).tolist())
    durations=np.divide(boot[:,-3],boot[:,-2],out=np.full(repetitions,np.nan),where=boot[:,-2]>0)
    result['joint_capture_time']=dict(episodes=int(total[-2]),
        difference_seconds=float(total[-3]/total[-2]) if total[-2] else None,
        ci95_seconds=np.nanquantile(durations,[.025,.975]).tolist() if total[-2] else None,
        scope='paired duration conditional on capture by both methods; negative favors left')
    return result


def source_ranking_summary(measurement, repetitions=4000):
    """Compare predicted ordering only on outcome-independent random candidates."""
    rows = measurement['records']; labels = list(rows[0]['ranking']['predictions'])
    counts = []; correct = {label:[] for label in labels}; errors = {label:[] for label in labels}
    strata = {}
    for index, row in enumerate(rows):
        ranking = row['ranking']; lookup = {int(k):i for i,k in enumerate(ranking['indices'])}
        positions = np.asarray([lookup[int(k)] for k in ranking['random_indices']], int)
        factual = np.asarray([item['success'] for item in row['outcomes']['factual']], float)
        outcomes = np.asarray([[item['success'] for item in row['outcomes'][ranking['labels'][i]]]
                               for i in positions], float)
        if outcomes.shape[1] != len(factual) or len(factual) != len(row['futures']):
            raise ValueError('Ranking outcomes must retain every common paired future.')
        measured = (outcomes-factual).mean(1)
        left, right = np.triu_indices(len(measured), 1)
        direction = np.sign(measured[left]-measured[right]); informative = direction != 0
        counts.append(int(informative.sum()))
        for label in labels:
            predicted = np.asarray(ranking['predictions'][label]['predicted_gain'])[0, positions]
            if not np.isfinite(predicted).all():
                raise ValueError('Ranking predictions must be finite and precede outcome acquisition.')
            ordering = np.sign(predicted[left]-predicted[right])
            credit = (ordering == direction).astype(float)
            credit[ordering == 0] = .5
            correct[label].append(float(credit[informative].sum()))
            errors[label].append(float(np.mean((predicted-measured)**2)))
        strata.setdefault(tuple(row['scene'][1:]), []).append(index)
    counts = np.asarray(counts); rng = np.random.default_rng(435801)
    resamples = np.concatenate([rng.choice(indices, (repetitions, len(indices)), replace=True)
                               for indices in strata.values()], axis=1)
    totals = counts[resamples].sum(1)
    result = dict(roots=len(rows), informative_roots=int((counts > 0).sum()),
        comparable_pairs=int(counts.sum()), models={},
        scope='uniform random candidates; measured outcome ties excluded; stratified root bootstrap')
    bootstraps = {}
    for label in labels:
        success = np.asarray(correct[label]); error = np.asarray(errors[label])
        values = np.divide(success[resamples].sum(1), totals,
            out=np.full(repetitions, np.nan), where=totals > 0)
        bootstraps[label] = values
        result['models'][label] = dict(
            concordance=float(success.sum()/counts.sum()) if counts.sum() else None,
            ci95=np.nanquantile(values, [.025,.975]).tolist() if counts.sum() else None,
            gain_mse=float(error.mean()))
    if 'conditional' in labels and 'ordinary' in labels and counts.sum():
        result['conditional_minus_ordinary'] = dict(
            concordance=result['models']['conditional']['concordance']-result['models']['ordinary']['concordance'],
            ci95=np.nanquantile(bootstraps['conditional']-bootstraps['ordinary'], [.025,.975]).tolist())
    return result


def _source_save(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ('pdf', 'png', 'svg'):
        fig.savefig(directory/f'{name}.{extension}', dpi=300, bbox_inches='tight',
                    facecolor='white')
    plt.close(fig)


def _source_rate(entry):
    rows = entry['rows']
    return 100*np.mean([row['success'] for row in rows])


def _source_boxes(ax, boxes, arrows, fontsize=8):
    from matplotlib.patches import FancyArrowPatch
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    for x,y,w,h,text,color in boxes:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.015',
            lw=.9,edgecolor=color,facecolor=color+'12'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fontsize)
    for start,end in arrows:
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,
            linewidth=.9,color='#3D4650'))


def source_method_figures(data, directory):
    fig,ax=plt.subplots(figsize=(3.5,2.7),layout='constrained')
    ax.add_patch(Circle((0,0),15,color='#DDE8ED',ec='#204A68',lw=1))
    ax.text(0,0,'Protected\nregion',ha='center',va='center',fontsize=8)
    positions=np.array([[23,17],[28,-2],[20,-19]])
    for i,(x,y) in enumerate(positions):
        ax.add_patch(Polygon([[x-2,y],[x+1.3,y+1],[x+1.3,y-1]],color='#00857D'))
        ax.text(x+1,y+2.3,f'$D_{i+1}$',fontsize=8)
        ax.plot([x,x+8,x+15],[y,y+4,y+2],color='#4477AA',lw=1,alpha=.7)
        ax.plot([x,x+6,x+13],[y,y-5,y-4],color='#00857D',lw=1,alpha=.7)
    ax.plot(*positions[[0,1,2,0]].T,'--',color='#909BA4',lw=.8)
    ax.add_patch(Polygon([[47,6],[51,8],[51,4]],color='#D17B27'))
    ax.annotate('Intruder',xy=(49,6),xytext=(39,23),fontsize=8,
                arrowprops=dict(arrowstyle='-',color='#D17B27'))
    ax.annotate('',xy=(16,6),xytext=(47,6),arrowprops=dict(arrowstyle='->',color='#D17B27',lw=1.4))
    ax.text(-17,-31,'1  Which fleet mixture is compatible?\n2  Which change improves the task?',fontsize=8)
    ax.set(xlim=(-18,56),ylim=(-35,29),aspect='equal');ax.axis('off')
    _source_save(fig,directory,'fig1-task')

    fig,ax=plt.subplots(figsize=(3.5,3.2),layout='constrained')
    boxes=[(.18,.86,.64,.11,'Current measured fleet state','#204A68'),
        (.02,.65,.43,.12,'Learned SAC commands','#4477AA'),
        (.55,.65,.43,.12,'Boids commands','#8A9198'),
        (.12,.43,.76,.13,'M1: candidate motion + joint selection','#204A68'),
        (.12,.20,.76,.14,'M2: factual value + intervention gain\nKeep / intervene now / intervene later','#00857D'),
        (.18,.01,.64,.11,'Execute; observe the next state','#3D4650')]
    arrows=[((.3,.86),(.24,.77)),((.7,.86),(.76,.77)),((.24,.65),(.35,.56)),
        ((.76,.65),(.65,.56)),((.5,.43),(.5,.34)),((.5,.20),(.5,.12))]
    _source_boxes(ax,boxes,arrows,fontsize=7.6)
    _source_save(fig,directory,'fig2-architecture')

    example=data.get('six_source_mechanism_illustration') or data.get('mechanism_illustration')
    if example:
        fig,axes=plt.subplots(2,2,figsize=(7.2,4.6),layout='constrained')
        diagnostic=example['common_state_diagnosis']
        paths=np.asarray(diagnostic['paths']);i,j=diagnostic['pair']
        for boat,color in ((i,'#4477AA'),(j,'#D17B27')):
            for path in paths[boat]:
                axes[0,0].plot(path[:,0],path[:,1],color=color,alpha=.22,lw=.7)
            for label,style in (('independent','--'),('joint','-')):
                k=diagnostic['choices'][label][boat];path=paths[boat,k]
                axes[0,0].plot(path[:,0],path[:,1],style,color=color,lw=1.5,
                    label=f'D{boat+1}, {label}')
        axes[0,0].set(title=f'(a) Common-state forecast at {diagnostic["time"]:.1f} s',
            xlabel='x (m)',ylabel='y (m)',aspect='equal')
        axes[0,0].legend(fontsize=6,ncol=2,loc='upper center',bbox_to_anchor=(.5,-.25))
        heat=axes[0,1].imshow(diagnostic['pair_risk'],origin='lower',cmap='Blues',aspect='equal')
        for label,marker,color in (('independent','x','#C35B50'),('joint','o','#00857D')):
            k=diagnostic['choices'][label]
            style=dict(facecolors='none',edgecolors=color) if marker=='o' else dict(color=color)
            axes[0,1].scatter(k[j],k[i],marker=marker,s=58,linewidths=1.4,label=label,**style)
        axes[0,1].set(title='(b) Same candidate-pair cost',xlabel=f'D{j+1} candidate',ylabel=f'D{i+1} candidate')
        axes[0,1].set_xticks(range(9));axes[0,1].set_yticks(range(9))
        fig.colorbar(heat,ax=axes[0,1],shrink=.8,label='Predicted interaction cost')
        all_points=[point for trace in example['trajectories'].values() for point in trace['points']]
        all_xy=np.concatenate([np.vstack((point['attacker'],point['defenders'])) for point in all_points])
        bounds=np.vstack((all_xy.min(0)-4,all_xy.max(0)+4))
        bounds[0]=np.minimum(bounds[0],[-17,-17]);bounds[1]=np.maximum(bounds[1],[17,17])
        for ax,label,title in ((axes[1,0],'independent','(c) Independent: collision'),
                               (axes[1,1],'joint','(d) Joint: capture')):
            points=example['trajectories'][label]['points']
            attacker=np.asarray([p['attacker'] for p in points]);defenders=np.asarray([p['defenders'] for p in points])
            ax.add_patch(Circle((0,0),15,ec='#84929C',fc='#EDF2F5',lw=.8))
            ax.plot(*attacker.T,color='#D17B27',lw=1.6,label='Intruder')
            for boat in range(defenders.shape[1]):
                ax.plot(*defenders[:,boat].T,color=plt.cm.Blues(.35+.6*boat/defenders.shape[1]),lw=1)
                ax.scatter(*defenders[-1,boat],s=10,c='#204A68')
            ax.scatter(*attacker[-1],marker='x',s=24,c='#D17B27')
            ax.set(title=title,xlabel='x (m)',ylabel='y (m)',aspect='equal',xlim=bounds[:,0],ylim=bounds[:,1])
        _source_save(fig,directory,'fig3-joint-fusion')

    refined = data.get('six_source_value') or data.get('refined_value', {})
    ranking = refined.get('ranking_summary') or data.get('ranking_summary')
    has_ranking = bool(ranking and ranking['comparable_pairs'])
    fig,axes=plt.subplots(1,3 if has_ranking else 2,figsize=(7.2,2.8),layout='constrained')
    boxes=[(.02,.39,.22,.22,'Same state\nFrozen\npolicy','#204A68'),
        (.38,.66,.59,.20,'Factual choice\nFeedback → outcome $Y_0$','#4477AA'),
        (.38,.20,.59,.20,'One-vessel change\nFeedback → outcome $Y_z$','#00857D'),
        (.39,.01,.57,.12,'Paired gain: $Y_z-Y_0$','#3D4650')]
    _source_boxes(axes[0],boxes,[((.24,.55),(.38,.76)),((.24,.46),(.38,.30)),
        ((.67,.66),(.67,.40)),((.67,.20),(.67,.13))],fontsize=7)
    axes[0].text(.67,.53,'Shared future disturbance',ha='center',fontsize=7,color='#59636E')
    axes[0].set_title('(a) Conditional closed-loop comparison',fontsize=9)
    measurement=refined.get('choice_validation') or data.get('choice_validation')
    if measurement:
        maximum=.1
        for label,color,marker in [('conditional','#00857D','o'),('ordinary','#D17B27','x')]:
            predicted=[];measured=[];accepted=rescues=harms=0
            for record in measurement['records']:
                predicted.append(record['decisions'][label]['predicted_gain'][0])
                factual=np.asarray([r['success'] for r in record['outcomes']['factual']])
                value=np.asarray([r['success'] for r in record['outcomes'][label]])
                measured.append(float(np.mean(value-factual)))
                accepted+=bool(record['decisions'][label]['accepted'][0])
                rescues+=int(((value==1)&(factual==0)).sum())
                harms+=int(((value==0)&(factual==1)).sum())
            maximum=max(maximum,max(predicted))
            axes[1].scatter(predicted,measured,s=17,alpha=.65,c=color,marker=marker,
                label=f'{label}: {rescues} rescued, {harms} harmed')
        axes[1].plot([0,maximum],[0,maximum],':',color='#8B9198')
        axes[1].axhline(0,color='#CAD0D5',lw=.7)
        axes[1].set(xlabel='Predicted executed gain',ylabel='Measured paired gain',
            title='(b) New-state intervention decisions' if refined.get('choice_validation')
                else '(b) New-state decision validation',ylim=(-1.05,1.05))
        axes[1].legend(fontsize=6,loc='lower right')
    else:
        axes[1].axis('off');axes[1].text(.5,.65,'New-state decision validation',ha='center',fontsize=9)
        axes[1].text(.5,.40,'64 factual-trajectory states\n8 new paired futures per state\nMeasurement in progress',
                    ha='center',va='center',fontsize=8,color='#59636E')
    if has_ranking:
        for index,(label,color) in enumerate((('conditional','#00857D'),('ordinary','#D17B27'))):
            value = ranking['models'][label]; center = 100*value['concordance']
            interval = 100*np.asarray(value['ci95'])
            axes[2].bar(index,center,color=color,width=.65)
            axes[2].errorbar(index,center,yerr=[[center-interval[0]],[interval[1]-center]],
                color='#3D4650',capsize=3,fmt='none',lw=.8)
            axes[2].text(index,interval[1]+2,f'{center:.1f}',ha='center',fontsize=7)
        axes[2].axhline(50,color='#8B9198',ls=':',lw=.8)
        axes[2].set(xticks=(0,1),xticklabels=('Conditional','MLP'),ylim=(0,108),
            ylabel='Measured-order concordance (%)',title='(c) Random-candidate ranking')
        axes[2].tick_params(axis='x',labelsize=7)
        axes[2].text(.5,-.26,f"{ranking['informative_roots']}/{ranking['roots']} informative roots;\n"
            f"{ranking['comparable_pairs']} nontied comparisons",ha='center',
            transform=axes[2].transAxes,fontsize=6.5)
    _source_save(fig,directory,'fig4-intervention-value')


def source_result_figures(data, directory):
    numerical=data['numerical']
    configuration_curve=None
    configuration_spread=None
    configuration_updates=[0,6,12,18,24]
    pipeline_runs=data.get('independent_full_pipeline',{})
    complete_pipeline=len(pipeline_runs)==5 and all(
        all(all(entries.values()) for entries in run['frozen_predecessors'].values()) and
        all(run['configurations'].get(arm,{}).get(label) and
            all(run['configurations'][arm][label]['evaluations'].values())
            for arm in ('F01','F11') for label in ('conditional','ordinary')) and
        all(run['configurations'][arm]['conditional']['nodes'].get(str(update)) and
            all(run['configurations'][arm]['conditional']['nodes'][str(update)]['evaluations'].values())
            for arm in ('F01','F11') for update in (6000,12000,18000,24000))
        for run in pipeline_runs.values())
    if complete_pipeline:
        histories={label:[] for label in ('F00','F10','F01','F11')}
        for run in pipeline_runs.values():
            predecessors={label:{suite:dict(entry,tasks=entry['tasks'][::2],rows=entry['rows'][::2])
                for suite,entry in entries.items() if suite in ('source-development','new4310-200')}
                for label,entries in run['frozen_predecessors'].items()}
            reference=predecessors['F00']
            for label,entries in predecessors.items():
                histories[label].append([_source_rate(dict(rows=source_rows(entries)))]*5)
            for label,predecessor in [('F01','F00'),('F11','F10')]:
                values=[histories[predecessor][-1][0]]
                for update in (6000,12000,18000,24000):
                    entries=run['configurations'][label]['conditional']['nodes'][str(update)]['evaluations']
                    if set(entries)!=set(reference) or any(
                            entries[suite]['tasks']!=reference[suite]['tasks'] for suite in reference):
                        raise ValueError('Independent learning runs require the same paired task screen.')
                    values.append(_source_rate(dict(rows=source_rows(entries))))
                histories[label].append(values)
        configuration_curve={label:np.mean(values,axis=0) for label,values in histories.items()}
        configuration_spread={label:np.std(values,axis=0,ddof=1) for label,values in histories.items()}
    six_learning=data.get('six_source_configuration_learning',{}).get('configurations',{})
    six_learning_ready=all(six_learning.get(label,{}).get(str(update),{}).get('evaluations') and
        all(six_learning[label][str(update)]['evaluations'].values())
        for label in ('F01','F11') for update in (0,6000,12000,18000,24000,30000))
    if six_learning_ready:
        configuration_updates=[0,6,12,18,24,30]
        configuration_curve={};configuration_spread=None
        reference=six_learning['F01']['0']['evaluations']
        for label,predecessor in [('F01','F00'),('F11','F10')]:
            values=[]
            for update in (0,6000,12000,18000,24000,30000):
                entries=six_learning[label][str(update)]['evaluations']
                if set(entries)!=set(reference) or any(
                        entries[suite]['tasks']!=reference[suite]['tasks'] for suite in reference):
                    raise ValueError('Six-vessel learning curves require the same paired screen at every update.')
                values.append(_source_rate(dict(rows=source_rows(entries))))
            configuration_curve[label]=values
            configuration_curve[predecessor]=[values[0]]*len(values)
    for recipe in ('coverage','identity'):
        if configuration_curve is not None:break
        group=data.get('configuration_learning',{}).get(recipe,{})
        configurations=group.get('configurations',{})
        if not all(configurations.get(label) and all(
            configurations[label]['nodes'].get(str(update)) and
            all(configurations[label]['nodes'][str(update)]['evaluations'].values())
            for update in (6000,12000,18000,24000)) for label in ('F01','F11')):
            continue
        predecessors=group['frozen_predecessors']
        reference=predecessors['F00']
        scores={label:[_source_rate(dict(rows=source_rows(entries)))]*5
            for label,entries in predecessors.items()}
        for label,predecessor in [('F01','F00'),('F11','F10')]:
            scores[label]=[scores[predecessor][0]]
            for update in (6000,12000,18000,24000):
                entries=configurations[label]['nodes'][str(update)]['evaluations']
                if set(entries)!=set(reference) or any(
                    entries[suite]['tasks']!=reference[suite]['tasks'] for suite in reference):
                    raise ValueError('Configuration learning requires the same complete paired screen.')
                scores[label].append(_source_rate(dict(rows=source_rows(entries))))
        configuration_curve=scores
        break
    if configuration_curve is None:
        fig,axes=plt.subplots(1,3,figsize=(7.2,2.75),layout='constrained')
        overall_axis,repeat_axis=axes[1:]
    else:
        fig,grid=plt.subplots(2,2,figsize=(7.2,5.35),layout='constrained')
        axes=grid.ravel();overall_axis,repeat_axis=axes[2:]
        for label in ('F00','F10','F01','F11'):
            axes[1].plot(configuration_updates,configuration_curve[label],marker='o',ms=3,
                color=SOURCE_COLORS[label],label=label+(' (frozen)' if label in ('F00','F10') else ''))
            if configuration_spread is not None:
                mean=np.asarray(configuration_curve[label]);spread=configuration_spread[label]
                axes[1].fill_between(configuration_updates,np.maximum(0,mean-spread),
                    np.minimum(100,mean+spread),color=SOURCE_COLORS[label],alpha=.13,lw=0)
        axes[1].set(xlabel='Additional value updates (thousand)',ylabel='Success (%)',
            ylim=(0,102),xticks=configuration_updates,
            title='(b) Six-vessel value learning' if six_learning_ready else '(b) Matched refit learning')
        axes[1].legend(fontsize=6.5,ncol=2,loc='lower right')
        axes[1].text(.5,-.30,('228 paired episodes; zero denotes the factual predecessor'
            if six_learning_ready else 'Five complete pipelines; mean and SD on 228 paired episodes'
            if complete_pipeline else 'Separate refit recipe; 228 paired development episodes'),
            ha='center',transform=axes[1].transAxes,fontsize=6.5)
    for family,color in [('arboids','#4477AA'),('rp','#D17B27'),('sac','#8B9198')]:
        histories=[]
        for curve in data['learning_curves'].get(family,[]):
            history={int(row['step']):100*float(row['def_sr']) for row in curve['rows']}
            if not history:continue
            histories.append(history)
            steps=sorted(history)
            axes[0].plot(np.asarray(steps)/1e6,[history[step] for step in steps],color=color,lw=.5,alpha=.25)
        if histories:
            common=sorted(set.intersection(*(set(history) for history in histories)))
            values=np.asarray([[history[step] for step in common] for history in histories])
            x=np.asarray(common)/1e6;mean=values.mean(0)
            label={'arboids':'ARBoids','rp':'RP','sac':'SAC'}[family]
            axes[0].plot(x,mean,color=color,lw=1.2,label=f'{label} (n={len(histories)})')
            if len(histories)>1:
                spread=values.std(0,ddof=1)
                axes[0].fill_between(x,np.maximum(0,mean-spread),np.minimum(100,mean+spread),color=color,alpha=.15,lw=0)
    axes[0].set(xlim=(0,1),ylim=(0,100),xlabel='Training steps (million)',ylabel='Development success (%)',
        title='(a) Six-vessel baseline learning')
    complete=sum(curve['completed'] for curves in data['learning_curves'].values() for curve in curves)
    axes[0].text(.96,.03,f'{complete}/15 runs complete',ha='right',transform=axes[0].transAxes,fontsize=7,color='#59636E')
    axes[0].legend(loc='upper left')
    labels=['F00','F10','F01','F11']
    if complete_pipeline:
        distributions={label:[_source_rate(dict(rows=source_rows(
            run['frozen_predecessors'][label] if label in ('F00','F10') else
            run['configurations'][label]['conditional']['evaluations'])))
            for run in pipeline_runs.values()] for label in labels}
    else:
        distributions={label:[_source_rate(dict(rows=source_rows(numerical[label])))] for label in labels}
    baselines=data.get('trained_source_baselines',{})
    baseline_seeds=set.intersection(*[{seed for seed,entry in baselines.get(family,{}).items()
        if entry.get('evaluations') and all(entry['evaluations'].values())}
        for family in ('sac','rp','arboids')])
    colors=[SOURCE_COLORS[label] for label in labels]
    if baseline_seeds:
        for family,label in [('sac','SAC'),('rp','RP'),('arboids','ARBoids')]:
            distributions[label]=[_source_rate(dict(rows=source_rows(baselines[family][seed]['evaluations'])))
                for seed in sorted(baseline_seeds)]
        labels=['SAC','RP','ARBoids',*labels]
        colors=['#8B9198','#D17B27','#4477AA',*colors]
    six_value=data.get('six_source_value',{})
    six_complete=bool(six_value) and all(six_value.get(kind,{}).get('evaluations') and
        all(six_value[kind]['evaluations'].values()) for kind in ('conditional','ordinary'))
    if six_complete and '42' in baseline_seeds:
        labels=['SAC','RP','ARBoids','M1','MLP','Full']
        groups=[baselines[family]['42']['evaluations'] for family in ('sac','rp','arboids')]
        groups.extend([six_value['predecessor'],six_value['ordinary']['evaluations'],
            six_value['conditional']['evaluations']])
        six_configuration=data.get('six_source_configurations',{}).get('42',{})
        if all(six_configuration.get(label,{}).get('evaluations') and
                all(six_configuration[label]['evaluations'].values()) for label in ('F00','F01')):
            labels=['SAC','RP','ARBoids','F00','F10','F01','MLP','F11']
            groups=groups[:3]+[six_configuration['F00']['evaluations'],six_value['predecessor'],
                six_configuration['F01']['evaluations'],six_value['ordinary']['evaluations'],
                six_value['conditional']['evaluations']]
        distributions={label:[_source_rate(dict(rows=source_rows(group)))]
            for label,group in zip(labels,groups)}
        colors=(['#8B9198','#D17B27','#4477AA','#B6BCC1','#204A68','#AA66AA','#D9A063','#00857D']
            if len(labels)==8 else ['#8B9198','#D17B27','#4477AA','#204A68','#D9A063','#00857D'])
    values=[float(np.mean(distributions[label])) for label in labels]
    spreads=[float(np.std(distributions[label],ddof=1)) if len(distributions[label])>1 else 0.
             for label in labels]
    overall_axis.bar(range(len(labels)),values,color=colors,width=.65)
    if any(spreads):
        overall_axis.errorbar(range(len(labels)),values,yerr=spreads,fmt='none',color='#3D4650',capsize=2,lw=.7)
    overall_axis.set_xticks(range(len(labels)),labels,fontsize=6.5)
    for i,(value,spread) in enumerate(zip(values,spreads)):
        overall_axis.text(i,value+spread+2,f'{value:.1f}',ha='center',fontsize=6.5)
    panel='(b)' if configuration_curve is None else '(c)'
    overall_axis.set(ylim=(0,112),ylabel='Success (%)',title=panel+(' Six-vessel source comparison' if six_complete
        else ' Baselines and configurations'
        if baseline_seeds else ' Four configurations' if complete_pipeline else ' Reference four configurations'))
    annotation='856 paired development episodes'
    if six_complete:
        annotation+='\nSame six-vessel source, training seed 42\nMLP uses its development-selected gain margin'
    elif baseline_seeds:
        annotation+=f"\nBaselines: six-vessel training, n={len(baseline_seeds)}"
        annotation+=f"\nF00–F11: three-to-six source transfer, n={5 if complete_pipeline else 1}"
    overall_axis.text(.5,-.30,annotation,ha='center',transform=overall_axis.transAxes,fontsize=6.5)
    six_repeats=data.get('same_fleet_m1',{})
    six_repeat_seeds=[str(seed) for seed in range(42,47)]
    six_repeats_ready=all(seed in baseline_seeds and seed in six_repeats and
        all(six_repeats[seed].get(kind,{}).get('evaluations') and
            all(six_repeats[seed][kind]['evaluations'].values())
            for kind in ('joint','independent')) for seed in six_repeat_seeds)
    six_pipeline=data.get('six_source_configurations',{})
    six_pipeline_ready=all(six_pipeline.get(seed,{}).get(label,{}).get('evaluations') and
        all(six_pipeline[seed][label]['evaluations'].values())
        for seed in six_repeat_seeds for label in ('F00','F10','F01','F11','ordinary_F01','ordinary_F11'))
    if six_pipeline_ready:
        series=[('F00','F00','#8B9198','^','-'),('F10','F10','#204A68','s','-'),
            ('F01','F01','#9A6A30','o','-'),('F11','F11','#00857D','o','-'),
            ('MLP on F00','ordinary_F01','#D9A063','v','--'),
            ('MLP on F10','ordinary_F11','#D17B27','s','--')]
        for label,key,color,marker,linestyle in series:
            scores=[_source_rate(dict(rows=source_rows(six_pipeline[seed][key]['evaluations'])))
                for seed in six_repeat_seeds]
            repeat_axis.plot(range(42,47),scores,marker=marker,color=color,lw=1,
                linestyle=linestyle,label=label)
        repeat_axis.set(ylim=(0,102),xticks=range(42,47),xlabel='Independent pipeline seed',
            ylabel='Success (%)',title='(d) Five complete learning pipelines')
        repeat_axis.text(.5,-.30,'Independent source learning and own-predecessor value data',
            ha='center',transform=repeat_axis.transAxes,fontsize=6.5)
    elif six_repeats_ready:
        for label,kind,color,marker in [('ARBoids','source','#8B9198','^'),
                ('Independent','independent','#C35B50','s'),('Joint','joint','#204A68','o')]:
            groups=[baselines['arboids'][seed]['evaluations'] if kind=='source' else
                six_repeats[seed][kind]['evaluations'] for seed in six_repeat_seeds]
            scores=[_source_rate(dict(rows=source_rows(group))) for group in groups]
            repeat_axis.plot(range(42,47),scores,marker=marker,color=color,lw=1,label=label)
        repeat_axis.set(ylim=(0,102),xticks=range(42,47),xlabel='Independent source-training seed',
            ylabel='Success (%)',title='(d) Five source and fusion runs')
        repeat_axis.text(.5,-.30,'Same one-million-step source within each paired comparison',
            ha='center',transform=repeat_axis.transAxes,fontsize=6.5)
    else:
        for label,color,marker in [('conditional','#00857D','o'),('ordinary','#D17B27','s')]:
            seeds=[];scores=[]
            repeats=({seed:{kind:list(run['configurations']['F11'][kind]['evaluations'].values())
                        for kind in ('conditional','ordinary')} for seed,run in pipeline_runs.items()}
                     if complete_pipeline else data['value_repeats'])
            for seed,entry in repeats.items():
                if all(entry[label]):
                    rows=[r for e in entry[label] for r in e['rows']]
                    seeds.append(int(seed));scores.append(100*np.mean([r['success'] for r in rows]))
            repeat_axis.plot(seeds,scores,marker=marker,color=color,lw=1,label=label)
        repeat_axis.set(ylim=(0,100) if complete_pipeline else (75,100),xticks=range(42,47),
            xlabel='Independent pipeline seed' if complete_pipeline else 'Independent value-fit seed',ylabel='Success (%)',
            title=('(d) Five complete pipelines' if complete_pipeline else
                '(c) Five value-learning runs' if configuration_curve is None else '(d) Five value-learning runs'))
    repeat_axis.legend(loc='lower left',fontsize=6.5,ncol=2 if six_pipeline_ready else 1)
    _source_save(fig,directory,'fig5-overall-performance')

    same_fleet=data.get('same_fleet_m1',{}).get('42',{})
    same_fleet_ready=all(same_fleet.get(kind,{}).get('evaluations') and
        all(same_fleet[kind]['evaluations'].values()) for kind in ('joint','independent'))
    margin_controls=data.get('ordinary_margin_controls',{})
    margins_ready=all(margin_controls.get(label,{}).get('evaluations') and
        all(margin_controls[label]['evaluations'].values())
        for label in ('same-margin','margin008','unsupported-margin'))
    refined=data.get('six_source_value') or data.get('refined_value',{})
    refined_ready=all(refined.get(label,{}).get('evaluations') and
        all(refined[label]['evaluations'].values()) for label in ('conditional','ordinary'))
    six_configuration=data.get('six_source_configurations',{}).get('42',{})
    f01_ready=all(six_configuration.get(label,{}).get('evaluations') and
        all(six_configuration[label]['evaluations'].values()) for label in ('F01','ordinary_F01'))
    six_core_ready=same_fleet_ready and all(six_configuration.get(label,{}).get('evaluations') and
        all(six_configuration[label]['evaluations'].values())
        for label in ('F00','F10','F01','F11','ordinary_F01','ordinary_F11'))
    graph_comparisons=[]
    if six_core_ready:
        for arm,color in [('F01','#866340'),('F11','#447B73')]:
            for entry in data.get('six_source_graph_returns',{}).get(arm,{}).values():
                if (entry['conditional']['policy_sha256']!=six_configuration[arm]['policy_sha256'] or
                    not all(entry[label]['evaluations'] and all(entry[label]['evaluations'].values())
                        for label in ('conditional','ordinary_graph','ordinary_mlp'))):
                    continue
                graph_comparisons.append((f'M2: {arm} minus tuned graph return',
                    entry['conditional']['evaluations'],entry['ordinary_graph']['evaluations'],color))
                break
    extra_height=.84 if six_core_ready else .28*(same_fleet_ready+margins_ready+2*refined_ready+f01_ready)
    extra_height+=.28*len(graph_comparisons)
    fig=plt.figure(figsize=(7.2,4.9+extra_height),layout='constrained')
    top,bottom=fig.subfigures(2,1,height_ratios=(2.6,1.45+extra_height))
    axes=top.subplots(1,2)
    contribution_axis=bottom.subplots()
    labels=['state-interaction','candidate-interaction','M1-independent','F10','M1-strong']
    generic=data.get('generic_replacements',{}).get('balanced',{})
    stronger_generic=bool(generic) and all(generic.get(mode,{}).get('evaluations') and
        all(generic[mode]['evaluations'].values()) for mode in ('state','candidate'))
    groups=dict(numerical)
    if stronger_generic:
        groups.update({'state-interaction':generic['state']['evaluations'],
            'candidate-interaction':generic['candidate']['evaluations']})
    values=[100*np.mean([r['success'] for r in source_rows(groups[k])]) for k in labels]
    axes[0].bar(range(len(labels)),values,color=[SOURCE_COLORS[k] for k in labels])
    axes[0].set_xticks(range(len(labels)),['State\nattention','Candidate\nattention','Independent\nselection','Joint','Refined\njoint'],fontsize=6.5)
    for i,y in enumerate(values):axes[0].text(i,y+1.5,f'{y:.1f}',ha='center',fontsize=7)
    axes[0].set(ylim=(0,105),ylabel='Success (%)',title='(a) Transferred-source M1 replacements')
    repeat_gains=[]
    for entry in data.get('source_backbone_repeats',{}).values():
        if all(entry['joint']) and all(entry['independent']):
            for joint,independent in zip(entry['joint'],entry['independent']):
                if joint['tasks']!=independent['tasks']:
                    raise ValueError('Source-backbone repeats require paired tasks.')
            joint=[r['success'] for e in entry['joint'] for r in e['rows']]
            independent=[r['success'] for e in entry['independent'] for r in e['rows']]
            repeat_gains.append(100*np.mean(np.asarray(joint)-np.asarray(independent)))
    if repeat_gains:
        axes[0].text(.5,-.27,f'Separate 228-episode screens, {len(repeat_gains)} source backbones:\n'
            f'mean joint-selection gain {np.mean(repeat_gains):.1f} pp '
            f'(range {min(repeat_gains):.1f}–{max(repeat_gains):.1f})',
            ha='center',transform=axes[0].transAxes,fontsize=6.5)
    six_replacements=data.get('six_source_replacements',{})
    replacement_models=six_replacements.get('models',{})
    replacement_ready=same_fleet_ready and (
        six_replacements.get('teacher_sha256')==same_fleet['joint']['policy_sha256']) and all(
        replacement_models.get(label,{}).get('evaluations') and
        all(replacement_models[label]['evaluations'].values()) for label in ('state','candidate','single-reference'))
    if replacement_ready:
        axes[0].clear()
        replacements=[('State\nattention',replacement_models['state']['evaluations'],'#9AAFC0'),
            ('Candidate\nattention',replacement_models['candidate']['evaluations'],'#7F98AF'),
            ('Single\nreference',replacement_models['single-reference']['evaluations'],'#B6B0A2'),
            ('Independent\nselection',same_fleet['independent']['evaluations'],'#C35B50'),
            ('Joint\nselection',same_fleet['joint']['evaluations'],'#204A68')]
        for i,(label,group,color) in enumerate(replacements):
            source_paired(group,same_fleet['joint']['evaluations'],repetitions=100)
            value=100*np.mean([row['success'] for row in source_rows(group)])
            axes[0].bar(i,value,color=color);axes[0].text(i,value+1.5,f'{value:.1f}',ha='center',fontsize=7)
        axes[0].set_xticks(range(len(replacements)),[x[0] for x in replacements],fontsize=6.5)
        axes[0].set(ylim=(0,105),ylabel='Success (%)',title='(a) Six-vessel M1 replacements')
        axes[0].text(.5,-.27,'856 paired episodes; same learned source',
            ha='center',transform=axes[0].transAxes,fontsize=6.5)
    suites=list(numerical['F11'])[:2]
    subset=lambda label:{suite:dict(numerical[label][suite],tasks=numerical[label][suite]['tasks'][::2],
        rows=numerical[label][suite]['rows'][::2]) for suite in suites}
    controls=[('Ordinary\nMLP',subset('ordinary'),'#D17B27')]
    matched_margin=data.get('ordinary_margin_controls',{}).get('same-margin',{})
    if matched_margin.get('evaluations') and all(matched_margin['evaluations'].values()):
        group={suite:dict(matched_margin['evaluations'][suite],
            tasks=matched_margin['evaluations'][suite]['tasks'][::2],
            rows=matched_margin['evaluations'][suite]['rows'][::2]) for suite in suites}
        source_paired(group,subset('F11'),repetitions=100)
        controls.append((f"MLP\nmargin {matched_margin['gain_margin']:.2f}",group,'#D9A063'))
    names={'F11-r107-immediate-only':'Immediate\nonly','F11-r107-no-temporal-bounds':'No time\nbounds',
           'F11-r107-no-factual-anchor':'No factual\nanchor'}
    pending=0
    for stem,entries in data['components'].items():
        if all(entries):
            group=dict(zip(suites,entries))
            source_paired(group,subset('F11'),repetitions=100)
            controls.append((names[stem],group,'#99B8B3'))
        else:pending+=1
    controls.append(('Full\nM2',subset('F11'),'#00857D'))
    for i,(label,group,color) in enumerate(controls):
        y=100*np.mean([r['success'] for r in source_rows(group)])
        axes[1].bar(i,y,color=color);axes[1].text(i,y+1.5,f'{y:.1f}',ha='center',fontsize=7)
    axes[1].set_xticks(range(len(controls)),[x[0] for x in controls],fontsize=6.5)
    axes[1].set(ylim=(0,105),ylabel='Success (%)',title='(b) Transferred-source M2 components')
    axes[1].text(.5,-.27,'228 paired screen episodes'+(f'; {pending} controls running' if pending else ''),
        ha='center',transform=axes[1].transAxes,fontsize=6.5)
    six_components=data.get('six_source_components',{})
    component_models=six_components.get('controls',{})
    component_ready=refined_ready and (
        six_components.get('reference',{}).get('policy_sha256')==refined['conditional']['policy_sha256']) and (
        six_components.get('ordinary',{}).get('evaluations') and
        all(six_components['ordinary']['evaluations'].values())) and all(
        component_models.get(label,{}).get('evaluations') and
        all(component_models[label]['evaluations'].values())
        for label in ('immediate-only','no-factual-calibration','neural-value-only'))
    if component_ready:
        axes[1].clear()
        controls=[('Tuned\nordinary MLP',six_components['ordinary']['evaluations'],'#D17B27')]
        for label,title in [('immediate-only','Immediate\nonly'),
                ('no-factual-calibration','No factual\ncalibration'),('neural-value-only','Neural\nvalue only')]:
            controls.append((title,component_models[label]['evaluations'],'#99B8B3'))
        controls.append(('Full\nM2',six_components['reference']['evaluations'],'#00857D'))
        for i,(label,group,color) in enumerate(controls):
            source_paired(group,six_components['reference']['evaluations'],repetitions=100)
            value=100*np.mean([row['success'] for row in source_rows(group)])
            axes[1].bar(i,value,color=color);axes[1].text(i,value+1.5,f'{value:.1f}',ha='center',fontsize=7)
        axes[1].set_xticks(range(len(controls)),[x[0] for x in controls],fontsize=6.5)
        axes[1].set(ylim=(0,105),ylabel='Success (%)',title='(b) Six-vessel M2 components')
        axes[1].text(.5,-.27,'856 paired episodes; frozen value parameters',
            ha='center',transform=axes[1].transAxes,fontsize=6.5)
    contributions=[('Joint minus independent (3-to-6 source)',numerical['M1-strong'],numerical['M1-independent'],'#204A68'),
        ('Full minus M1 only',numerical['F11'],numerical['F10'],'#4477AA'),
        ('Full minus ordinary MLP',numerical['F11'],numerical['ordinary'],'#00857D')]
    if same_fleet_ready:
        contributions.insert(1,('Joint minus independent (6-vessel source)',
            same_fleet['joint']['evaluations'],same_fleet['independent']['evaluations'],'#5C72A6'))
    if margins_ready:
        best_margin=max(margin_controls.values(),key=lambda entry:sum(
            row['success'] for row in source_rows(entry['evaluations'])))
        contributions.append((f"Full minus MLP (margin {best_margin['gain_margin']:.2f})",
            numerical['F11'],best_margin['evaluations'],'#D17B27'))
    if refined_ready:
        prefix='Six-vessel full' if data.get('six_source_value') else 'Refit'
        contributions.extend([
            (prefix+' minus its M1 predecessor',refined['conditional']['evaluations'],
                refined['predecessor'],'#557A46'),
            (prefix+' minus tuned ordinary MLP' if data.get('six_source_value') else
                'Refit minus its matched ordinary MLP',refined['conditional']['evaluations'],
                refined['ordinary']['evaluations'],'#00857D')])
    if f01_ready:
        contributions.append(('Six-vessel F01 minus tuned ordinary MLP',
            six_configuration['F01']['evaluations'],
            six_configuration['ordinary_F01']['evaluations'],'#9A6A30'))
    if six_core_ready:
        configurations={label:entry['evaluations'] for label,entry in six_configuration.items()
            if isinstance(entry,dict) and entry.get('evaluations')}
        contributions=[('M1: joint minus independent',same_fleet['joint']['evaluations'],
                same_fleet['independent']['evaluations'],'#204A68'),
            ('M1: F11 minus F01',configurations['F11'],configurations['F01'],'#5C72A6'),
            ('M2: F01 minus F00',configurations['F01'],configurations['F00'],'#9A6A30'),
            ('M2: F11 minus F10',configurations['F11'],configurations['F10'],'#557A46'),
            ('M2: F01 minus tuned ordinary MLP',configurations['F01'],configurations['ordinary_F01'],'#BB8B45'),
            ('M2: F11 minus tuned ordinary MLP',configurations['F11'],configurations['ordinary_F11'],'#00857D')]
        contributions.extend(graph_comparisons)
    effects=[source_paired(left,right) for _,left,right,_ in contributions]
    for y,((label,left,right,color),effect) in enumerate(zip(contributions,effects)):
        value=effect['difference_pp'];low,high=effect['ci95_pp']
        contribution_axis.errorbar(value,y,xerr=[[value-low],[high-value]],fmt='D',
            color=color,markersize=4,capsize=2,elinewidth=1)
        contribution_axis.text(high+.35,y,f'{value:+.1f} [{low:.1f}, {high:.1f}]',
            va='center',fontsize=7,color=color)
    contribution_axis.axvline(0,color='#A8AFB4',ls='--',lw=.7)
    contribution_axis.set(yticks=range(len(contributions)),
        yticklabels=[label for label,*_ in contributions],
        ylim=(len(contributions)-.5,-.5),
        xlim=(min(-.5,min(effect['ci95_pp'][0] for effect in effects)-1),
            max(effect['ci95_pp'][1] for effect in effects)+6),
        xlabel='Paired success-rate gain (percentage points)\n95% layout-cluster intervals; training seed fixed',
        title='(c) '+('Six-vessel ' if six_core_ready else '')+'paired contributions on 856 development episodes')
    contribution_axis.tick_params(axis='y',labelsize=7)
    contribution_axis.xaxis.label.set_fontsize(7)
    _source_save(fig,directory,'fig6-mechanism-controls')

    fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),layout='constrained')
    for label in ['F00','F01','M1-independent','F10','M1-strong','F11']:
        rows=source_rows(numerical[label]);agility=sorted(set(row['agility'] for row in rows))
        scores=[100*np.mean([r['success'] for r in rows if r['agility']==a]) for a in agility]
        axes[0].plot(agility,scores,marker='o',ms=3,color=SOURCE_COLORS[label],label=SOURCE_LABELS[label])
        entries=data['fleet'].get(label,{})
        counts=sorted(int(n) for n,entry in entries.items() if entry)
        scores=[_source_rate(entries[str(n)]) for n in counts]
        axes[1].plot(counts,scores,marker='o',ms=3,color=SOURCE_COLORS[label])
    axes[0].set(xlabel='Intruder agility',ylabel='Success (%)',ylim=(0,102),title='(a) Agility')
    axes[0].legend(fontsize=5.5,ncol=2)
    axes[1].set(xlabel='Number of defenders',ylabel='Success (%)',xticks=(3,4,5,7,8),ylim=(0,102),title='(b) Fleet-size screen')
    axes[1].text(.5,.02,'16 scenes per fleet size',ha='center',transform=axes[1].transAxes,fontsize=6.5)
    adaptation=data.get('common_opponent',{})
    fresh=all(adaptation.get('fresh_before',{}).get(label) and
        adaptation.get('fresh_after',{}).get(label) for label in ('F01','F11','ordinary'))
    before_key,after_key=('fresh_before','fresh_after') if fresh else ('before','after')
    ready=[label for label in ('F01','F11','ordinary')
        if adaptation.get(before_key,{}).get(label) and adaptation.get(after_key,{}).get(label)]
    if ready:
        for index,label in enumerate(ready):
            before=adaptation[before_key][label];after=adaptation[after_key][label]
            if (before['tasks']!=after['tasks'] or
                    before['evaluation_context']['frozen_opponent_sha256']!=after['evaluation_context']['frozen_opponent_sha256']):
                raise ValueError('Adaptation recovery needs the same layouts and frozen opponent.')
            axes[2].bar(index-.17,_source_rate(before),width=.32,color=SOURCE_COLORS[label],alpha=.35)
            axes[2].bar(index+.17,_source_rate(after),width=.32,color=SOURCE_COLORS[label])
            retained=adaptation.get('retained_apf',{}).get(label,{})
            if retained and all(retained.values()):
                contrast=source_paired(retained,numerical[label],repetitions=100)
                axes[2].text(index,3,f'APF {contrast["difference_pp"]:+.1f} pp',ha='center',fontsize=5.5)
        axes[2].set(xticks=range(len(ready)),xticklabels=[SOURCE_LABELS[label] for label in ready],
            ylim=(0,108),ylabel='Success (%)',title='(c) Common-opponent adaptation')
        axes[2].tick_params(axis='x',labelsize=6)
        axes[2].text(.5,-.24,('64 new layouts' if fresh else '64 adaptation-development layouts')+
            '; light: before, dark: after\nAPF: change on the original paired episodes',
            ha='center',transform=axes[2].transAxes,fontsize=6)
    else:
        axes[2].axis('off');axes[2].set_title('(c) Common-opponent adaptation')
        axes[2].text(.5,.55,'Common attacker training\n↓\nPaired changed-opponent evaluation\n↓\nAdaptation + original-opponent retention',
            ha='center',va='center',fontsize=7.5)
        axes[2].text(.5,.13,'Evaluation in progress',ha='center',fontsize=7,color='#59636E')
    expected_scope={label:item['policy_sha256'] for label,item in six_configuration.items()
                    if label in ('F00','F10','F01','F11','ordinary_F01','ordinary_F11')}
    if same_fleet_ready:expected_scope['independent']=same_fleet['independent']['policy_sha256']
    scope=data.get('six_source_scope',{})
    scope_colors={'F00':'#7B8794','F10':'#4477AA','F01':'#78A854','F11':'#00857D',
                  'independent':'#C35B50','ordinary_F01':'#865E9E','ordinary_F11':'#D17B27'}
    scope_labels={'F00':'Source','F10':'Joint','F01':'Value only','F11':'Full',
                  'independent':'Independent','ordinary_F01':'MLP on source','ordinary_F11':'MLP on joint'}
    scope_matches=len(expected_scope)==7 and all(scope.get('models',{}).get(label,{}).get('sha256')==identity
        for label,identity in expected_scope.items())
    if scope_matches:
        for axis,kind,labels,levels in [(axes[0],'agility',list(scope_colors),(2.,2.5,3.,3.5)),
                (axes[1],'fleet',['F00','F10','independent','F01','F11'],(3,4,5,7,8))]:
            cells=scope.get(kind,{})
            if not all(cells.get(label,{}).get(str(level)) for label in labels for level in levels):continue
            for level in levels:
                tasks=cells[labels[0]][str(level)]['tasks']
                if any(cells[label][str(level)]['tasks']!=tasks for label in labels):
                    raise ValueError('Scope comparisons require paired layouts in every cell.')
            axis.clear()
            for label in labels:
                axis.plot(levels,[_source_rate(cells[label][str(level)]) for level in levels],
                    marker='o',ms=3,lw=1,color=scope_colors[label],label=scope_labels[label])
            axis.set(ylim=(0,102),ylabel='Success (%)',xticks=levels,
                xlabel='Intruder agility' if kind=='agility' else 'Number of defenders',
                title='(a) Six-vessel agility' if kind=='agility' else '(b) Shared-policy fleet transfer')
            if kind=='agility':axis.legend(fontsize=5.2,ncol=2)
            axis.text(.5,.02,'64 paired layouts per cell',ha='center',transform=axis.transAxes,fontsize=6)
    adapted=data.get('six_source_opponent_adaptation',{})
    adapted_labels=('F01','ordinary_F01','F11','ordinary_F11')
    adapted_context=adapted.get('context') or {}
    adapted_ready=len(expected_scope)==7 and all(adapted_context.get('before_sha256',{}).get(label)==identity
        for label,identity in expected_scope.items()) and all(adapted.get(stage,{}).get(label,{}).get(environment)
        for stage in ('before','after') for label in adapted_labels for environment in ('APF','opponent'))
    if adapted_ready:
        axes[2].clear()
        for index,label in enumerate(adapted_labels):
            before=adapted['before'][label];after=adapted['after'][label]
            for environment in ('APF','opponent'):
                if before[environment]['tasks']!=after[environment]['tasks']:
                    raise ValueError('Fresh adaptation and retention outcomes must share their layouts.')
            if before['opponent']['evaluation_context']['frozen_opponent_sha256']!=after['opponent']['evaluation_context']['frozen_opponent_sha256']:
                raise ValueError('Adaptation must use one frozen learned opponent.')
            for shift,record,alpha in [(-.17,before['opponent'],.35),(.17,after['opponent'],1.)]:
                axes[2].bar(index+shift,_source_rate(record),width=.32,color=scope_colors[label],alpha=alpha)
            retention=source_paired({'fresh':after['APF']},{'fresh':before['APF']},repetitions=100)
            axes[2].text(index,3,f'APF {retention["difference_pp"]:+.1f}',ha='center',fontsize=5.5)
        axes[2].set(xticks=range(4),xticklabels=['Value\nonly','Source\nMLP','Full','Joint\nMLP'],
            ylim=(0,108),ylabel='Success (%)',title='(c) Learned-opponent recovery')
        axes[2].tick_params(axis='x',labelsize=6)
        axes[2].text(.5,-.27,'128 fresh paired layouts; light: before, dark: after\nAPF: paired retention change (pp)',
            ha='center',transform=axes[2].transAxes,fontsize=6)
    _source_save(fig,directory,'fig7-scope-adaptation')

    fig,axes=plt.subplots(1,3,figsize=(7.2,3.2),layout='constrained')
    labels=['source','m1-base','independent-r33','m1-r50','m2-only-r65','full-r61',
            'ordinary-r99-matched','full-r97-short-response','ordinary-r97-short-response']
    pretty=['Source','Joint base','Independent','Refined joint','Value only','Reference full',
            'Matched MLP','Refit full','Refit MLP']
    groups={label:[r for r in data['vrx'] if r['comparison_label']==label] for label in labels}
    new_labels=['F00','F10','independent','F01','ordinary_F01','ordinary_F11','F11']
    new_groups={label:[r for r in data.get('six_source_vrx',[]) if r['comparison_label']==label]
                for label in new_labels}
    new_vrx_ready=len(expected_scope)==7 and all(len(new_groups[label])==8 and all(
        record.get('passed') and record.get('selected_checkpoint_sha256')==expected_scope[label]
        for record in new_groups[label]) for label in new_labels)
    if new_vrx_ready:
        labels=new_labels;groups=new_groups
        pretty=[scope_labels[label] for label in labels]
    if not all(len(v)==8 for v in groups.values()):
        raise ValueError('VRX figure requires all eight paired scenes for every displayed method.')
    identities=[[(r['setting'],r['agility'],r['seed']) for r in groups[label]] for label in labels]
    if not all(set(x)==set(identities[0]) for x in identities):
        raise ValueError('VRX methods must share setting, agility and seed.')
    captures=[sum(r['success'] for r in groups[k]) for k in labels]
    collisions=[sum(r['defender_collision'] for r in groups[k]) for k in labels]
    breaches=[8-a-b for a,b in zip(captures,collisions)]
    axes[0].barh(range(len(labels)),captures,color='#00857D',label='Capture')
    axes[0].barh(range(len(labels)),collisions,left=captures,color='#C35B50',label='Collision')
    axes[0].barh(range(len(labels)),breaches,left=np.asarray(captures)+collisions,color='#D17B27',label='Breach')
    axes[0].set_yticks(range(len(labels)),pretty,fontsize=6.5);axes[0].invert_yaxis()
    axes[0].set(xlim=(0,8),xticks=(0,2,4,6,8),xlabel='Episodes',title='(a) Eight paired VRX scenes')
    axes[0].legend(fontsize=6,loc='lower left',bbox_to_anchor=(0,-.35),ncol=2)
    pair_labels=('full-r61','full-r97-short-response');scene_key=(1,2.5,432461)
    if new_vrx_ready:
        pair_labels=('F10','F11')
        scene_key=min((r['setting'],r['agility'],r['seed']) for r in groups['F11'])
        found=False
        for left,right in [('F10','F11'),('ordinary_F11','F11'),('independent','F10')]:
            first={(r['setting'],r['agility'],r['seed']):r for r in groups[left]}
            second={(r['setting'],r['agility'],r['seed']):r for r in groups[right]}
            rescues=[key for key in sorted(first) if not first[key]['success'] and second[key]['success']]
            if rescues:
                pair_labels=(left,right);scene_key=rescues[0];found=True;break
        if not found:
            first={(r['setting'],r['agility'],r['seed']):r for r in groups['F10']}
            failures=sorted((r['setting'],r['agility'],r['seed']) for r in groups['F11'] if not r['success'])
            if failures:scene_key=failures[0]
    scene_records=[next(r for r in groups[label] if (r['setting'],r['agility'],r['seed'])==scene_key)
                   for label in pair_labels]
    scene_xy=np.concatenate([np.concatenate((np.asarray(r['trace']['AttPos']).reshape(-1,2),
        np.asarray(r['trace']['DefPos']).reshape(-1,2))) for r in scene_records])
    bounds=np.vstack((scene_xy.min(0)-4,scene_xy.max(0)+4))
    bounds[0]=np.minimum(bounds[0],[-17,-17]);bounds[1]=np.maximum(bounds[1],[17,17])
    for letter,ax,label,record in zip(('b','c'),axes[1:],pair_labels,scene_records):
        outcome='capture' if record['success'] else 'collision' if record['defender_collision'] else 'breach'
        name=scope_labels[label] if new_vrx_ready else ('Reference' if label=='full-r61' else 'Refit')
        title=f'({letter}) {name}: {outcome}'
        trace=record['trace'];attacker=np.asarray(trace['AttPos']);defenders=np.asarray(trace['DefPos']).reshape(-1,6,2)
        ax.add_patch(Circle((0,0),15,ec='#84929C',fc='#EDF2F5',lw=.8))
        ax.plot(*attacker.T,color='#D17B27',lw=1.6)
        for boat in range(6):
            ax.plot(*defenders[:,boat].T,color=plt.cm.Blues(.35+.6*boat/6),lw=.9)
            ax.scatter(*defenders[-1,boat],s=8,c='#204A68')
        ax.scatter(*attacker[-1],s=25,marker='x',c='#D17B27')
        ax.set(xlim=bounds[:,0],ylim=bounds[:,1],aspect='equal',xlabel='x (m)',ylabel='y (m)',title=title)
    _source_save(fig,directory,'fig8-vrx-comparison')


def build_source_evidence(path, directory):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('format')!='source-mechanism-paper-evidence-v1' or data.get('role')!='development':
        raise ValueError('This exporter requires explicitly identified development evidence.')
    numerical=data['numerical']
    for label,group in numerical.items():
        if not all(group.values()):raise ValueError(f'Incomplete primary evidence: {label}')
    comparisons={}
    for name,left,right in [('joint_vs_independent','M1-strong','M1-independent'),
                            ('complete_vs_predecessor','F11','F10'),
                            ('conditional_vs_ordinary','F11','ordinary')]:
        comparisons[name]=source_paired(numerical[left],numerical[right])
        comparisons[name]['task_outcomes']=source_task_consequences(numerical[left],numerical[right])
    if data.get('catalog_ranking'):
        data['ranking_summary'] = source_ranking_summary(data['catalog_ranking'])
        comparisons['random_candidate_ranking'] = data['ranking_summary']
    refined=data.get('refined_value',{})
    if refined and all(refined.get(label,{}).get('evaluations') and
            all(refined[label]['evaluations'].values()) for label in ('conditional','ordinary')):
        for label,right in [('refined_predecessor',refined['predecessor']),
                            ('matched_ordinary',refined['ordinary']['evaluations'])]:
            comparison=source_paired(refined['conditional']['evaluations'],right)
            comparison['task_outcomes']=source_task_consequences(refined['conditional']['evaluations'],right)
            comparisons['refitted_conditional_vs_'+label]=comparison
        if refined.get('choice_validation'):
            refined['ranking_summary']=source_ranking_summary(refined['choice_validation'])
            comparisons['refitted_random_candidate_ranking']=refined['ranking_summary']
    six_value=data.get('six_source_value',{})
    if six_value and all(six_value.get(label,{}).get('evaluations') and
            all(six_value[label]['evaluations'].values()) for label in ('conditional','ordinary')):
        for label,right in [('predecessor',six_value['predecessor']),
                            ('tuned_ordinary',six_value['ordinary']['evaluations'])]:
            comparison=source_paired(six_value['conditional']['evaluations'],right)
            comparison['task_outcomes']=source_task_consequences(six_value['conditional']['evaluations'],right)
            comparisons['six_source_conditional_vs_'+label]=comparison
        six_value['ranking_summary']=source_ranking_summary(six_value['choice_validation'])
        comparisons['six_source_random_candidate_ranking']=six_value['ranking_summary']
    for seed,configuration in data.get('six_source_configurations',{}).items():
        if all(configuration.get(label,{}).get('evaluations') and
                all(configuration[label]['evaluations'].values()) for label in ('F01','ordinary_F01')):
            left=configuration['F01']['evaluations'];right=configuration['ordinary_F01']['evaluations']
            contrast=source_paired(left,right)
            contrast['task_outcomes']=source_task_consequences(left,right)
            comparisons[f'six_source_F01_vs_tuned_ordinary_seed{seed}']=contrast
    for label,entry in data.get('ordinary_margin_controls',{}).items():
        if entry.get('evaluations') and all(entry['evaluations'].values()):
            comparisons['conditional_vs_plain_'+label]=source_paired(numerical['F11'],entry['evaluations'])
    for family,seeds in data.get('trained_source_baselines',{}).items():
        for seed,entry in seeds.items():
            if entry.get('evaluations') and all(entry['evaluations'].values()):
                comparisons[f'reference_complete_vs_fully_trained_{family}_seed{seed}']=source_paired(
                    numerical['F11'],entry['evaluations'])
    for seed,entry in data.get('same_fleet_m1',{}).items():
        if all(entry.get(kind,{}).get('evaluations') and all(entry[kind]['evaluations'].values())
                for kind in ('joint','independent')):
            comparison=source_paired(entry['joint']['evaluations'],entry['independent']['evaluations'])
            comparison['task_outcomes']=source_task_consequences(
                entry['joint']['evaluations'],entry['independent']['evaluations'])
            comparisons[f'same_fleet_joint_vs_independent_seed{seed}']=comparison
    source_method_figures(data,directory)
    source_result_figures(data,directory)
    print(json.dumps(dict(role=data['role'],comparisons=comparisons,output=str(directory)),indent=2))



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-evidence',type=Path,help='Current source-environment paired evidence JSON')
    parser.add_argument('--study',type=Path,default=ROOT/'train/experiments/ia-crrl-20261007')
    parser.add_argument('--output',type=Path,default=ROOT/'paper/figures/interaction')
    parser.add_argument('--manuscript',type=Path,default=ROOT/'paper/interaction_aware_residual_rl.tex')
    parser.add_argument('--draft-only',action='store_true')
    parser.add_argument('--without-extensions',action='store_true')
    args=parser.parse_args()
    if args.source_evidence:
        build_source_evidence(args.source_evidence,args.output)
        return
    schematics(args.output)
    if not args.draft_only: finish(args.study,args.output,args.manuscript,not args.without_extensions)


if __name__=='__main__': main()
