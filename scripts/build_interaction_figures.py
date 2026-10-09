"""Export the seven paper figure groups from verified, fixed-endpoint artifacts."""
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


def finish(study,directory,manuscript,include_extensions=True):
    analysis=json.loads((study/'analysis.json').read_text())
    if not analysis['complete']: raise ValueError('The 90000-row fixed numerical matrix is incomplete')
    formal=analysis.get('formal_evidence')
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
    paired=[r for r in analysis['paired_comparisons'] if r['reference']=='arboids_cbf' and r['cell']=='n3-a2.25' and r['metric']=='capture_time']
    if formal:
        contrast=formal['A']['effects']['capture_time']
        contrasts=[('A: task','s',contrast),('B: conditional value','return',formal['B']['E_env']['all']),
            ('C: six-defender task','s',formal['C']['strong']['capture_time']),
            ('D: configuration matching','return',formal['D']['configuration_matching']),
            ('D: signed interaction','return',formal['D']['conditional_interaction']),
            ('D: noise-corrected interaction','return squared',formal['D']['noise_corrected_interaction_second_moment'])]
        blocks.append(r'\begin{table*}[t]\centering\small\caption{Prespecified A--D paired effects across all five training seeds. Task and value-error contrasts are Full minus reference; negative values favor Full. Matching contrasts are original minus permuted; interaction signs have a different meaning.}\begin{tabular}{@{}llrr@{}}\toprule Evidence & Unit & Effect & 95\% CI\\\midrule'+'\n'+
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
    else:
        if len(paired)!=1: raise ValueError('Missing paired primary capped-time contrast against ARBoids+CBF')
        contrast=paired[0]
    headline=(r'\subsection{Fixed-endpoint evidence}'+'\n'+
        f'At three defenders and agility 2.25, the Full minus ARBoids+CBF capped-capture-time difference is {contrast["difference"]:.2f} s '
        f'(hierarchical paired 95\\% CI [{contrast["ci95"][0]:.2f}, {contrast["ci95"][1]:.2f}] s). '
        'Tables report all task outcomes and capped capture time; the exported analysis retains condition-specific paired intervals for every metric. '
        'The conditional-value error and matched auxiliary control are assessed alongside these task outcomes.\n')
    replace_results(manuscript,headline+'\n\n'.join(blocks))
    text=manuscript.read_text(encoding='utf-8')
    evidence=(f'At the fixed endpoint with three defenders and intruder agility 2.25, the complete-method minus common-safety baseline difference in capped capture time is '
        f'{contrast["difference"]:.2f} s '
        f'(paired 95\\% confidence interval [{contrast["ci95"][0]:.2f}, {contrast["ci95"][1]:.2f}] s).')
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
        f'三艇、敏捷度 2.25：完整方法相对共同安全基线 ARBoids+CBF 的平均封顶捕获耗时差为 {contrast["difference"]:.2f} 秒，'
        f'分层配对 95% 区间 [{contrast["ci95"][0]:.2f}, {contrast["ci95"][1]:.2f}] 秒。\n\n'+
        '固定训练终点、五个独立正式种子、90,000 个数值测试回合和 1,200 个 VRX 回合已汇总。'
        '旧 80 秒实验保留在下文，独立归属原方法及原协议。\n')
    if start in text:
        prefix,rest=text.split(start);_,suffix=rest.split(end)
        text=prefix+start+'\n'+content+end+suffix
    else: text=start+'\n'+content+end+'\n\n'+text
    path.write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'train/experiments/ia-crrl-20261007')
    parser.add_argument('--output',type=Path,default=ROOT/'paper/figures/interaction')
    parser.add_argument('--manuscript',type=Path,default=ROOT/'paper/interaction_aware_residual_rl.tex')
    parser.add_argument('--draft-only',action='store_true')
    parser.add_argument('--without-extensions',action='store_true')
    args=parser.parse_args()
    schematics(args.output)
    if not args.draft_only: finish(args.study,args.output,args.manuscript,not args.without_extensions)


if __name__=='__main__': main()
