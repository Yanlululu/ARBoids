"""Scene-clustered descriptive analysis of the fixed mechanism diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from source_arboids import SourcePolicy, snapshot, actor_thrust, sha256, verify_source
from gate_mechanism_diagnostics import restore, project_to_segments, RECOVERY_METHODS, ranking_metrics
from evaluate_gate_contribution import controller, write_csv
from gen_fig_gate_contribution import read_csv, save


LABELS={'original':'Original ARBoids','reactive_joint':'Reactive joint',
        'predictive_joint':'Predictive joint','fine_joint':'Finer predictive grid',
        'cbf':'CBFpy','projected_cbf':'CBF projected to gate','actual_shortlist':'Actual short-horizon choice'}
CHINESE={'original':'原始 ARBoids','reactive_joint':'瞬时联合','predictive_joint':'当前预测联合',
         'fine_joint':'细网格预测联合','cbf':'CBF','projected_cbf':'CBF 投影门控',
         'actual_shortlist':'实际短时结果选择'}
COLORS=dict(zip(RECOVERY_METHODS,['#777777','#0072B2','#D55E00','#E69F00','#009E73','#CC79A7','#332288']))
EXECUTION={'held':'Held thrust','replan_complete':'Replanned, full horizon',
           'replan_common_prefix':'Replanned, common prefix',
           'held_matched_replan_sample':'Held, same candidate sample'}
EPS=1e-12


def f(row,key):
    return float(row[key]) if row.get(key) not in (None,'') else None


def scene_key(row):
    return row['suite'],int(row['scene_seed'])


def cluster_mean(rows,key,seed=20867):
    grouped=defaultdict(list)
    for row in rows:
        value=f(row,key)
        if value is not None and np.isfinite(value):
            grouped[scene_key(row)].append(value)
    values=np.array([np.mean(x) for x in grouped.values()])
    if len(values)==0:
        return None,None,None,0
    rng=np.random.default_rng(seed)
    boot=values[rng.integers(0,len(values),size=(5000,len(values)))].mean(axis=-1)
    lo,hi=np.quantile(boot,[.025,.975])
    return float(values.mean()),float(lo),float(hi),len(values)


def prediction_summaries(rows):
    groups=defaultdict(list)
    for row in rows:
        categories=[('all','all'),('category',row['category']),('defenders',row['defenders']),
                    ('density',row['density']),('defenders_density',row['defenders']+'_'+row['density'])]
        if row['category']!='failure':
            categories.append(('category','natural'))
        for kind,value in categories:
            groups[(kind,value,row['execution'],int(row['horizon_steps']))].append(row)
    result=[]
    for (kind,value,execution,horizon),group in sorted(groups.items()):
        row=dict(group_type=kind,group=value,execution=execution,horizon_steps=horizon,
                 horizon=horizon*.2,states=len(group),scenes=len({scene_key(r) for r in group}))
        for key in ('dmin_mae','dmin_bias','risk_rank_spearman','distance_rank_spearman',
                    'miss_rate_5','miss_rate_7','false_safe_rate_5','false_safe_rate_7',
                    'position_rmse','heading_mae','complete_fraction','effective_steps'):
            mean,lo,hi,n=cluster_mean(group,key)
            row.update({key:mean,key+'_low':lo,key+'_high':hi,key+'_scenes':n})
        for key in ('actual_danger_5','actual_danger_7','predicted_safe_5','predicted_safe_7','missed_5','missed_7'):
            row[key+'_pooled_candidates']=sum(int(r[key]) for r in group)
        result.append(row)
    return result


def matched_candidate_metrics(root):
    """Compare execution assumptions on identical candidate IDs and censoring."""
    states={r['case_id']:r for r in read_csv(root/'states.csv')}
    groups=defaultdict(list)
    for row in read_csv(root/'candidate_outcomes.csv'):
        groups[(row['case_id'],int(row['horizon_steps']))].append(row)
    rows,coverage,hazards=[],[],[]
    for (case_id,horizon),group in groups.items():
        complete=[r for r in group if int(r['complete'])]
        meta=states[case_id]
        coverage.append(dict(**meta,horizon_steps=horizon,sampled_candidates=len(group),
                             complete_candidates=len(complete),complete_fraction=len(complete)/len(group),
                             early_collisions=sum(int(r['outcome'])==2 and not int(r['complete']) for r in group),
                             early_source_losses=sum(int(r['outcome'])==1 and not int(r['complete']) for r in group),
                             early_successes=sum(int(r['outcome'])>2 and not int(r['complete']) for r in group)))
        if complete:
            rows.append(dict(**meta,execution='held_matched_replan_sample',horizon_steps=horizon,horizon=horizon*.2,
                             **ranking_metrics([float(r['predicted_dmin']) for r in complete],
                                               [float(r['held_dmin']) for r in complete])))
        for threshold in (5,7):
            observed_danger=[r for r in group if float(r['replan_dmin'])<threshold]
            missed=sum(float(r['predicted_dmin'])>=threshold for r in observed_danger)
            unknown=[r for r in group if not int(r['complete']) and float(r['replan_dmin'])>=threshold]
            hazards.append(dict(**meta,horizon_steps=horizon,threshold_m=threshold,
                sampled_candidates=len(group),observed_danger=len(observed_danger),missed=missed,
                miss_rate=missed/len(observed_danger) if observed_danger else None,
                unobserved_full_horizon=len(unknown),
                predicted_safe_unknown=sum(float(r['predicted_dmin'])>=threshold for r in unknown)))
    return rows,coverage,hazards


def recovery_summary(rows):
    groups=defaultdict(list)
    for row in rows:
        for kind,value in [('all','all'),('defenders',row['defenders']),
                           ('source_outcome',row['source_outcome']),('lead',str(round(float(row['actual_lead']))))]:
            groups[(row['future'],row['method'],kind,value)].append(row)
    result=[]
    for (future,method,kind,value),group in sorted(groups.items()):
        row=dict(future=future,method=method,group_type=kind,group=value,branches=len(group),
                 scenes=len({scene_key(r) for r in group}),states=len({r['case_id'] for r in group}),
                 successes=sum(int(r['success']) for r in group),collisions=sum(int(r['collision']) for r in group),
                 source_losses=sum(int(r['source_loss']) for r in group))
        for key in ('success','collision','source_loss','duration','team_return','force_square_time_mean',
                    'force_square_integral','thrust_variation_per_second','deviation_integral','projection_rms'):
            mean,lo,hi,n=cluster_mean(group,key)
            row.update({key+'_mean':mean,key+'_low':lo,key+'_high':hi})
        result.append(row)
    return result


def recovery_pairs(rows):
    index=defaultdict(dict)
    for row in rows:
        index[(row['case_id'],row['repeat'])][row['method']]=row
    checks=[]
    for methods in index.values():
        ref=methods['predictive_joint']
        gates=[methods[m] for m in RECOVERY_METHODS if m!='cbf']
        record={k:ref[k] for k in ('suite','scene_seed','case_id','defenders','actual_lead','source_outcome','future','repeat')}
        record.update(cbf_success=int(methods['cbf']['success']),
            predictive_success=int(ref['success']),
            any_tested_gate_success=int(any(int(g['success']) for g in gates)),
            fine_rescue=int(int(methods['fine_joint']['success'])>int(ref['success'])),
            actual_choice_rescue=int(int(methods['actual_shortlist']['success'])>int(ref['success'])),
            projected_success=int(methods['projected_cbf']['success']),
            cbf_only_success=int(int(methods['cbf']['success']) and not any(int(g['success']) for g in gates)),
            cbf_success_projection_failure=int(int(methods['cbf']['success']) and not int(methods['projected_cbf']['success'])))
        checks.append(record)
    contrasts=[]
    for future in ('recorded','unseen'):
        for other in ('reactive_joint','fine_joint','projected_cbf','actual_shortlist','cbf'):
            deltas=[]
            for methods in index.values():
                ref,alt=methods['predictive_joint'],methods[other]
                if ref['future']!=future:
                    continue
                deltas.append(dict(suite=ref['suite'],scene_seed=ref['scene_seed'],
                                   delta=int(alt['success'])-int(ref['success'])))
            mean,lo,hi,n=cluster_mean(deltas,'delta')
            contrasts.append(dict(future=future,method=other,comparison='predictive_joint',
                                  scene_clusters=n,branches=len(deltas),success_gain=mean,low=lo,high=hi,
                                  paired_wins=sum(r['delta']>0 for r in deltas),paired_losses=sum(r['delta']<0 for r in deltas)))
    return checks,contrasts


def action_geometry(root,spec):
    torch.set_num_threads(1)
    policy=SourcePolicy(spec['checkpoint'])
    with (root/'states.pkl').open('rb') as file:
        cases=pickle.load(file)
    result=[]
    for case in cases:
        if case['meta']['category']!='failure':
            continue
        env,obs=restore(case['state'])
        action=policy(obs)
        _,physical,_=controller('cbf',env.defender_num).control(snapshot(env),action,env.boids_actions)
        theta,projected,residual=project_to_segments(physical,action,env.boids_actions)
        for i in range(env.defender_num):
            result.append(dict(case_id=case['meta']['case_id'],scene_seed=case['meta']['scene_seed'],
                vessel=i,defenders=env.defender_num,source_outcome=case['meta']['source_outcome'],
                residual_N=float(residual[i]),theta=float(theta[i]),
                boids_left=float(env.boids_actions[i,0]),boids_right=float(env.boids_actions[i,1]),
                learned_left=float(actor_thrust(action)[i,0]),learned_right=float(actor_thrust(action)[i,1]),
                cbf_left=float(physical[i,0]),cbf_right=float(physical[i,1]),
                projected_left=float(projected[i,0]),projected_right=float(projected[i,1])))
    return result


def prediction_figure(rows,folder):
    fig,axes=plt.subplots(2,2,figsize=(7.1,5.3),layout='constrained')
    for col,category in enumerate(('natural','failure')):
        for execution,color,marker in [('held_matched_replan_sample','#0072B2','o'),('replan_complete','#D55E00','s')]:
            selected=sorted([r for r in rows if r['group_type']=='category' and r['group']==category
                             and r['execution']==execution],key=lambda r:r['horizon'])
            for ax,key in ((axes[0,col],'dmin_mae'),(axes[1,col],'distance_rank_spearman')):
                valid=[r for r in selected if r[key] is not None]
                x=[r['horizon'] for r in valid]
                y=[r[key] for r in valid]
                ax.plot(x,y,marker=marker,color=color,label=EXECUTION[execution])
                ax.fill_between(x,[r[key+'_low'] for r in valid],[r[key+'_high'] for r in valid],color=color,alpha=.13)
                ax.set_xticks([.2,.6,1.,2.])
                ax.set(xlabel='Prediction horizon (s)')
        axes[0,col].set_title(('a  Natural-state sample','b  Pre-failure sample')[col])
        axes[0,col].set_ylabel('Minimum-distance MAE (m)')
        axes[1,col].set_ylabel('Candidate distance rank (Spearman)')
        axes[1,col].set_ylim(-1.05,1.05)
        axes[1,col].axhline(0,color='#444444',ls='--',lw=.6)
    axes[0,0].legend(loc='upper left',fontsize=7)
    fig.suptitle('Held-action prediction vs. matched execution and replanning',fontsize=11)
    save(fig,folder,'fig_prediction_validity')


def recovery_figure(rows,folder):
    fig,axes=plt.subplots(1,2,figsize=(7.3,3.7),sharey=True,layout='constrained')
    for ax,future,title in zip(axes,('recorded','unseen'),('a  Recorded future (60 states)','b  Three unseen futures (180 branches)')):
        index={r['method']:r for r in rows if r['group_type']=='all' and r['future']==future}
        for i,method in enumerate(RECOVERY_METHODS):
            r=index[method]
            value=r['success_mean']*100
            ax.errorbar(value,i,xerr=[[value-r['success_low']*100],[r['success_high']*100-value]],
                        fmt='o',capsize=3,color=COLORS[method])
            ax.text(max(2.,value+3),i+.13,f"{r['successes']}/{r['branches']}",fontsize=7,color=COLORS[method])
        ax.set(xlabel='Successful continuations (%)',title=title,xlim=(-3,104))
    axes[0].set_yticks(range(len(RECOVERY_METHODS)),[LABELS[m] for m in RECOVERY_METHODS])
    axes[0].invert_yaxis()
    fig.suptitle('Same-state recovery | 20 selected failure scenes',fontsize=11)
    save(fig,folder,'fig_same_state_recovery')


def projection_figure(rows,checks,folder):
    residuals=defaultdict(list)
    for row in rows:
        residuals[row['case_id']].append(row['residual_N'])
    fig,axes=plt.subplots(1,2,figsize=(7.0,3.1),layout='constrained')
    values=np.sort([max(x) for x in residuals.values()])
    axes[0].step(values,np.arange(1,len(values)+1)/len(values)*100,where='post',color='#009E73')
    axes[0].set(xlabel='Largest vessel projection residual (N)',ylabel='Cumulative states (%)',
                title='a  Initial CBF action outside the gate segment')
    recorded=[r for r in checks if r['future']=='recorded']
    for category,color,label in [('both','#0072B2','CBF + tested gate succeed'),
                                 ('cbf_only','#009E73','Only CBF succeeds'),('other','#777777','CBF does not succeed')]:
        selected=[r for r in recorded if ('both' if r['cbf_success'] and r['any_tested_gate_success']
                     else 'cbf_only' if r['cbf_only_success'] else 'other')==category]
        if selected:
            axes[1].scatter([max(residuals[r['case_id']]) for r in selected],
                            [float(r['actual_lead']) for r in selected],s=24,alpha=.75,color=color,label=label)
    axes[1].set(xlabel='Largest vessel projection residual (N)',ylabel='Time before recorded failure (s)',
                title='b  Geometry with recovery outcomes',yticks=[1,2,3])
    axes[1].legend(fontsize=6.7,loc='best')
    save(fig,folder,'fig_action_space')


def transition_figure(rows,folder):
    order=[3,2,1]
    names=['Success','Collision','Source loss']
    fig,axes=plt.subplots(2,2,figsize=(6.4,5.8),layout='constrained')
    for i,suite in enumerate(('mechanism','generalization')):
        for j,other in enumerate(('reactive_joint','cbf')):
            # Existing transfer file has reactive->predictive; reverse for this display.
            before,after=('reactive_joint','predictive_joint') if other=='reactive_joint' else ('predictive_joint','cbf')
            matrix=np.zeros((3,3),dtype=int)
            for row in rows:
                if row['suite']!=suite or row['before']!=before or row['after']!=after:
                    continue
                a,b=int(row['before_outcome']),int(row['after_outcome'])
                a=3 if a>2 else a
                b=3 if b>2 else b
                if other=='reactive_joint':
                    a,b=b,a
                matrix[order.index(a),order.index(b)]+=int(row['scenes'])
            ax=axes[i,j]
            ax.imshow(matrix,cmap='Blues')
            for a in range(3):
                for b in range(3):
                    ax.text(b,a,str(matrix[a,b]),ha='center',va='center',color='white' if matrix[a,b]>.55*matrix.max() else '#202020')
            ax.set_xticks(range(3),names,fontsize=7)
            ax.set_yticks(range(3),names,fontsize=7)
            ax.set(xlabel=LABELS[other],ylabel='Predictive joint',title=('Main: ' if i==0 else 'Variation: ')+LABELS[other])
            ax.grid(False)
    save(fig,folder,'fig_outcome_transitions')


def pct(value):
    return '未定义' if value is None else f'{value*100:.1f}%'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('train/experiments/gate-mechanism-diagnostics-20261006'))
    args=parser.parse_args()
    root=args.root
    summary=json.loads((root/'summary.json').read_text(encoding='utf-8'))
    spec=json.loads((root/'specification.json').read_text(encoding='utf-8'))
    if not summary['passed']:
        raise RuntimeError('Mechanism run did not pass.')
    for name,expected in spec['code'].items():
        if sha256(Path(__file__).resolve().parents[1]/name)!=expected:
            raise RuntimeError('Diagnostic implementation changed: '+name)
    if verify_source()!=spec['source'] or sha256(spec['checkpoint'])!=spec['checkpoint_sha256']:
        raise RuntimeError('Source or checkpoint changed.')
    matched,coverage,hazards=matched_candidate_metrics(root)
    prediction=prediction_summaries(read_csv(root/'prediction_metrics.csv')+matched)
    recovery_rows=read_csv(root/'recovery_episodes.csv')
    recovered=recovery_summary(recovery_rows)
    checks,contrasts=recovery_pairs(recovery_rows)
    geometry=action_geometry(root,spec)
    write_csv(root/'prediction_summary.csv',prediction)
    write_csv(root/'matched_candidate_metrics.csv',matched)
    write_csv(root/'horizon_coverage.csv',coverage)
    write_csv(root/'observed_hazard_detection.csv',hazards)
    write_csv(root/'recovery_summary.csv',recovered)
    write_csv(root/'recovery_mechanisms.csv',checks)
    write_csv(root/'recovery_paired_effects.csv',contrasts)
    write_csv(root/'initial_action_geometry.csv',geometry)
    folder=root/'figures'
    folder.mkdir(exist_ok=True)
    prediction_figure(prediction,folder)
    recovery_figure(recovered,folder)
    projection_figure(geometry,checks,folder)
    transition_figure(read_csv(root/'outcome_transitions.csv'),folder)
    print(json.dumps(dict(passed=True,figures=4,prediction_summary_rows=len(prediction),
                         recovery_summary_rows=len(recovered),recovery_branches=len(recovery_rows),
                         geometry_rows=len(geometry)),ensure_ascii=False))


if __name__=='__main__':
    main()
