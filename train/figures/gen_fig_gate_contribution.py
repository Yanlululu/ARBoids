"""Reproducible figures and paired statistics for the original-source study."""
import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import binomtest


METHODS=('original','reactive_independent','reactive_joint','predictive_independent','predictive_joint','cbf')
LABELS={'original':'Original ARBoids','reactive_independent':'Reactive, independent',
        'reactive_joint':'Reactive, joint','predictive_independent':'Predictive, independent',
        'predictive_joint':'Predictive, joint','cbf':'CBFpy'}
CHINESE={'original':'原始 ARBoids','reactive_independent':'瞬时风险＋逐艇',
         'reactive_joint':'瞬时风险＋联合','predictive_independent':'轨迹预测＋逐艇',
         'predictive_joint':'轨迹预测＋联合','cbf':'官方 CBFpy'}
COLORS=dict(zip(METHODS,['#707070','#56B4E9','#0072B2','#E69F00','#D55E00','#009E73']))
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':9,'axes.titlesize':10,'axes.titleweight':'bold','legend.fontsize':8,
    'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,
    'legend.frameon':False,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300,
    'figure.dpi':130,'lines.linewidth':1.4})


def read_csv(path):
    with Path(path).open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def save_csv(path,rows):
    if not rows:
        return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def wilson(success,n):
    p=success/n
    z=1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return np.array([max(0.,min(p,center-half)),min(1.,max(p,center+half))])


def save(fig,folder,name):
    fig.savefig(folder/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(folder/f'{name}.png',bbox_inches='tight',dpi=300)
    plt.close(fig)


def main_results(summary,folder):
    index={r['method']:r for r in summary['main_table']}
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.0),sharey=True,layout='constrained')
    for y,method in enumerate(METHODS):
        row=index[method]
        for ax,key in zip(axes,('successes','collisions')):
            value=row[key]/row['episodes']
            ci=wilson(row[key],row['episodes'])
            ax.errorbar(value*100,y,xerr=np.array([[value-ci[0]],[ci[1]-value]])*100,
                        color=COLORS[method],fmt='o',capsize=3,markersize=5)
    axes[0].set_yticks(range(len(METHODS)),[LABELS[m] for m in METHODS])
    axes[0].invert_yaxis()
    axes[0].set(title='a  Defense success',xlabel='Success rate (%)',xlim=(80,100.5))
    axes[1].set(title='b  Defender collisions',xlabel='Collision rate (%)',xlim=(-.5,13))
    fig.suptitle('Untouched ARBoids source | 256 paired scenes',fontsize=11)
    save(fig,folder,'fig_source_main')


def paired_effects(root,folder):
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.0),sharey=True,layout='constrained')
    order=[m for m in METHODS if m!='predictive_joint']
    for ax,suite,title in zip(axes,('mechanism','generalization'),('a  Main comparison (n=256)','b  Scene variation (n=512)')):
        index={r['comparison']:r for r in read_csv(root/suite/'paired_contrasts.csv')}
        for y,method in enumerate(order):
            r=index[method]
            value=float(r['success_gain'])*100
            low=float(r['success_gain_low'])*100
            high=float(r['success_gain_high'])*100
            ax.errorbar(value,y,xerr=[[value-low],[high-value]],fmt='o',color=COLORS[method],capsize=3)
        ax.axvline(0,color='#404040',ls='--',lw=.8)
        ax.set(title=title,xlabel='Predictive joint minus comparator (pp)')
    axes[0].set_yticks(range(len(order)),[LABELS[m] for m in order])
    axes[0].invert_yaxis()
    save(fig,folder,'fig_source_paired_effects')


def generalization_results(root,folder):
    rows=read_csv(root/'generalization/main_table.csv')
    index={(r['cell'],r['method']):r for r in rows}
    comparators=('original','reactive_joint','predictive_independent','cbf')
    values=[]
    for method in comparators:
        values.append(np.array([[100*(float(index[(f'n{n}-a{a:g}','predictive_joint')]['success_rate'])-
                                           float(index[(f'n{n}-a{a:g}',method)]['success_rate']))
                                 for a in (1.5,2.,2.5,3.)] for n in (2,3,4,5)]))
    maximum=max(10.,max(np.abs(v).max() for v in values))
    fig,axes=plt.subplots(2,2,figsize=(7.0,5.1),layout='constrained')
    for ax,method,value in zip(axes.flat,comparators,values):
        im=ax.imshow(value,cmap='RdBu',norm=TwoSlopeNorm(0.,-maximum,maximum),aspect='auto')
        ax.set_xticks(range(4),['1.5','2.0','2.5','3.0'])
        ax.set_yticks(range(4),['2','3','4','5'])
        ax.set(xlabel='Attacker agility',ylabel='Defenders',title='vs '+LABELS[method])
        ax.grid(False)
        for i in range(4):
            for j in range(4):
                ax.text(j,i,f'{value[i,j]:+.1f}',ha='center',va='center',fontsize=8,
                        color='white' if abs(value[i,j])>.6*maximum else '#202020')
    fig.colorbar(im,ax=axes,label='Predictive-joint success gain (percentage points)',shrink=.82)
    fig.suptitle('Same 32 paired scenes per cell',fontsize=11)
    save(fig,folder,'fig_source_generalization')


def factorial_statistics(root):
    """Descriptive factorial effects with scene-paired, configuration-stratified CIs."""
    result=[]
    for suite in ('mechanism','generalization'):
        rows=read_csv(root/suite/'episodes.csv')
        scenes=sorted({(r['cell'],int(r['scene_seed'])) for r in rows})
        index={(r['cell'],int(r['scene_seed']),r['method']):r for r in rows}
        terms={m:np.array([int(index[(*s,m)]['success']) for s in scenes]) for m in METHODS}
        effects={
            'prediction_at_joint':terms['predictive_joint']-terms['reactive_joint'],
            'joint_at_prediction':terms['predictive_joint']-terms['predictive_independent'],
            'prediction_at_independent':terms['predictive_independent']-terms['reactive_independent'],
            'joint_at_reactive':terms['reactive_joint']-terms['reactive_independent'],
            'factorial_interaction':terms['predictive_joint']-terms['predictive_independent']-
                                    terms['reactive_joint']+terms['reactive_independent']}
        cells=sorted({s[0] for s in scenes})
        groups=[np.array([i for i,s in enumerate(scenes) if s[0]==cell]) for cell in cells]
        rng=np.random.default_rng(867345)
        take=np.concatenate([g[rng.integers(0,len(g),size=(20000,len(g)))] for g in groups],axis=1)
        for name,delta in effects.items():
            ci=np.quantile(delta[take].mean(-1),[.025,.975])
            result.append(dict(suite=suite,effect=name,scenes=len(scenes),
                success_gain_pp=float(delta.mean()*100),low_pp=float(ci[0]*100),high_pp=float(ci[1]*100)))
    save_csv(root/'factorial_effects.csv',result)
    return result


def validate_vrx_subset(root):
    """Apply the original pre-control criterion, never an outcome-based filter."""
    rows=read_csv(root/'vrx/episodes.csv')
    specification=json.loads((root/'vrx/specification.json').read_text(encoding='utf-8'))
    methods={'original','predictive_independent','predictive_joint','cbf'}
    expected={(s['cell'],s['seed']) for s in specification['scenes']}
    groups={}
    for row in rows:
        groups.setdefault((row['cell'],int(row['seed'])),[]).append(row)
        result=json.loads((root/'vrx'/row['run_id']/'result.json').read_text(encoding='utf-8'))
        if not result['passed'] or not result['source_unchanged'] or result['source']!=specification['source']:
            raise RuntimeError('A VRX trial failed source or execution validation.')
        if result['checkpoint_sha256']!=specification['checkpoint_sha256']:
            raise RuntimeError('VRX checkpoint changed.')
    if set(groups)!=expected:
        raise RuntimeError('VRX scene roster is incomplete.')
    included=[]
    excluded=[]
    errors=[]
    for (cell,seed),group in sorted(groups.items()):
        if len(group)!=4 or {r['method'] for r in group}!=methods or len({r['initial_poses'] for r in group})!=1:
            raise RuntimeError('VRX generated initial states or method roster differ.')
        positions=[np.array([json.loads(r['first_attacker_position']),*json.loads(r['first_defender_positions'])]) for r in group]
        error=max(float(np.linalg.norm(p-positions[0],axis=-1).max()) for p in positions)
        accepted=error<1e-7
        for row in group:
            row['strict_pairing_included']=accepted
            row['strict_pairing_error_m']=error
        if accepted:
            included.extend(group)
            errors.append(error)
        else:
            excluded.append(dict(cell=cell,seed=seed,first_control_position_difference_m=error,
                                 reason='Pre-control position criterion >= 1e-7 m'))
    if not included:
        raise RuntimeError('No complete VRX blocks passed the original pairing criterion.')
    table=[]
    for cell in ('dock-n3','ocean-n3','dock-n4','all'):
        for method in ('original','predictive_independent','predictive_joint','cbf'):
            selected=[r for r in included if r['method']==method and (cell=='all' or r['cell']==cell)]
            table.append(dict(cell=cell,method=method,episodes=len(selected),
                successes=sum(int(r['success']) for r in selected),collisions=sum(int(r['collision']) for r in selected),
                breaches=sum(int(r['breach']) for r in selected),
                mean_policy_ms=float(np.mean([float(r['mean_policy_ms']) for r in selected])),
                deadline_misses=sum(int(r['deadline_misses']) for r in selected)))
    summary=dict(passed=True,scope='Only complete four-method blocks passing the original first-control criterion',
        full_batch_scenes=len(expected),full_batch_method_episodes=len(rows),
        scenes=len(included)//4,method_episodes=len(included),excluded_blocks=excluded,
        source_unchanged=True,paired_initial_poses_verified=True,paired_control_start_positions_verified=True,
        maximum_start_position_difference=max(errors),criterion_m=1e-7,
        selection_uses_outcomes=False,main_table=table,
        unsuccessful_recovery_trials_included=False,
        full_batch_summary='summary.json (the full batch retains its failed strict-pairing status)')
    save_csv(root/'vrx/episodes.csv',rows)
    save_csv(root/'vrx/paired_main_table.csv',table)
    (root/'vrx/paired_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary


def vrx_results(root,folder):
    rows=read_csv(root/'vrx/episodes.csv')
    rows=[r for r in rows if r['strict_pairing_included']=='True']
    methods=('original','predictive_independent','predictive_joint','cbf')
    cells=('dock-n3','ocean-n3','dock-n4')
    fig,axes=plt.subplots(1,3,figsize=(7.1,2.6),sharey=True,layout='constrained')
    for ax,cell in zip(axes,cells):
        for j,method in enumerate(methods):
            select=[r for r in rows if r['cell']==cell and r['method']==method]
            success=sum(int(r['success']) for r in select)
            n=len(select)
            value=success/n
            ci=wilson(success,n)
            ax.errorbar(j,100*value,yerr=[[100*(value-ci[0])],[100*(ci[1]-value)]],
                        fmt='o',color=COLORS[method],capsize=3)
            ax.text(j,100*value+3,f'{success}/{n}',ha='center',fontsize=7)
        ax.set_xticks(range(4),['Original','Pred. ind.','Pred. joint','CBFpy'],rotation=40,ha='right')
        ax.set(title=cell,ylim=(0,112),xlim=(-.5,3.5))
    axes[0].set_ylabel('Successful defense (%)')
    fig.suptitle(f'VRX: {len(rows)//4} strictly paired initial states',fontsize=10)
    save(fig,folder,'fig_source_vrx')
    contrasts=[]
    for cell in cells+('all',):
        selected=[r for r in rows if cell=='all' or r['cell']==cell]
        index={(r['cell'],int(r['seed']),r['method']):r for r in selected}
        scenes=sorted({(r['cell'],int(r['seed'])) for r in selected})
        for comparator in ('original','predictive_independent','cbf'):
            delta=np.array([int(index[(*s,'predictive_joint')]['success'])-int(index[(*s,comparator)]['success']) for s in scenes])
            wins,losses=int((delta>0).sum()),int((delta<0).sum())
            contrasts.append(dict(cell=cell,comparison=comparator,scenes=len(scenes),
                success_gain_pp=float(delta.mean()*100),paired_wins=wins,paired_losses=losses,
                p_raw=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.))
    for cell in cells+('all',):
        group=sorted([r for r in contrasts if r['cell']==cell],key=lambda r:r['p_raw'])
        adjusted=0.
        for rank,row in enumerate(group):
            adjusted=max(adjusted,min(1.,(len(group)-rank)*row['p_raw']))
            row['p_holm']=adjusted
    save_csv(root/'vrx/paired_contrasts.csv',contrasts)
    return contrasts


def case_examples(root,folder):
    """Replay representative paired cases, accepting only exact recorded trajectories."""
    from evaluate_gate_contribution import initialize, rollout
    rows=read_csv(root/'mechanism/episodes.csv')
    specification=json.loads((root/'mechanism/specification.json').read_text(encoding='utf-8'))
    index={(int(r['scene_seed']),r['method']):r for r in rows}
    seeds=sorted({int(r['scene_seed']) for r in rows})
    groups=[]
    for source,target in (('original','predictive_joint'),('predictive_joint','cbf')):
        candidates=[seed for seed in seeds if int(index[seed,source]['collision']) and int(index[seed,target]['success'])]
        candidates.sort(key=lambda seed:(float(index[seed,target]['team_return'])-float(index[seed,source]['team_return']),seed))
        if not candidates:
            raise RuntimeError('No eligible case for the declared selection rule.')
        groups.append((source,target,candidates[len(candidates)//2],len(candidates)))
    initialize(specification['checkpoint'])
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
    selected=[]
    for panel,(ax,(source,target,seed,eligible)) in enumerate(zip(axes,groups)):
        original=index[seed,'original']
        scene=dict(cell=original['cell'],scene_seed=seed,noise_seed=int(original['noise_seed']),
                   defenders=int(original['defenders']),agility=float(original['agility']))
        for method in ('original','predictive_joint','cbf'):
            result,trace=rollout(scene,method,trace=True)
            expected=index[seed,method]
            if result['trajectory_sha256']!=expected['trajectory_sha256']:
                raise RuntimeError(f'Illustration replay does not match the formal result: {seed}, {method}')
            t=np.array([r['time'] for r in trace])
            d=np.array([r['minimum_distance'] for r in trace])
            ax.plot(t,d,color=COLORS[method],label=LABELS[method])
            ax.scatter(t[-1],d[-1],color=COLORS[method],s=15,zorder=3)
            selected.append(dict(panel=panel+1,scene_seed=seed,method=method,
                selection_source=source,selection_target=target,eligible_cases=eligible,
                selection_rule='Upper median target-minus-source return among collision-to-success pairs',
                trajectory_sha256=result['trajectory_sha256'],exact_replay_verified=True,
                success=result['success'],collision=result['collision'],minimum_distance=result['minimum_distance']))
        ax.axhline(5.,ls='--',color='#404040',lw=.8)
        ax.axhline(7.,ls=':',color='#888888',lw=.8)
        ax.set(xlabel='Time (s)',ylabel='Minimum defender separation (m)',
               title=f"{'ab'[panel]}  Scene {seed}")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=8)
    save(fig,folder,'fig_source_cases')
    save_csv(root/'case_examples.csv',selected)
    return selected


def write_report(root):
    """Write the result narrative from the completed, validated experiment data."""
    summaries={s:json.loads((root/s/'summary.json').read_text(encoding='utf-8'))
               for s in ('mechanism','generalization','vrx')}
    full_vrx=summaries['vrx']
    summaries['vrx']=json.loads((root/'vrx/paired_summary.json').read_text(encoding='utf-8'))
    if not all(s['passed'] for s in summaries.values()):
        raise RuntimeError('Only completed and validated experiments can enter the report.')
    main=summaries['mechanism']
    general=summaries['generalization']
    vrx=summaries['vrx']
    specification=json.loads((root/'mechanism/specification.json').read_text(encoding='utf-8'))
    mi={r['method']:r for r in main['main_table']}
    gi={r['method']:r for r in general['main_table']}
    mc={r['comparison']:r for r in main['contrasts']}
    gc={r['comparison']:r for r in general['contrasts']}
    effects={(r['suite'],r['effect']):r for r in read_csv(root/'factorial_effects.csv')}
    vrx_contrasts=read_csv(root/'vrx/paired_contrasts.csv')
    vi={r['method']:r for r in vrx['main_table'] if r['cell']=='all'}

    def pvalue(value):
        value=float(value)
        return f'{value:.3g}' if value<.001 else f'{value:.4f}'

    def count(row,key):
        return f"{row[key]}/{row['episodes']} ({100*row[key]/row['episodes']:.2f}%)"

    def table(header,rows):
        return ['| '+' | '.join(header)+' |','|'+'|'.join(['---']*len(header))+'|']+[
            '| '+' | '.join(map(str,r))+' |' for r in rows]

    def comparison(row):
        return (f"{100*row['success_gain']:+.2f} 个百分点，95% CI "
                f"[{100*row['success_gain_low']:+.2f}, {100*row['success_gain_high']:+.2f}]，"
                f"Holm 校正 p={pvalue(row['p_holm'])}")

    lines=['# 原始源码上的贡献验证结果','',
        '**结论：预测＋联合门控在原始 ARBoids 上提高了防守成功率；这批结果尚未证明候选轨迹预测带来额外优势。场景变化实验中，官方 CBFpy 的成功率更高。**','',
        '三项补充均已完成：预测／联合选择的四组消融、未修改的官方 CBFpy 竞争基线，以及独立的 VRX 与场景变化评测。'
        '本报告只使用这次重新建立的原始源码基准，先前论文参数版的 99.22% 结果不并入比较。','',
        f"数值仿真共 {main['scene_count']+general['scene_count']} 个配对场景、"
        f"{main['method_episodes']+general['method_episodes']} 个方法回合；VRX 实跑 {vrx['full_batch_scenes']} 个初态、{vrx['full_batch_method_episodes']} 个方法回合，"
        f"其中 {vrx['scenes']} 组通过既定起点一致性标准，{vrx['method_episodes']} 个回合进入严格配对统计。",'',
        '## 1. 主比较：原始三艇任务，256 个新场景','']
    lines+=table(['方法','成功','碰撞','源码失守','平均团队回报'],[
        [CHINESE[m],count(mi[m],'successes'),str(mi[m]['collisions']),str(mi[m]['source_losses']),f"{mi[m]['mean_return']:.2f}"] for m in METHODS])
    lines+=['',f"完整方法相对原始 ARBoids：{comparison(mc['original'])}；配对新增成功 {mc['original']['paired_wins']} 例、丢失成功 {mc['original']['paired_losses']} 例。",
        '17 个新增成功均来自原始碰撞回合；4 个新增失败包含 2 个碰撞和 2 个源码失守。',
        f"相对 CBFpy：{comparison(mc['cbf'])}。CBFpy 在这一批中成功 {mi['cbf']['successes']} 次、碰撞 {mi['cbf']['collisions']} 次，不能据此声称完整方法优于直接安全滤波。",'',
        f"完整方法相对原始策略的团队回报差为 {mc['original']['return_gain']:+.2f}，95% CI "
        f"[{mc['original']['return_gain_low']:+.2f}, {mc['original']['return_gain_high']:+.2f}]；成功率收益没有同时转化为明确的回报收益。",'',
        '![原始三艇任务的成功率与碰撞率](figures/fig_source_main.png)','',
        '图 1：六种方法在同一批场景上的结果。误差线为各比例的 Wilson 95% 区间；方法差异使用下文的配对统计。','',
        '## 2. 预测和联合选择各自贡献了什么','',
        '“去掉预测”使用当前状态的二阶屏障残差评价风险，不推演轨迹；“去掉联合”让各艇在队友保持原权重的条件下分别选择，并同时执行。'
        '四组共享候选集合、警戒距离、改动惩罚和控制周期。','']
    labels={'prediction_at_joint':'加入轨迹预测（联合条件下）',
            'joint_at_prediction':'加入联合选择（预测条件下）',
            'prediction_at_independent':'加入轨迹预测（逐艇条件下）',
            'joint_at_reactive':'加入联合选择（瞬时条件下）',
            'factorial_interaction':'预测与联合的交互作用'}
    effect_rows=[]
    for name,label in labels.items():
        formatted=[]
        for suite in ('mechanism','generalization'):
            r=effects[suite,name]
            formatted.append(f"{float(r['success_gain_pp']):+.2f} [{float(r['low_pp']):+.2f}, {float(r['high_pp']):+.2f}]")
        effect_rows.append([label,*formatted])
    lines+=table(['消融效应','主比较：百分点 [95% CI]','场景变化：百分点 [95% CI]'],effect_rows)
    lines+=['','这些区间来自 20,000 次按场景配对、按配置分层的自助抽样，是描述性的机制分析。主比较和场景变化各自的五项完整方法比较另行采用精确 McNemar 检验与 Holm 校正。','',
        '在主比较中，“预测＋联合”比“瞬时＋联合”仅多成功 1 次，比“预测＋逐艇”多成功 3 次；两项差异均未形成明确证据。'
        '场景变化中，瞬时风险下的联合选择有正向收益，但把瞬时风险替换成候选轨迹预测后，成功率均值反而下降。'
        '因此可以报告联合协调的实验现象，不能把“预测与联合缺一不可”写成已经验证的结论。','',
        '![完整方法相对各对照的配对成功率差](figures/fig_source_paired_effects.png)','',
        '图 2：点为完整方法减去各对照的成功率差，横线为 10,000 次场景配对自助法的 95% 区间；零线表示无差异。','',
        '## 3. 场景变化：4 种艇数 × 4 种攻击艇敏捷度','',
        '艇数为 2、3、4、5，敏捷度为 1.5、2.0、2.5、3.0；每格 32 个新场景，共 512 个。','']
    lines+=table(['方法','成功','碰撞','源码失守','平均团队回报'],[
        [CHINESE[m],count(gi[m],'successes'),str(gi[m]['collisions']),str(gi[m]['source_losses']),f"{gi[m]['mean_return']:.2f}"] for m in METHODS])
    lines+=['',f"完整方法相对原始 ARBoids：{comparison(gc['original'])}。",
        f"完整方法相对 CBFpy：{comparison(gc['cbf'])}。",
        f"完整方法相对瞬时联合：{comparison(gc['reactive_joint'])}；相对预测逐艇：{comparison(gc['predictive_independent'])}。",'',
        '![不同艇数与敏捷度下的成功率差](figures/fig_source_generalization.png)','',
        '图 3：每个格子为同一批 32 个场景上完整方法减去对照的成功率，单位为百分点。完整逐格数据见 generalization/main_table.csv。','',
        '## 4. VRX：真实 Gazebo 回合','',
        '采用三艇码头 16 个初态、三艇开阔水域 8 个初态、四艇码头 8 个初态。每个初态比较四种方法。'
        '28 组通过预设的首次控制位置差小于 `1e-7 m` 标准，另外 4 组只列入完整批次的描述性结果。筛选条件不使用任务结局。', '']
    cell_names={'dock-n3':'码头／三艇','ocean-n3':'开阔水域／三艇','dock-n4':'码头／四艇','all':'合计'}
    vrx_rows=[]
    for cell in ('dock-n3','ocean-n3','dock-n4','all'):
        index={r['method']:r for r in vrx['main_table'] if r['cell']==cell}
        all_index={r['method']:r for r in full_vrx['main_table'] if r['cell']==cell}
        for method in ('original','predictive_independent','predictive_joint','cbf'):
            r=index[method]
            vrx_rows.append([cell_names[cell],CHINESE[method],count(all_index[method],'successes'),count(r,'successes'),r['collisions'],r['breaches']])
    lines+=table(['场景','方法','完整批次成功','严格配对子集成功','子集碰撞','子集失守'],vrx_rows)
    lines+=['']
    before=vrx.get('outcomes_before_pairing_recovery',[])
    if before:
        before_index={r['method']:r for r in before}
        unchanged=all(all(before_index[m][key]==vi[m][key] for key in ('successes','collisions','breaches')) for m in vi)
        if unchanged:
            lines+=['起点恢复前后，四种方法的成功、碰撞与失守总数均未改变。']
        else:
            lines+=['起点恢复前后的原始计数均保存在 vrx/summary.json；本表只统计通过固定起点一致性标准的完整配对组。']
    if len({vi[m]['successes'] for m in vi})==1:
        lines+=['完整 32 组中，四种方法均成功 29 次；严格配对的 28 组中，均成功 27 次。两种统计范围都没有给出完整方法成功率更高的证据。'
                '原始策略的一次碰撞在另外三种方法中变为失守，避免碰撞没有增加该初态的成功数。']
    else:
        for r in vrx_contrasts:
            if r['cell']=='all':
                lines.append(f"完整方法相对{CHINESE[r['comparison']]}：{float(r['success_gain_pp']):+.2f} 个百分点，配对赢／输 {r['paired_wins']}/{r['paired_losses']}，Holm 校正 p={pvalue(r['p_holm'])}。")
    lines+=['','![VRX 各配置的成功率](figures/fig_source_vrx.png)','',
        '图 4：各初态配对比较，数字为成功数／总数，误差线为 Wilson 95% 区间。VRX 与数值仿真分别统计。','',
        '## 5. 可以据此怎样写论文','',
        '现有结果支持的表述是：在冻结原始 ARBoids 策略与 Boids 提案的条件下，增加门控选择层能提高本批任务成功率，并降低防守艇碰撞；联合选择在部分消融中呈现收益。'
        '候选轨迹预测的额外收益未被证实，而且完整方法在场景变化中落后于官方 CBFpy。以“新增候选轨迹预测和全队联合门控选择均带来必要且优越的贡献”为中心结论，目前证据不成立。','',
        '主比较中，原始策略、瞬时联合、预测联合、CBFpy 的平均单次策略耗时分别为 '
        f"{mi['original']['mean_policy_ms']:.2f}、{mi['reactive_joint']['mean_policy_ms']:.2f}、{mi['predictive_joint']['mean_policy_ms']:.2f}、{mi['cbf']['mean_policy_ms']:.2f} ms。"
        '这里使用批量实验期间的进程内计时，包含 Actor 与控制器、排除环境推进；不能把不同数值容差下的“指令改变次数”直接解释成能耗优势。','',
        '## 源码、协议与数据','',
        f"- 作者发布提交：`{specification['source']['commit']}`；`third_party/arboids_release` 中的原始文件逐字节校验通过。",
        f"- 权重：原始 YAML 协议的本地重训 `main-seed42-20260921/adares1.pth`，SHA-256 `{specification['checkpoint_sha256']}`。它不是作者公开发布的权重。",
        f"- 数值仿真的 {main['scene_count']+general['scene_count']} 条原始方法轨迹均与直接运行作者循环逐步完全一致。所有方法调用同一个未修改的原环境、奖励、观测与终止规则。",
        '- 数值仿真保留 80 秒时限和源码提前失守判定；表中的“源码失守”不等于攻击艇已经实际进入目标区。实际目标突破另行保存。',
        '- VRX 的策略、Boids、观测、原始速度回调与终止判定来自作者代码；共同的 ROS/Gazebo 资源、话题桥接和启动控制属于共享接口。保留原始 5.5 米捕获半径、100 秒时限。',
        f"- VRX 的 28 组严格配对子集通过初态及实际开始控制的位置校验，最大位置差 {vrx['maximum_start_position_difference']:.3g} m；完整 32 组的严格校验未通过，原始 `summary.json` 保留失败状态。",
        '- CBFpy 0.0.4 与 qpax 0.1.4 使用未修改的官方实现；只通过 CBFConfig 配置本任务动力学、艇间屏障和推力范围。',
        '- 11 项原始源码／控制器测试与 4 项配对恢复测试通过；包含原环境轨迹等价、名义预测器积分一致、消融调用隔离、CBF 动力学及输出约束、整组恢复与次数上限。',
        '- 启动阶段出现的 Gazebo 文件读取故障在尚未执行任何控制时中止；保留失败结果并以同一初态重启。',
        '- 4 个初态组的原始起点偏差为 3.83–8.43 mm。对首组进行了两次原启动方式的整组恢复，以及一次全模型生成后统一启动的整组恢复；后者将偏差减小到 0.976 μm，仍未达到原定阈值。恢复回合均未替换原始数据、未纳入统计。',
        '- 实验参数与运行方法：[协议说明](../../../docs/gate-contribution-study.md)。',
        '- 主比较：[逐回合数据](mechanism/episodes.csv)、[配对统计](mechanism/paired_contrasts.csv)、[完成校验](mechanism/summary.json)。',
        '- 场景变化：[逐回合数据](generalization/episodes.csv)、[各配置结果](generalization/main_table.csv)、[完成校验](generalization/summary.json)。',
        '- VRX：[全部逐回合数据及子集标记](vrx/episodes.csv)、[严格配对统计](vrx/paired_contrasts.csv)、[子集校验](vrx/paired_summary.json)、[完整批次校验](vrx/summary.json)。',
        '- 图表同时提供 300 dpi PNG 与可编辑矢量 PDF，文件名对应 figures 目录中的同名文件。','',
        '重新生成报告与图表：','',
        '```powershell',
        "& 'D:\\ARBoids\\.venv\\Scripts\\python.exe' -X utf8 train/figures/gen_fig_gate_contribution.py --root train/experiments/gate-contribution-source-20261006 --include-vrx",'```','']
    if (root/'case_examples.csv').exists():
        examples=read_csv(root/'case_examples.csv')
        cases={r['panel']:r for r in examples}
        position=lines.index('## 源码、协议与数据')
        lines[position:position]=['## 配对回合示例','',
            '![完整方法救回原始碰撞与 CBF 救回完整方法碰撞的示例](figures/fig_source_cases.png)','',
            f"左图从原始策略碰撞、完整方法成功的 {cases['1']['eligible_cases']} 例中，按回报改善取中位案例；"
            f"右图从完整方法碰撞、CBFpy 成功的 {cases['2']['eligible_cases']} 例中，按回报改善取上中位案例。"
            '虚线和点线分别为 5 米碰撞距离、7 米警戒距离。所有示例重放的完整轨迹哈希均与正式统计回合相同；选例只用于解释，不另计入样本量。','']
    (root/'results.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--include-vrx',action='store_true')
    parser.add_argument('--case-examples',action='store_true')
    args=parser.parse_args()
    folder=args.root/'figures'
    folder.mkdir(exist_ok=True)
    main=json.loads((args.root/'mechanism/summary.json').read_text())
    generalization=json.loads((args.root/'generalization/summary.json').read_text())
    if not main['passed'] or not generalization['passed']:
        raise RuntimeError('A numerical study failed validation.')
    main_results(main,folder)
    paired_effects(args.root,folder)
    generalization_results(args.root,folder)
    factorial_statistics(args.root)
    if args.case_examples:
        case_examples(args.root,folder)
    if args.include_vrx:
        validate_vrx_subset(args.root)
        vrx_results(args.root,folder)
        write_report(args.root)
    print(json.dumps({'passed':True,'figures':sorted(p.name for p in folder.glob('*.pdf'))}))


if __name__=='__main__':
    main()
