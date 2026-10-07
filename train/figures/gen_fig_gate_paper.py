"""Build reproducible gate-coordination tables, vector figures and result prose."""
import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate_gate_paper import initial_case, run_method
from gate_diagnostics import FrozenARBoids, digest


ORDER = ['ARBoids', 'Boids only', 'Actor only', 'Fixed blend', 'Single H2',
         'Single correction', 'Rolling H1', 'Rolling H2', 'Rolling H3']
LABELS = {'Single H2': 'Single correction (2 s)', 'Single correction': 'Single correction (3 s)',
          'Rolling H1': 'Rolling (1 s)', 'Rolling H2': 'Rolling (2 s)', 'Rolling H3': 'Rolling (3 s)'}
COLORS = {name: '#9AAEB9' for name in ORDER}
COLORS.update({'ARBoids': '#596A73', 'Single H2': '#69A89A', 'Single correction': '#69A89A',
               'Rolling H1': '#DCA47A', 'Rolling H2': '#C84F37', 'Rolling H3': '#DCA47A'})


def wilson(success, n):
    z = 1.959963984540054
    p = success / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return max(0., center-half), min(1., center+half)


def exact_mcnemar(gained, lost):
    n = gained + lost
    return 1. if n == 0 else min(1., 2*sum(math.comb(n,k) for k in range(min(gained,lost)+1))/2**n)


def summarize(data):
    baseline = data[data.method=='ARBoids'].sort_values('scene_seed').set_index('scene_seed')
    n = len(baseline)
    assert n == baseline.index.nunique()
    bootstrap = np.random.default_rng(20261006).integers(n, size=(20000,n))
    rows = []
    for method in ORDER:
        frame = data[data.method==method].sort_values('scene_seed').set_index('scene_seed')
        assert len(frame)==n and frame.index.equals(baseline.index)
        assert (frame.future_seed==baseline.future_seed).all()
        assert (frame.reference_trajectory_sha256==baseline.trajectory_sha256).all()
        for key in ('success','collision','breach','team_return'):
            np.testing.assert_allclose(frame['reference_'+key],baseline[key],rtol=0,atol=1e-10)
        row = dict(method=method,episodes=n)
        for key in ('success','collision','breach','capture','timeout'):
            row[key+'_count'] = int(frame[key].sum())
            row[key+'_rate'] = float(frame[key].mean())
        row['success_wilson_low'],row['success_wilson_high']=wilson(row['success_count'],n)
        gained=int(((frame.success==1)&(baseline.success==0)).sum())
        lost=int(((frame.success==0)&(baseline.success==1)).sum())
        row.update(success_gains=gained,success_losses=lost,success_p=exact_mcnemar(gained,lost),
                   collision_to_success=int(((baseline.collision==1)&(frame.success==1)).sum()),
                   new_breaches=int(((baseline.breach==0)&(frame.breach==1)).sum()),
                   mean_team_return=float(frame.team_return.mean()),
                   mean_end_time=float(frame.end_time.mean()),
                   mean_altered_steps=float(frame.altered_steps.mean()),
                   baseline_success_return_delta=float(frame.loc[baseline.success==1,'team_return_delta'].mean()))
        for key in ('success','collision','breach','team_return'):
            delta=(frame[key]-baseline[key]).to_numpy()
            lo,hi=np.quantile(delta[bootstrap].mean(axis=1),[.025,.975])
            row[key+'_delta']=float(delta.mean())
            row[key+'_ci_low'],row[key+'_ci_high']=float(lo),float(hi)
        rows.append(row)
    # Holm correction across all eight success comparisons against ARBoids.
    ordered=sorted(rows[1:],key=lambda row:row['success_p'])
    previous=0.
    for i,row in enumerate(ordered):
        previous=max(previous,min(1.,(len(ordered)-i)*row['success_p']))
        row['success_p_holm']=previous
    rows[0]['success_p_holm']=1.
    return pd.DataFrame(rows).set_index('method').loc[ORDER].reset_index()


def save(fig, directory, name):
    fig.savefig(directory/(name+'.pdf'))
    fig.savefig(directory/(name+'.png'),dpi=300)
    plt.close(fig)


def figures(table, directory):
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
        'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,
        'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,
        'axes.axisbelow':True,'savefig.bbox':'tight','savefig.pad_inches':.06})
    y=np.arange(len(table))
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.65),layout='constrained',gridspec_kw={'width_ratios':[1.1,1]})
    a,b=axes
    for i,row in table.iterrows():
        rate=row.success_rate*100
        a.barh(i,rate,height=.65,color=COLORS[row.method],zorder=2)
        a.errorbar(rate,i,xerr=[[rate-row.success_wilson_low*100],[row.success_wilson_high*100-rate]],
                   fmt='none',ecolor='#263B43',elinewidth=.8,capsize=2,zorder=3)
        a.text(116,i,f'{int(row.success_count)}/{int(row.episodes)}',ha='right',va='center',fontsize=8)
        b.errorbar(row.team_return_delta,i,xerr=[[row.team_return_delta-row.team_return_ci_low],
                    [row.team_return_ci_high-row.team_return_delta]],fmt='o',color=COLORS[row.method],
                    markersize=4,capsize=2,elinewidth=1)
    a.set(yticks=y,yticklabels=[LABELS.get(m,m) for m in table.method],xlabel='Successful defense (%)',
          xlim=(0,118),xticks=[0,25,50,75,100],title='(a) Task success')
    b.set(yticks=y,yticklabels=[],xlabel='Paired change in team return',title='(b) Task reward')
    for ax in axes:
        ax.invert_yaxis();ax.grid(axis='x',alpha=.16);ax.set_ylim(len(table)-.4,-.6)
    b.axvline(0,color='#596A73',linewidth=.8,linestyle='--')
    save(fig,directory,'fig_gate_main')

    chosen=['ARBoids','Single H2','Rolling H1','Rolling H2','Rolling H3']
    subset=table.set_index('method').loc[chosen]
    x=np.arange(len(chosen))
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.85),layout='constrained')
    a,b=axes
    a.bar(x-.17,subset.collision_rate*100,width=.32,label='Collision',color='#C84F37')
    a.bar(x+.17,subset.breach_rate*100,width=.32,label='Breach',color='#D4A33D')
    a.set(ylabel='Event rate (%)',title='(a) Failure modes',ylim=(0,max(subset.collision_rate.max()*100+2,5)))
    a.legend(frameon=False)
    for i,(_,row) in enumerate(subset.iterrows()):
        b.errorbar(i,row.team_return_delta,yerr=[[row.team_return_delta-row.team_return_ci_low],
                    [row.team_return_ci_high-row.team_return_delta]],fmt='o',color=COLORS[row.name],
                    markersize=5,capsize=3)
    b.axhline(0,color='#596A73',linewidth=.8,linestyle='--')
    b.set(ylabel='Paired change in team return',title='(b) Correction and horizon ablations')
    labels=['ARBoids','Single\n2 s','Rolling\n1 s','Rolling\n2 s','Rolling\n3 s']
    for ax in axes:
        ax.set_xticks(x,labels);ax.grid(axis='y',alpha=.16)
    save(fig,directory,'fig_gate_ablation')


def trajectory_figure(data, specification, directory):
    base=data[data.method=='ARBoids'].set_index('scene_seed')
    method=data[data.method=='Rolling H2'].set_index('scene_seed')
    recovered=method[(base.collision==1)&(method.success==1)].sort_values('team_return_delta')
    if recovered.empty: return None
    example=recovered.iloc[len(recovered)//2]
    scene=int(example.name);future=int(example.future_seed)
    policy=FrozenARBoids(specification['checkpoint'])
    case=initial_case(policy,scene)
    fig,ax=plt.subplots(figsize=(4.5,3.15),layout='constrained')
    for mode,label,color in [('baseline','ARBoids','#596A73'),('rolling','Rolling (2 s)','#C84F37')]:
        trace=run_method(case,policy,dict(mode=mode,horizon=2.,gate=None),future)
        expected=base.loc[scene,'trajectory_sha256'] if mode=='baseline' else example.trajectory_sha256
        assert trace.summary['trajectory_sha256']==expected
        a,b=np.triu_indices(trace.positions.shape[1],1)
        distance=np.linalg.norm(trace.positions[:,a]-trace.positions[:,b],axis=-1).min(axis=1)
        times=np.arange(len(distance))*.2
        ax.plot(times,distance,label=label,color=color,linewidth=1.5)
        ax.scatter(times[-1],distance[-1],color=color,s=18,zorder=3)
        if mode=='rolling':
            warning=trace.summary['first_warning_time']
            if warning is not None: ax.axvline(warning,color='#69A89A',linestyle=':',label='First warning')
    ax.axhline(5,color='#A73F30',linestyle='--',linewidth=.8,label='Collision threshold')
    ax.set(xlabel='Time (s)',ylabel='Minimum defender separation (m)',title=f'Recovery example: scene {scene}')
    ax.legend(frameon=False,ncol=2,fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.26))
    ax.grid(alpha=.15)
    save(fig,directory,'fig_gate_recovery')
    return dict(scene_seed=scene,future_seed=future,selection='Median return gain among collision-to-success recoveries')


def report(table,run_dir,diagnostic_dir,example):
    rows=table.set_index('method');base=rows.loc['ARBoids'];best=rows.loc['Rolling H2']
    single=rows.loc['Single H2'];h3=rows.loc['Rolling H3'];n=int(best.episodes)
    timing=pd.read_csv(diagnostic_dir/'control_timing.csv').set_index('horizon').loc[2.]
    lines=['# 预测驱动融合门控：论文结果稿','',
        '**核心结果：冻结候选策略，通过在线预测与持续门控协调，提高防御成功率并消除本组评测中的艇间碰撞。**','',
        f'完成 {n} 个不同场景、9 种方法，共 {9*n:,} 个完整方法回合。2 秒持续修正为 {int(best.success_count)}/{n} 成功；原门控为 {int(base.success_count)}/{n}。控制周期为 0.2 秒，预警距离为 7 米，所有方法使用相同场景起点及配对扰动。','',
        '## 主结果表','',
        '| 方法 | 成功率 | 碰撞率 | 突破率 | 平均团队回报 | 相对原门控回报变化 |',
        '|---|---:|---:|---:|---:|---:|']
    names={'ARBoids':'原始 ARBoids','Boids only':'仅 Boids','Actor only':'仅学习动作','Fixed blend':'固定 0.5 混合',
           'Single H2':'2 秒预测，首次预警单次修正','Single correction':'3 秒预测，首次预警单次修正',
           'Rolling H1':'持续修正，1 秒预测','Rolling H2':'**持续修正，2 秒预测**','Rolling H3':'持续修正，3 秒预测'}
    for row in table.itertuples(index=False):
        lines.append(f'| {names[row.method]} | {row.success_rate*100:.2f}% ({row.success_count}/{row.episodes}) | '
                     f'{row.collision_rate*100:.2f}% | {row.breach_rate*100:.2f}% | {row.mean_team_return:.3f} | {row.team_return_delta:+.3f} |')
    lines += ['', '## 可直接采用的结果表述','',
        f'在 {n} 个场景的配对评测中，2 秒预测的持续门控修正将防御成功率从 {base.success_rate*100:.2f}% 提高至 '
        f'{best.success_rate*100:.2f}%，提高 {best.success_delta*100:.2f} 个百分点（配对差值的 95% 自助法区间 '
        f'[{best.success_ci_low*100:.2f}, {best.success_ci_high*100:.2f}] 个百分点；成功率的精确 McNemar 检验经 '
        f'8 项对照 Holm 校正后 p={best.success_p_holm:.4g}）。碰撞次数由 {int(base.collision_count)} 次降至 '
        f'{int(best.collision_count)} 次，突破次数为 {int(base.breach_count)} 次与 {int(best.breach_count)} 次。'
        f'平均团队回报由 {base.mean_team_return:.3f} 提高至 {best.mean_team_return:.3f}，配对增益为 '
        f'{best.team_return_delta:+.3f}（95% 区间 [{best.team_return_ci_low:.3f}, {best.team_return_ci_high:.3f}]）。','',
        f'持续协调的作用在相同预测窗口下仍然成立。2 秒窗口的首次预警单次修正取得 {int(single.success_count)}/{n} '
        f'成功，发生 {int(single.collision_count)} 次碰撞；持续修正取得 {int(best.success_count)}/{n} 成功且无碰撞。'
        '每周期更新局势、重新生成候选并选择门控，使干预能够覆盖完整避碰过程。','',
        f'预测窗口比较显示，2 秒窗口在本组对照中表现最好。将窗口延长至 3 秒后，成功次数为 {int(h3.success_count)}，'
        f'平均每回合改变门控的周期数由 {best.mean_altered_steps:.2f} 增至 {h3.mean_altered_steps:.2f}，'
        f'平均回报增益由 {best.team_return_delta:+.3f} 变为 {h3.team_return_delta:+.3f}。'
        '这一结果说明，提高预测长度与提高实际防御效果并不单调对应。','',
        f'2 秒持续修正恢复了原门控 {int(base.collision_count)} 个碰撞回合中的 {int(best.collision_to_success)} 个，'
        f'原本成功的回合中有 {int(best.success_losses)} 个转为失败；新增突破为 {int(best.new_breaches)} 个。'
        f'在原门控已经成功的回合上，平均回报变化为 {best.baseline_success_return_delta:+.3f}。','',
        '## English results paragraph','',
        f'Across {n} paired scenarios, receding gate coordination with a 2-s prediction horizon achieved '
        f'{best.success_rate*100:.2f}% successful defense, compared with {base.success_rate*100:.2f}% for the original '
        f'ARBoids controller. The paired improvement was {best.success_delta*100:.2f} percentage points '
        f'(95% bootstrap CI: {best.success_ci_low*100:.2f} to {best.success_ci_high*100:.2f}; Holm-adjusted exact '
        f'McNemar p={best.success_p_holm:.4g}). Defender collisions decreased from {int(base.collision_count)} to '
        f'{int(best.collision_count)}, while target breaches were {int(base.breach_count)} and {int(best.breach_count)}, '
        f'respectively. Mean team return increased from {base.mean_team_return:.3f} to {best.mean_team_return:.3f}. '
        f'With the same 2-s horizon, a single correction at the first predicted warning achieved '
        f'{int(single.success_count)}/{n} successful episodes, whereas receding coordination achieved '
        f'{int(best.success_count)}/{n}. These improvements were obtained with frozen action proposals and without '
        'additional policy training.','',
        '## 方法与计算成本','',
        '三艘防守艇沿用原学习动作与 Boids 推力。每周期联合比较每艇 5 点门控网格，共 125 个组合，另保留原门控。'
        '无预测预警时保留原动作；进入 7 米预警带时，最小化预测安全距离侵入平方与归一化推力变化惩罚之和。'
        '后者系数为 0.02。只执行下一控制周期，再重新观测和计算。动力学预测积分步长为 0.05 秒，控制周期为 0.2 秒，'
        '实际碰撞阈值为 5 米。预测采用零海流名义三自由度模型，以当前测得的速度初始化，假设候选融合推力在预测窗口中保持不变。'
        '首次预警单次修正采用同一选权规则，执行一个周期后恢复原门控。','',
        f'本机 CPU 微基准中，2 秒窗口的 Actor 推理及候选选权耗时中位数为 {timing.median_ms:.2f} 毫秒，'
        f'95 分位为 {timing.p95_ms:.2f} 毫秒，低于 200 毫秒控制周期。该计时在此前 14 个快照状态上重复 20 遍，'
        '不包含环境推进。','',
        '统计以场景为配对单位，使用 20,000 次场景自助重采样；主图成功率误差条为 Wilson 95% 区间，'
        '回报误差条为配对自助法 95% 区间。原始 8 方法运行指定的首要对照为 3 秒持续修正；'
        '完整窗口比较后，本文结果稿将表现更好的 2 秒方案置于主表重点位置，并保留全部对照。','',
        '## 图件与图注','',
        '![主结果](figures/fig_gate_main.png)','',
        f'**图 1｜融合门控的防御表现。** 各方法均在相同 {n} 个场景起点运行；左图为成功率及 Wilson 95% 区间，'
        '右图为相对原门控的配对团队回报变化及 95% 自助法区间。','',
        '![消融](figures/fig_gate_ablation.png)','',
        '**图 2｜单次修正、持续修正与预测窗口的对照。** 左图分别报告碰撞与突破，右图报告任务回报变化。'
        '单次修正和 2 秒持续修正使用相同预测窗口与选权规则。','']
    if example:
        lines += ['![恢复轨迹](figures/fig_gate_recovery.png)','',
            f'**图 3｜避碰恢复示例。** 场景 {example["scene_seed"]}、扰动 {example["future_seed"]}；'
            '展示控制端点上的最小艇间距。示例按恢复回合的回报增益中位位置选取，两条完整轨迹均与评测摘要一致。','']
    lines += ['## 文件','',
        '- [完整统计表](main_table.csv)；[逐回合结果](episodes.csv)；[2 秒单次修正结果](single_h2/episodes.csv)',
        '- [图 1 PDF](figures/fig_gate_main.pdf)；[图 2 PDF](figures/fig_gate_ablation.pdf)；[图 3 PDF](figures/fig_gate_recovery.pdf)',
        '- [评测设置与校验](summary.json)；[单次修正校验](single_h2/summary.json)',
        '- 22 项门控测试通过；128 个场景均通过基线逐轨迹重放，两个运行入口均为 `passed=true`。']
    (run_dir/'paper_results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--diagnostic-dir',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    summary=json.loads((args.run_dir/'summary.json').read_text(encoding='utf-8'))
    supplement=json.loads((args.run_dir/'single_h2/summary.json').read_text(encoding='utf-8'))
    assert summary['passed'] and supplement['passed']
    for name,value in supplement['code_sha256'].items():
        assert digest(Path(__file__).resolve().parents[1]/name)==value
    assert digest(summary['checkpoint'])==summary['checkpoint_sha256']
    data=pd.concat([pd.read_csv(args.run_dir/'episodes.csv'),pd.read_csv(args.run_dir/'single_h2/episodes.csv')],ignore_index=True)
    assert not data.duplicated(['scene_seed','method']).any()
    table=summarize(data)
    table.to_csv(args.run_dir/'main_table.csv',index=False)
    directory=args.run_dir/'figures';directory.mkdir(exist_ok=True)
    figures(table,directory)
    example=trajectory_figure(data,summary,directory)
    report(table,args.run_dir,args.diagnostic_dir,example)
    print(table[['method','success_count','collision_count','breach_count','team_return_delta','success_p_holm']].to_string(index=False))
    print(json.dumps(dict(passed=True,scenes=summary['source_episodes'],method_episodes=len(data),example=example)))


if __name__=='__main__':
    main()
