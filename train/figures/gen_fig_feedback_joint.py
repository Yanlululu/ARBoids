"""Paired analysis and figures for the frozen feedback/control-space repair."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import study_runtime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import binomtest

from gen_fig_gate_contribution import read_csv, save_csv, save, wilson
from evaluate_feedback_joint import METHODS, BASELINES, DIAGNOSTIC_ROOT, code_hashes, CHECKPOINT
from source_arboids import sha256, verify_source


LABELS={'original':'Original ARBoids','reactive_joint':'Reactive gate','predictive_joint':'Old predictive gate',
        'cbf':'CBFpy','feedback_joint':'Feedback + full thrust','held_joint':'Held prediction + full thrust',
        'feedback_gate':'Feedback + gate projection'}
CN={'original':'原始 ARBoids','reactive_joint':'瞬时联合门控','predictive_joint':'旧预测联合门控',
    'cbf':'CBF','feedback_joint':'反馈预测＋完整推力','held_joint':'保持推力预测＋完整推力',
    'feedback_gate':'反馈预测＋线段投影'}
COLORS=dict(zip(METHODS,['#777777','#56B4E9','#D55E00','#009E73','#0072B2','#E69F00','#CC79A7']))


def number(row,key):
    return float(row[key])


def bootstrap(values,seed=719,repetitions=10000):
    a=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    means=a[rng.integers(0,len(a),size=(repetitions,len(a)))].mean(axis=1)
    return float(a.mean()),*map(float,np.quantile(means,[.025,.975]))


def clustered(rows,key):
    groups=defaultdict(list)
    for r in rows:
        groups[int(r['scene_seed'])].append(number(r,key))
    return bootstrap([np.mean(a) for a in groups.values()])


def subsets(rows):
    return {'all':rows,'main':[r for r in rows if r['cell'].startswith('main-')],
            'generalization':[r for r in rows if not r['cell'].startswith('main-')]}


def tables(rows):
    result=[]
    for suite,group in subsets(rows).items():
        for method in METHODS:
            part=[r for r in group if r['method']==method]
            result.append(dict(suite=suite,method=method,episodes=len(part),
                **{key:sum(int(r[key]) for r in part) for key in ('success','collision','source_loss','capture','timeout')},
                return_mean=float(np.mean([number(r,'team_return') for r in part])),
                duration_mean=float(np.mean([number(r,'duration') for r in part]))))
    return result


def comparisons(rows):
    result=[]
    for suite,group in subsets(rows).items():
        index={(int(r['scene_seed']),r['method']):r for r in group}
        seeds=sorted({int(r['scene_seed']) for r in group})
        family=[]
        for method in METHODS:
            if method=='feedback_joint':
                continue
            new=[index[s,'feedback_joint'] for s in seeds]
            other=[index[s,method] for s in seeds]
            delta=np.array([int(a['success'])-int(b['success']) for a,b in zip(new,other)])
            wins,losses=int(np.sum(delta>0)),int(np.sum(delta<0))
            d,lo,hi=bootstrap(delta)
            rd,rlo,rhi=bootstrap([number(a,'team_return')-number(b,'team_return') for a,b in zip(new,other)])
            family.append(dict(suite=suite,reference=method,episodes=len(seeds),wins=wins,losses=losses,
                delta_success_pp=100*d,ci_low_pp=100*lo,ci_high_pp=100*hi,
                p_raw=float(binomtest(wins,wins+losses).pvalue) if wins+losses else 1.,
                delta_return=rd,return_ci_low=rlo,return_ci_high=rhi))
        running=0.
        for rank,row in enumerate(sorted(family,key=lambda r:r['p_raw'])):
            running=max(running,min(1.,(len(family)-rank)*row['p_raw']))
            row['p_holm']=running
        result.extend(family)
    return result


def prediction_summary(rows):
    for r in rows:
        r['absolute_dmin_error']=abs(number(r,'dmin_error'))
    result=[]
    for category in ('all','natural','failure'):
        group=[r for r in rows if category=='all' or
               (r['category']=='failure')==(category=='failure')]
        for mode in ('feedback','held'):
            part=[r for r in group if r['prediction']==mode]
            row=dict(category=category,prediction=mode,states=len(part),
                     scenes=len({r['scene_seed'] for r in part}),
                     observed_steps_mean=float(np.mean([number(r,'observed_steps') for r in part])))
            for key in ('position_rmse','absolute_dmin_error'):
                mean,lo,hi=clustered(part,key)
                row.update({key:mean,key+'_low':lo,key+'_high':hi})
            result.append(row)
    return result


def recovery_tables(rows):
    result=[]
    for future in ('recorded','unseen'):
        for method in METHODS:
            part=[r for r in rows if r['future']==future and r['method']==method]
            mean,lo,hi=clustered(part,'success')
            result.append(dict(future=future,method=method,branches=len(part),
                scenes=len({r['scene_seed'] for r in part}),
                **{key:sum(int(r[key]) for r in part) for key in ('success','collision','source_loss')},
                scene_success_mean=mean,scene_ci_low=lo,scene_ci_high=hi))
    return result


def recovery_comparisons(rows):
    results=[]
    for future in ('recorded','unseen'):
        group=[r for r in rows if r['future']==future]
        index={(r['case_id'],int(r['repeat']),r['method']):r for r in group}
        keys=sorted({(r['case_id'],int(r['repeat'])) for r in group})
        for method in ('predictive_joint','cbf','held_joint','feedback_gate'):
            differences=[]
            for k in keys:
                a,b=index[(*k,'feedback_joint')],index[(*k,method)]
                differences.append(dict(scene_seed=a['scene_seed'],delta=int(a['success'])-int(b['success'])))
            mean,lo,hi=clustered(differences,'delta')
            results.append(dict(future=future,reference=method,
                wins=sum(r['delta']>0 for r in differences),losses=sum(r['delta']<0 for r in differences),
                scene_delta_pp=100*mean,ci_low_pp=100*lo,ci_high_pp=100*hi))
    return results


def make_figures(root,table,contrasts,predictions,recovery,runtime):
    folder=root/'figures'
    folder.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.25),sharey=True,layout='constrained')
    for ax,suite,title in zip(axes,('main','generalization'),('a  New three-vessel scenes (128)','b  Team / agility variations (256)')):
        for y,method in enumerate(METHODS):
            row=next(r for r in table if r['suite']==suite and r['method']==method)
            value=row['success']/row['episodes']
            lo,hi=wilson(row['success'],row['episodes'])
            ax.errorbar(100*value,y,xerr=np.array([[value-lo],[hi-value]])*100,fmt='o',
                        color=COLORS[method],capsize=3)
            ax.text(101.,y,f'{row["success"]}/{row["episodes"]}',ha='left',va='center',fontsize=8)
        ax.set(title=title,xlabel='Defense success (%)',xlim=(50,115),xticks=[50,60,70,80,90,100])
    axes[0].set_yticks(range(len(METHODS)),[LABELS[m] for m in METHODS])
    axes[0].invert_yaxis()
    save(fig,folder,'fig_feedback_results')
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
    for ax,key,title in zip(axes,('position_rmse','absolute_dmin_error'),('a  Position prediction','b  Minimum separation prediction')):
        for j,mode in enumerate(('held','feedback')):
            group=[next(r for r in predictions if r['category']==c and r['prediction']==mode)
                   for c in ('natural','failure')]
            y=np.array([r[key] for r in group]);lo=np.array([r[key+'_low'] for r in group]);hi=np.array([r[key+'_high'] for r in group])
            ax.bar(np.arange(2)+(j-.5)*.33,y,.3,label={'held':'Held thrust','feedback':'Matching feedback'}[mode],
                   color=('#E69F00','#0072B2')[j],yerr=[y-lo,hi-y],capsize=3)
        ax.set(title=title,xticks=[0,1],xticklabels=['Natural states','Failure states'],ylabel='Error (m)')
    axes[0].legend(loc='upper left')
    fig.suptitle('Same selected policy, actual committed execution | scene-clustered 95% CI',fontsize=10)
    save(fig,folder,'fig_feedback_prediction')
    fig,axes=plt.subplots(1,2,figsize=(8.2,3.25),sharey=True,layout='constrained')
    for ax,future,title in zip(axes,('recorded','unseen'),('a  Recorded future (60 branches)','b  New future noise (180 branches)')):
        for y,method in enumerate(METHODS):
            r=next(r for r in recovery if r['future']==future and r['method']==method)
            ax.barh(y,100*r['success']/r['branches'],color=COLORS[method],height=.64)
            ax.text(100*r['success']/r['branches']+1,y,f'{r["success"]}/{r["branches"]}',va='center',fontsize=8)
        ax.set(title=title,xlabel='Task recovery (%)',xlim=(0,90))
    axes[0].set_yticks(range(len(METHODS)),[LABELS[m] for m in METHODS]);axes[0].invert_yaxis()
    fig.suptitle('Same 60 failure states from 20 source scenes',fontsize=10)
    save(fig,folder,'fig_feedback_recovery')
    fig,axes=plt.subplots(1,2,figsize=(7.1,2.8),layout='constrained')
    for ax,phase,title in zip(axes,('planning','feedback'),('a  Planning ticks','b  Ordinary feedback ticks')):
        group=[r for r in runtime if r['phase']==phase]
        x=[int(r['defenders']) for r in group]
        for key,label,marker in [('median_ms','Median','o'),('p95_ms','95th percentile','s'),('p99_ms','99th percentile','^')]:
            ax.plot(x,[float(r[key]) for r in group],marker=marker,label=label)
        ax.set(title=title,xlabel='Defender count',ylabel='Control time (ms)',xticks=[2,3,4,5],ylim=(0,None))
    axes[0].axhline(200,color='#D55E00',ls='--',label='200 ms deadline')
    axes[0].set_ylim(0,212)
    axes[0].legend(fontsize=7,loc='upper left',bbox_to_anchor=(.01,.91))
    save(fig,folder,'fig_feedback_runtime')


def report(root,table,contrasts,predictions,recovery,recovery_pairs,runtime,raw,vrx):
    frozen=json.loads((root/'frozen_method.json').read_text(encoding='utf-8'))
    accelerated=json.loads((root/'acceleration_summary.json').read_text(encoding='utf-8'))
    runtime_info=json.loads((root/'runtime_summary.json').read_text(encoding='utf-8'))
    index={(r['suite'],r['method']):r for r in table}
    pi={(r['category'],r['prediction']):r for r in predictions}
    full=index['all','feedback_joint'];gate=index['all','feedback_gate']
    outside=sum(number(r,'max_projection_residual')>10. for r in raw if r['method']=='feedback_joint')
    maximum=max(number(r,'max_projection_residual') for r in raw if r['method']=='feedback_joint')
    lines=['# 重规划失配与动作空间限制：修复及验证','',
        '两项结构问题均已改到控制器实现中：预测逐周期执行同一反馈律，外层只在已承诺的反馈段结束时更新；'
        '每周期优化全部防守艇的左右推力，融合动作只提供参考值。原始论文源码、奖励、观测、终止规则和对照控制器保持原样。','',
        f'冻结后的 384 个新场景中，新方法成功 **{full["success"]}/384、碰撞 {full["collision"]} 次**；'
        f'重新投影回融合线段后为 **{gate["success"]}/384、碰撞 {gate["collision"]} 次**。'
        '动作空间限制的实际影响得到支持。反馈预测的一致性与误差改善单独验证；本轮最终成功率没有超过 CBF 或保持推力预测消融，不能把结构修复写成独立性能优势。','',
        '## 实现如何消除两个问题','',
        '**反馈与重规划一致。** 每 0.2 秒重新计算原 Actor、Boids 和连续联合 CBF。外层每 2 秒选择一次反馈参数，'
        '预测只推进这 2 秒，并调用与执行端同一个 `feedback()`。参数在段内固定，物理推力持续随新观测更新。'
        '预测从不假定段末尚未作出的下一次外层选择。执行端不接收预测生成的固定推力序列。','',
        '**联合动作自由度。** 优化变量是 `2N` 个推进器推力，界限为每个推进器 `[-500,1000] N`。'
        '原融合动作只是二次代价的参考点，优化没有“必须落在 Actor–Boids 连线内”的约束。'
        '沿用强对照的 CBF 安全距离、增益、松弛和容差。五种外层参数影响参考动作，不限制 QP 的物理动作维数。','',
        f'新方法在 **{outside}/384** 个完整场景中产生了距离原融合线段超过 **10 N** 的控制，'
        f'最大逐艇投影残差为 **{maximum:.3f} N**，确认了实际物理动作自由度的扩展。','',
        '结构回归测试共 **16 项通过**，覆盖参考与加速实现：1/2 秒反馈段、跨三次段末重规划的命令与轨迹一致、'
        '每周期反馈更新、随机流不变、脱离融合线段、原 APF 去噪一致性、公开状态输入和回合重置。'
        '无扰动一致性测试的轨迹容差为 `1e-8`、推力容差为 `1e-5 N`。这些检验针对控制假设，扰动误差另列如下。','',
        '## 冻结方法与新场景结果','',
        '开发只使用既有 64 场景，在 1 秒与 2 秒反馈段中选一次；成功数分别为 '
        f'{frozen["candidates"][0]["successes"]}/64 和 {frozen["candidates"][1]["successes"]}/64，均无碰撞，因而冻结 2 秒。'
        '随后运行 128 个新三艇场景，以及艇数 2–5、攻击艇敏捷度 1.5/2/2.5/3 的 16 格×16 场景。'
        '七种方法共用每个场景的初态与随机流；384 条原始算法完整轨迹均与直接调用未修改源码的结果一致。','',
        '| 方法 | 新三艇：成功/128 | 变化场景：成功/256 | 总成功/384 | 碰撞 | 源码失守 | 平均队伍回报 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method in METHODS:
        row=index['all',method]
        lines.append(f'| {CN[method]} | {index["main",method]["success"]} | {index["generalization",method]["success"]} | '
                     f'{row["success"]} | {row["collision"]} | {row["source_loss"]} | {row["return_mean"]:.2f} |')
    lines+=['','成功沿用源码定义，包含捕获和守至超时；新完整方法与保持推力消融各有一次超时成功，其余成功均为捕获。'
            '“源码失守”包含攻击艇比所有防守艇更接近目标时的提前终止，不等同于到达目标圆内。','',
            '![新场景验证](figures/fig_feedback_results.png)','',
            '同场景配对差异如下，方向为“新完整方法减对照”。成功率区间使用 10,000 次场景配对 bootstrap；'
            '成功差异采用双侧精确 McNemar 检验，在每个场景套件内对六项比较做 Holm 校正。区间未做同时覆盖校正。','',
            '| 对照 | 新增成功 / 丢失成功 | 成功率差（百分点，95% CI） | Holm p | 队伍回报差 |',
            '|---|---:|---:|---:|---:|']
    for row in contrasts:
        if row['suite']=='all':
            lines.append(f'| {CN[row["reference"]]} | {row["wins"]} / {row["losses"]} | '
                f'{row["delta_success_pp"]:+.2f} [{row["ci_low_pp"]:.2f}, {row["ci_high_pp"]:.2f}] | '
                f'{row["p_holm"]:.4g} | {row["delta_return"]:+.2f} |')
    lines+=['','完整方法比线段投影消融显著改善成功率，但没有胜过 CBF；保持推力预测消融的成功数反而更多。'
            '因此本轮证明了动作空间修复的任务价值，尚未证明更准确的反馈预测经当前外层代价能够产生更高成功率。'
            '数值验证后未修改评分权重、候选策略、段长或样本量。','',
            '## 相同已选策略下的预测误差','',
            '在已有 103 个状态上先选定完整方法的同一反馈策略，再比较两种预测与实际反馈执行：'
            '逐周期重算反馈，或保持首次推力。模型输入、策略参数、真实执行和比较时间前缀完全相同；'
            '遇到原始终止就截断，不补造后续轨迹。自然状态与失败前状态分别按原始场景等权汇总。','',
            '| 状态组 | 状态数 | 比较前缀均值（秒） | 保持推力：位置 RMSE | 反馈预测：位置 RMSE | 保持推力：最小间距 MAE | 反馈预测：最小间距 MAE |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for category,title in [('natural','自然状态'),('failure','失败前状态')]:
        a,b=pi[category,'held'],pi[category,'feedback']
        lines.append(f'| {title} | {b["states"]} | {b["observed_steps_mean"]*.2:.2f} | '
                     f'{a["position_rmse"]:.3f} m | {b["position_rmse"]:.3f} m | '
                     f'{a["absolute_dmin_error"]:.3f} m | {b["absolute_dmin_error"]:.3f} m |')
    lines+=['','![反馈一致性与误差](figures/fig_feedback_prediction.png)','',
            '当前名义模型仍使用零海流、零 APF 随机力和固定攻击艇敏捷度 2.25。上述剩余误差包含扰动、'
            '攻击艇模型差异以及速度观测误差；它与过去“假设保持推力、实际每步换控制”的逻辑失配分开报告。','',
            '## 回到相同失败状态','',
            '使用先前固定的 20 个源场景、60 个失败前状态。每状态保留原记录的未来随机流，再使用三个新未来随机流。'
            '原始、旧瞬时/预测门控和 CBF 数据直接复用先前同状态结果，新方法及两项消融运行 720 条完整分支。'
            '以下分母是分支数，统计区间另按 20 个原始场景聚类计算。','',
            '| 方法 | 原未来：成功/60 | 碰撞 | 失守 | 新未来：成功/180 | 碰撞 | 失守 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    ri={(r['future'],r['method']):r for r in recovery}
    for method in METHODS:
        a,b=ri['recorded',method],ri['unseen',method]
        lines.append(f'| {CN[method]} | {a["success"]} | {a["collision"]} | {a["source_loss"]} | '
                     f'{b["success"]} | {b["collision"]} | {b["source_loss"]} |')
    lines+=['','![相同状态恢复](figures/fig_feedback_recovery.png)','',
            '## 实时执行与 VRX','',
            '首次 VRX 接入发现五候选规划周期有 7 次超过 200 ms。随后只把名义动力学中的独立艇积分向量化；'
            '反馈律、策略候选、评分、QP 与更新时刻均保持冻结。参考实现仍保存用于数值复现。','',
            f'103 个状态的 515 条候选预测中，加速前后策略选择全部一致；最大状态差 '
            f'{accelerated["maximum_state_error"]:.3g}，最大推力差 {accelerated["maximum_thrust_error"]:.3g} N，'
            f'最大评分差 {accelerated["maximum_score_error"]:.3g}。跨重规划回归测试也通过。','',
            '下表是在无其他本任务仿真运行时、单进程单 Torch 线程下测得的完整控制计算，包含状态读取、Actor、'
            '候选反馈预测、联合 QP 和动作映射，排除环境积分及 ROS 发布；首次编译另记于 JSON。','',
            '| 防守艇数 | 规划：中位 / P95 / P99（ms） | 普通反馈：中位 / P95 / P99（ms） | 超过 200 ms |',
            '|---|---:|---:|---:|']
    for n in (2,3,4,5):
        a=next(r for r in runtime if int(r['defenders'])==n and r['phase']=='planning')
        b=next(r for r in runtime if int(r['defenders'])==n and r['phase']=='feedback')
        v=lambda r:' / '.join(f'{float(r[k]):.2f}' for k in ('median_ms','p95_ms','p99_ms'))
        lines.append(f'| {n} | {v(a)} | {v(b)} | {int(a["deadline_misses"])+int(b["deadline_misses"])} |')
    lines+=['','![控制链耗时](figures/fig_feedback_runtime.png)','',
            f'测量主机：{runtime_info["processor"]}；{runtime_info["platform"]}。','',
            '真实 Gazebo 使用固定四个场景（三艇港口、三艇开阔水面、四艇港口、五艇港口），每个场景运行 '
            'CBF 和新方法；首次超时定位及加速后的验证分开保存，不并入过去的 VRX 统计。原始 APF、Boids、'
            '观测和终止判断不变。预测用数值名义模型，实际 Gazebo 的物理与攻击艇行为仍由原 VRX 执行。','',
            '| 场景 | CBF 结果 | 新方法结果 | 新方法 P99 / 最大控制时间（ms） | 新方法超时 | 初始位置最大差（m） |',
            '|---|---|---|---:|---:|---:|']
    pairs={r['cell']:r for r in read_csv(root/'vrx_fast/pairs.csv')}
    outcome=lambda r:{1:'失守',2:'碰撞',3:'捕获',4:'守至超时'}[int(r['outcome'])]
    for cell in ('dock-n3','ocean-n3','dock-n4','dock-n5'):
        a=next(r for r in vrx if r['cell']==cell and r['method']=='cbf')
        b=next(r for r in vrx if r['cell']==cell and r['method']=='feedback_joint')
        lines.append(f'| {cell} | {outcome(a)} | {outcome(b)} | '
                     f'{float(b["control_p99_ms"]):.2f} / {float(b["control_max_ms"]):.2f} | '
                     f'{b["deadline_misses"]} | {float(pairs[cell]["initial_error_m"]):.3g} |')
    isolated=[]
    for cell,seed in [('dock-n3',157000000),('dock-n5',157000003)]:
        path=root/'vrx_isolated'/f'{cell}-{seed}-feedback_joint'/'result.json'
        if path.exists():
            result=json.loads(path.read_text())
            assert result['passed'] and result['source_unchanged'] and result['feedback_code_unchanged']
            data=np.load(path.with_name('trajectory.npz'))
            baseline=next(r for r in vrx if r['cell']==cell and r['method']=='feedback_joint')
            assert result['initial_poses']==baseline['initial_poses']
            isolated.append(dict(cell=cell,steps=result['control_steps'],
                max_ms=1000*result['control_seconds']['maximum'],
                p99_ms=1000*float(np.quantile(data['ControlWallSeconds'],.99)),
                misses=result['control_deadline_misses']))
    if len(isolated)==2:
        lines+=['','并行两套 Gazebo 的加速版仍有 3 次超时，因此对这两组发生超时的场景按相同种子、'
                '相同参数逐个运行，检查并发负载。两次单独运行均成功捕获；它们用于延迟诊断，未替换上表的并行结果。','',
                '| 单独运行场景 | 控制周期数 | P99 / 最大控制时间（ms） | 超过 200 ms |',
                '|---|---:|---:|---:|']
        for row in isolated:
            lines.append(f'| {row["cell"]} | {row["steps"]} | {row["p99_ms"]:.2f} / {row["max_ms"]:.2f} | {row["misses"]} |')
        lines+=['','单进程控制链及上述单独运行满足本轮 200 ms 预算；并发仿真的超时仍作为实际测量结果保留。'
                '部署时应为控制器保留计算资源，不能把离线批量实验的并发配置直接当作实时配置。']
    lines+=['','## 文件与复现','',
        '- [控制器参考实现](../../feedback_joint_control.py)、[等价向量化实现](../../feedback_joint_fast.py)、[回归测试](../../tests/test_feedback_joint_control.py)。',
        '- [固定协议](../../../docs/feedback-joint-control.md)、[冻结参数](frozen_method.json)、[新场景完整数据](confirmation.csv)、[配对比较](paired_comparisons.csv)。',
        '- [同状态恢复分支](recovery.csv)、[聚类配对恢复比较](recovery_comparisons.csv)、[同策略预测误差](prediction.csv)。',
        '- [加速等价检查](acceleration_summary.json)、[独立延迟统计](runtime_summary.json)、[原 VRX 超时定位](vrx/summary.json)、[加速后 VRX 验证](vrx_fast/summary.json)。',
        '- 图像均附同名矢量 PDF 和 300 dpi PNG；汇总由 [绘图脚本](../../figures/gen_fig_feedback_joint.py) 从原始 CSV 生成。','',
        '接入数值环境时使用冻结的 `block_steps=10`，每个回合调用一次 `reset()`，随后每周期传入新观测：','',
        '```python',
        'from feedback_joint_control import observe',
        'from feedback_joint_fast import FastFeedbackJointController',
        'rule = FastFeedbackJointController(env.defender_num, policy, block_steps=10)',
        'rule.reset()',
        'done = 0',
        'while not done:',
        '    thrust, info = rule.control(observe(env, observations))',
        "    observations, reward, done, _ = env.step(env.thrust_to_action(thrust), 'RL')",
        '```','',
        'VRX 使用[专用入口](../../../vrx/run_feedback_joint.py)的 `--method feedback_joint --prediction-engine vectorized`，'
        '并用 `--frozen-method` 指向本目录的 `frozen_method.json`；入口会验证冻结源码和加速等价检查。','',
        '```powershell',
        "& 'D:\\ARBoids\\.venv\\Scripts\\python.exe' -X utf8 train/evaluate_feedback_joint.py --stage develop --workers 4",
        "& 'D:\\ARBoids\\.venv\\Scripts\\python.exe' -X utf8 train/evaluate_feedback_joint.py --stage confirm --workers 4",
        "& 'D:\\ARBoids\\.venv\\Scripts\\python.exe' -X utf8 train/evaluate_feedback_joint.py --stage recover --workers 4",
        "& 'D:\\ARBoids\\.venv\\Scripts\\python.exe' -X utf8 train/evaluate_feedback_joint.py --stage prediction --workers 4",
        "& 'D:\\ARBoids\\.venv\\Scripts\\python.exe' -X utf8 train/audit_feedback_acceleration.py",
        "& 'D:\\ARBoids\\.venv\\Scripts\\python.exe' -X utf8 train/benchmark_feedback_joint.py",
        "& 'D:\\ARBoids\\.venv\\Scripts\\python.exe' -X utf8 train/figures/gen_fig_feedback_joint.py",
        '```','',
        '未训练新网络。使用按原训练配置得到的冻结权重，SHA-256 为 '
        f'`{frozen["checkpoint_sha256"]}`；它不是作者发布的预训练权重。原代码与对照的逐字节身份验证保留在 specification 与各阶段 summary 中。',
        '',
        '实验组织继续采用已声明的 experimental-design 工作流：场景配对、开发后冻结、状态恢复按源场景聚类。'
        '设计来源沿用[前一份机制诊断报告](../gate-mechanism-diagnostics-20261006/results.md)。','']
    (root/'results.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('train/experiments/feedback-joint-20261006'))
    args=parser.parse_args()
    root=args.root
    frozen=json.loads((root/'frozen_method.json').read_text(encoding='utf-8'))
    spec=json.loads((root/'specification.json').read_text(encoding='utf-8'))
    assert code_hashes()==frozen['code']==spec['code']
    assert verify_source()==spec['source'] and sha256(CHECKPOINT)==spec['checkpoint_sha256']
    assert sha256(Path('docs/feedback-joint-control.md'))==spec['protocol_sha256']
    for stage in ('develop','confirm','recover','prediction'):
        assert json.loads((root/(stage+'_summary.json')).read_text())['passed']
    rows=read_csv(root/'confirmation.csv')
    assert len(rows)==384*len(METHODS)
    assert len({(r['scene_seed'],r['method']) for r in rows})==len(rows)
    assert set(int(r['scene_seed']) for r in rows)==set(spec['new_scene_seeds'])
    assert all(r['original_equivalent']=='True' for r in rows if r['method']=='original')
    table=tables(rows);paired=comparisons(rows)
    prediction=read_csv(root/'prediction.csv')
    assert len(prediction)==206
    prediction_table=prediction_summary(prediction)
    recovery=read_csv(root/'recovery.csv')
    assert len(recovery)==720
    recovery += [r for r in read_csv(DIAGNOSTIC_ROOT/'recovery_episodes.csv') if r['method'] in BASELINES]
    assert len(recovery)==60*4*len(METHODS)
    recovery_table=recovery_tables(recovery)
    recovery_pairs=recovery_comparisons(recovery)
    runtime=read_csv(root/'runtime_summary.csv')
    vrx=read_csv(root/'vrx_fast/episodes.csv')
    assert len(vrx)==8
    assert json.loads((root/'vrx_fast/summary.json').read_text())['passed']
    for name,data in [('main_table',table),('paired_comparisons',paired),('prediction_summary',prediction_table),
                      ('recovery_summary',recovery_table),('recovery_comparisons',recovery_pairs)]:
        save_csv(root/(name+'.csv'),data)
    make_figures(root,table,paired,prediction_table,recovery_table,runtime)
    report(root,table,paired,prediction_table,recovery_table,recovery_pairs,runtime,rows,vrx)
    print(json.dumps(dict(passed=True,confirmation_scenes=384,confirmation_rows=len(rows),
                         recovery_branches=len(recovery),prediction_states=103,figures=4),ensure_ascii=False))


if __name__=='__main__':
    main()
