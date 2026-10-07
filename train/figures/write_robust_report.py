"""Update the existing report from completed, frozen confirmation artifacts."""
import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BASE = Path('train/experiments/robust-predictive-interception-20261006')
ROOT = REPO/BASE/'confirmation'
NAMES = dict(original='原始 ARBoids', cbf='CBF', predictive='持续反馈长时预测',
             short_value='同参数 1 秒短时消融', strongest_short='较强的 2 秒短时方案',
             best_fixed='固定守卫策略', legacy_value='旧评分 + CBF 后续')


def read_csv(name):
    with (ROOT/name).open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def main():
    result = json.loads((ROOT/'analysis.json').read_text(encoding='utf-8'))
    if not result['passed']:
        raise RuntimeError('Analysis has not completed.')
    table = read_csv('outcome_summary.csv')
    index = {r['method']:r for r in table}
    contrasts = read_csv('paired_comparisons.csv')
    comparison = {r['reference']:r for r in contrasts}
    cells = read_csv('cell_summary.csv')
    runtime = json.loads((REPO/BASE/'runtime-sustain_h5.json').read_text(encoding='utf-8'))
    development = json.loads((REPO/BASE/'development2/summary.json').read_text(encoding='utf-8'))
    spec = json.loads((ROOT/'specification.json').read_text(encoding='utf-8'))
    execution = json.loads((ROOT/'execution.json').read_text(encoding='utf-8'))
    row = index['predictive']
    final = '本轮全部主要判据及计算时延要求均已通过。' if result['all_requirements_met'] else '本轮尚未全部通过主要判据。'
    original = (REPO/'results.md').read_text(encoding='utf-8')
    if '## 前两轮记录：旧版本的原始结果' in original:
        historical = original.split('### 第一轮：20 秒后续价值',1)[1]
        historical = '### 第一轮：20 秒后续价值'+historical
        historical, bibliography = historical.split('## 参考文献与软件',1)
    else:
        historical = original.split('## 第一轮：20 秒后续价值',1)[1]
        historical = '## 第一轮：20 秒后续价值'+historical
        historical, bibliography = historical.split('## 参考文献与软件',1)
        historical = '\n'.join('#'+line if line.startswith('#') else line for line in historical.splitlines())
    sections = [f'''# 预测拦截：成功率、任务效率与计算代价

更新日期：2026-10-06。主任务是六艇防御，评价成功率、未捕获封顶拦截耗时和200毫秒控制预算。

## 最新结果

**{final}** 冻结后的新版本在512个全新六艇场景中，成功 **{row['success']}/512**，捕获 **{row['capture']}/512**，碰撞 **{row['collision']}**，平均未捕获封顶耗时 **{float(row['capped_capture_time']):.2f}秒**。相对CBF耗时降低 **{float(comparison['cbf']['improvement_percent']):.2f}%**；相对同参数1秒短时消融降低 **{float(comparison['short_value']['improvement_percent']):.2f}%**；相对开发阶段选出的较强2秒短时方案降低 **{float(comparison['strongest_short']['improvement_percent']):.2f}%**。

同一最终实现的五进程并行版，独立测得完整规划调用 **p95 136.0毫秒、p99 146.6毫秒、最大160.4毫秒**。32个完整旧场景中的535次规划、2605次控制调用均未超过200毫秒；串行与并行的2605个控制输出和535个选中计划逐元素相同。启动编译另计，不能把这些数字理解为包含通信和船载执行器的端到端时延。

前两轮旧版本的主判据没有通过，原记录保留在后文。本轮先用已经观察过的数据修复和选择方法，再冻结代码、参数、分析程序及512个新场景名单；新的确认结果没有用于改参数。

## 改了什么，为什么有用

新版本每1秒选择一次整队反馈策略，每0.2秒根据最新观测重新计算控制。候选仍为原始CBF、Actor控制端点、Boids控制端点、目标运动外推导航和单艇守卫导航；所有实际推力仍通过原来冻结的官方CBFpy安全过滤器，未更换原始物理环境、噪声、Actor权重或任务判定。

每个候选先推演1秒执行区间，再持续使用该候选反馈策略推演最多20秒。后续推演因此评价该机动继续执行时的捕获效果。旧版本在执行区间后切回CBF，可能低估需要持续机动的候选。实际系统每秒重新选择，20秒尾部是明确假设下的价值估计；它不预知将来的重规划选择。与同参数短时消融相比，唯一的控制设置差异是是否保留这20秒后续推演。

还修正了一个失败排序错误：当所有候选都预测会失守时，旧“终止时间＋剩余捕获估计”可能偏好更早失守。新公式在相同失败类别内优先推迟失败。这项修正同时应用于新长时和两种短时方法，避免把纠错收益冒充长时预测的贡献。另保留旧评分与CBF后续的版本，使用同一加速实现作版本对照。

攻击艇机动系数仍由过去的公开观测在线估计，初值2.25；控制器不读取场景真实系数或未来随机数。预测捕获半径比实际半径缩小0.5米；离开CBF基线的候选保留0.15秒代价。碰撞、源码失守和估计捕获耗时的选择优先级不变。

计算加速采用批量Actor、数组Boids与观测生成、JIT双精度模型积分，以及五个候选并行推演。预测中省去未用于评分的奖励和重复约束诊断，实际控制仍记录诊断。批量网络和JIT改变浮点运算顺序，因此相对原参考实现检验数值误差；同一新实现的串行和并行则要求完全相同。已有两批256个旧六艇场景上，加速后的旧逻辑版本复现了原版本每批的成功数、捕获数和平均耗时。

这种选择利用有限候选的反馈rollout估计长期任务价值，背景可参见[Bertsekas的rollout研究](https://arxiv.org/abs/1910.00120)。收益以以下配对实验为依据。

## 第三轮：512个全新六艇场景

四档攻击艇机动系数1.5、2、2.5、3各128场。原任务将捕获和坚持至80秒都算作成功；以下分开列出。效率指标将所有未捕获回合计为80秒，包含超时成功。

| 方法 | 成功 / 512 | 捕获 | 碰撞 | 源码失守 | 超时成功 | 封顶耗时 / 秒 |
|---|---:|---:|---:|---:|---:|---:|''']
    for r in table:
        sections.append(f"| {NAMES[r['method']]} | {r['success']} | {r['capture']} | {r['collision']} | {r['source_loss']} | {r['timeout']} | {float(r['capped_capture_time']):.3f} |")
    sections.append('''
| 新版本的主要对照 | 平均耗时降低 | 配对耗时差 / 秒，98.333% CI | 成功率差 / 百分点 | 成功率差保守下界 / 百分点 | 全部判据 |
|---|---:|---|---:|---:|---|''')
    for name in result['primary_references']:
        r = comparison[name]
        passed = '通过' if r['meets_strong_criteria']=='True' else '未通过'
        sections.append(f"| {NAMES[name]} | {float(r['improvement_percent']):.2f}% | {float(r['delta_capture_seconds']):+.3f} [{float(r['delta_ci_low']):+.3f}, {float(r['delta_ci_high']):+.3f}] | {float(r['delta_success_pp']):+.3f} | {float(r['success_conservative_lower_pp']):+.3f} | {passed} |")
    sections.append(f'''
耗时差为“新方法−对照”，负值更好。主要判据要求对每个主要对照同时达到：平均封顶耗时降低至少10%，配对分层10000次自举区间排除零改善，观察到的成功数不少、碰撞数不多，成功率差保守下界不低于−3个百分点。四项对照全部纳入判定。98.333%区间用于保守处理三次确认机会，前两轮原定区间作为历史描述保留。

![第三轮成功与效率]({BASE.as_posix()}/confirmation/figures/fig_robust_primary.png)

![第三轮配对耗时]({BASE.as_posix()}/confirmation/figures/fig_robust_paired.png)

| 攻击方机动系数 | CBF成功 / 128 | 新版本成功 / 128 | 新版本捕获 / 128 | CBF封顶耗时 / 秒 | 新版本 / 秒 |
|---|---:|---:|---:|---:|---:|''')
    for cell in sorted({r['cell'] for r in cells}):
        group = {r['method']:r for r in cells if r['cell']==cell}
        a,b = group['predictive'],group['cbf']
        sections.append(f"| {cell.split('-a')[1]} | {b['success']} | {a['success']} | {a['capture']} | {float(b['capped_capture_time']):.3f} | {float(a['capped_capture_time']):.3f} |")
    append_runtime(sections,runtime)
    append_audit(sections,result,spec,execution,development)
    sections.extend(['\n## 前两轮记录：旧版本的原始结果\n\n以下两轮结果和时延属于各自旧实现，不用于替代第三轮确认。\n', historical,
                     '\n## 参考文献与软件\n',bibliography])
    (REPO/'results.md').write_text('\n'.join(sections),encoding='utf-8')
    print(json.dumps(dict(passed=True,report=str(REPO/'results.md'),all_requirements_met=result['all_requirements_met']),ensure_ascii=False))


def append_runtime(sections,runtime):
    sections.append('''
## 最终版本的独立时延

在其他实验工作进程退出后，使用32个旧六艇开发场景，每档机动系数8场，运行完整回合。计时包含观测适配、全部候选推演、CBF、策略选择和动作映射；环境积分与外部通信不计。串行与并行交替计时，真实环境每步只推进一次。每个工作进程完成Actor、安全QP和动力学编译预热后再进入稳定计时。

| 实现和调用类型 | 样本数 | 中位数 / ms | p95 / ms | p99 / ms | 最大值 / ms | 超过200 ms |
|---|---:|---:|---:|---:|---:|---:|''')
    names = {'serial':'串行','parallel':'五进程并行'}
    phases = {'planning':'完整规划','feedback':'普通反馈','all':'全部控制'}
    for r in runtime['groups']:
        sections.append(f"| {names[r['engine']]}·{phases[r['phase']]} | {r['samples']} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | {r['max_ms']:.2f} | {r['deadline_misses']} |")
    sections.append(f'''
Actor加载与编译为{runtime['policy_initialization_ms']/1000.:.2f}秒；串行首次规划{runtime['cold'][0]['first_plan_ms']/1000.:.2f}秒，五进程启动、全部工作进程预热和首次规划{runtime['cold'][1]['first_plan_ms']/1000.:.2f}秒。运行设备为AMD Ryzen 9 7940HX、Windows 11，每个候选进程单线程。部署前需要完成预热。本结论针对数值仿真中的控制计算，本轮没有新增VRX或实艇结果。

![最终实现的规划时延]({BASE.as_posix()}/confirmation/figures/fig_robust_runtime.png)
''')


def append_audit(sections,result,spec,execution,development):
    sections.append('''
## 开发与验证依据

第一阶段仅用32个旧六艇开发场景及两个已知失败场景，比较执行间隔、后续策略和捕获裕度的固定配置。第二阶段使用此前两批已经观察过的全部256个六艇场景，比较8种方法；这些数据只用于开发和选择，不计入本轮独立确认。

| 第二阶段开发配置 | 成功 / 256 | 捕获 | 超时成功 | 封顶耗时 / 秒 |
|---|---:|---:|---:|---:|''')
    names = dict(cbf='CBF',best_fixed='固定守卫',base_h10='2秒重选＋CBF后续',
        sustain_h5='1秒重选＋持续候选后续（选定）',sustain_h10='2秒重选＋持续候选后续',
        short_h5='1秒短时消融',short_h10='2秒短时消融',legacy_base_h10='旧评分＋CBF后续')
    for r in development['summary']:
        if r['group']=='all':
            sections.append(f"| {names[r['method']]} | {r['success']} | {r['capture']} | {r['timeout']} | {r['capped_capture_time']:.3f} |")
    sections.append(f'''
最终选择按成功数最多、碰撞数最少、平均封顶耗时最低依次排序。512个全新场景与已有{spec['historical_unique_scenes']}个场景不重叠；场景种子为193160000＋机动档索引×10000＋重复编号，噪声种子加1000000。每个场景内随机打乱方法顺序。统计单位为场景，四档机动系数分层自举。

本轮完整执行{execution['rows']}个方法回合；512次原始ARBoids运行均另用直接源码循环核对完整轨迹哈希。执行前后检查冻结源代码、Actor检查点、算法、参数、分析程序和协议哈希。执行核验为`passed=true`，最终主判据为`primary_strong_advantage={str(result['primary_strong_advantage']).lower()}`，时延要求为`meets_runtime={str(result['meets_runtime']).lower()}`。

模型加速检查包括2–6艇的物理状态、观测和Boids输出，以及2艇和6艇全部五个候选的闭环预测。随机推力积分测试的状态容差为10⁻⁸、观测容差10⁻⁷；闭环前段状态容差10⁻⁶、推力容差0.005 N、评分容差0.001秒。批量Actor与逐艇参考的输出容差为3×10⁻⁵。既有失败排序回归测试和零后续消融检查均通过。535次选中计划和2605次实际控制的并行一致性另行检查，要求逐元素相同。

沿用三艇训练的冻结Actor检查点`main-seed42-20260921/adares1.pth`，未训练新网络；权重是本地按原始配置训练的权重。源码提交、任务判定和物理尺度详见后文原始实验设置；本轮主要外推范围是六艇和上述四档机动系数。

## 本轮文件与复现

- [最终算法](train/jit_rollout.py)、[连续反馈与评分](train/array_rollout.py)、[编译动力学](train/jit_nominal_environment.py)、[五进程执行](train/parallel_jit_rollout.py)。
- [预定协议](docs/robust-predictive-interception-study.md)、[冻结配置与哈希]({BASE.as_posix()}/confirmation/specification.json)。
- [全部回合]({BASE.as_posix()}/confirmation/episodes.csv)、[结果汇总]({BASE.as_posix()}/confirmation/outcome_summary.csv)、[配对比较]({BASE.as_posix()}/confirmation/paired_comparisons.csv)、[逐格结果]({BASE.as_posix()}/confirmation/cell_summary.csv)、[最终判据]({BASE.as_posix()}/confirmation/analysis.json)。
- [独立时延汇总]({BASE.as_posix()}/runtime-sustain_h5.json)、[每次调用计时]({BASE.as_posix()}/runtime-sustain_h5.csv)、[开发数据]({BASE.as_posix()}/development2/episodes.csv)。

以下命令在仓库根目录使用项目Python环境执行。确认运行器拒绝覆盖已完成的数据；绘图及报告更新读取已有结果。

```powershell
& 'D:/ARBoids/.venv/Scripts/python.exe' -X utf8 train/benchmark_robust_prediction.py --method sustain_h5
& 'D:/ARBoids/.venv/Scripts/python.exe' -X utf8 train/evaluate_robust_prediction.py --stage freeze
& 'D:/ARBoids/.venv/Scripts/python.exe' -X utf8 train/evaluate_robust_prediction.py --stage confirm --workers 8
& 'D:/ARBoids/.venv/Scripts/python.exe' -X utf8 train/figures/gen_fig_robust_prediction.py
& 'D:/ARBoids/.venv/Scripts/python.exe' -X utf8 train/figures/write_robust_report.py
```

本轮三组图件均同时保存300 dpi PNG与矢量PDF。
''')


if __name__ == '__main__':
    main()
