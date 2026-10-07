"""Update results.md from the completed fourth confirmation, retaining prior results."""
import csv
import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
BASE = Path('train/experiments/adaptive-prediction-20261006')
ROOT = REPO/BASE/'confirmation'
NAMES = dict(original='原始 ARBoids', cbf='CBF', predictive='自适应间隔长时预测',
             short_value='同规则自适应短时消融', strongest_short='较强的固定 2 秒短时方案',
             best_fixed='固定守卫策略', previous='第三轮固定 1 秒长时预测')
HISTORY = '## 前三轮记录：原始确认结果'


def read_csv(path):
    with path.open(encoding='utf-8', newline='') as file:
        return list(csv.DictReader(file))


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    result = load(ROOT/'analysis.json')
    if not result['passed']:
        raise RuntimeError('Analysis has not completed.')
    table = read_csv(ROOT/'outcome_summary.csv')
    index = {r['method']:r for r in table}
    comparison = {r['reference']:r for r in read_csv(ROOT/'paired_comparisons.csv')}
    cells = read_csv(ROOT/'cell_summary.csv')
    episodes = read_csv(ROOT/'episodes.csv')
    runtime = load(REPO/BASE/'runtime-adaptive.json')
    development = load(REPO/BASE/'development/summary.json')
    spec, execution = load(ROOT/'specification.json'), load(ROOT/'execution.json')
    row = index['predictive']
    n = int(row['episodes'])
    planning = result['runtime_planning']
    status = '全部四项主要对照判据及计算时延要求均已通过。' if result['all_requirements_met'] else '本轮尚未全部通过预定要求。'
    original = (REPO/'results.md').read_text(encoding='utf-8')
    body, bibliography = original.rsplit('## 参考文献与软件', 1)
    if HISTORY in body:
        historical = body.split(HISTORY, 1)[1]
    else:
        historical = '## 第三轮当时的结果与解释\n\n'+body.split('## 最新结果', 1)[1]
        historical = '\n'.join('#'+line if line.startswith('#') else line for line in historical.splitlines())
        historical = '\n\n以下记录属于各轮当时冻结的实现和独立场景，保留原判定，不作为第四轮的独立样本。\n\n'+historical
    sections = [f'''# 预测拦截：成功率、任务效率与计算代价

更新日期：2026-10-06。主要结论限于原始数值环境中的六艇防御，未新增VRX或实艇实验。

## 最新结果

**{status}** 最终冻结版本在{n}个全新六艇场景中，成功 **{row['success']}/{n}**，捕获 **{row['capture']}/{n}**，碰撞 **{row['collision']}**，平均未捕获封顶耗时 **{float(row['capped_capture_time']):.2f}秒**。相对CBF耗时降低 **{float(comparison['cbf']['improvement_percent']):.2f}%**；相对同规则短时消融降低 **{float(comparison['short_value']['improvement_percent']):.2f}%**；相对较强固定2秒短时方案降低 **{float(comparison['strongest_short']['improvement_percent']):.2f}%**。

同一最终版本的五进程并行实现，完整规划调用 **p95 {planning['p95_ms']:.1f}毫秒、p99 {planning['p99_ms']:.1f}毫秒、最大{planning['max_ms']:.1f}毫秒**。32个完整旧场景的{runtime['exact_steps']}个控制输出和{runtime['exact_plans']}个选中计划均与串行逐元素相同。200毫秒要求按完整规划调用判定，启动预热另计。

前两轮的主判据未通过；第三轮已明确胜过同参数1秒短时消融并满足计算预算，但对较强2秒短时方案只提高9.39%，未达到全部对照至少10%的标准。前三轮原始结果保留在后文。本轮完成修复和时延验证后，冻结代码、参数、分析程序及新场景名单，再一次性运行独立确认。

## 方法与贡献

每次规划从五个整队反馈候选中选择一个：CBF基线、Actor端点、Boids端点、目标外推导航、单艇守卫导航。每0.2秒根据新观测计算反馈，实际推力全部通过原有官方CBFpy安全过滤器。候选推演使用公开观测及因果运动估计；不读取真实机动参数或未来随机数。

长时价值由执行区间和最多20秒的后续推演组成，后续持续执行同一个反馈候选。系统到下一次规划时重新选择，后续推演只是该候选持续执行的价值估计。旧方案在后续切回CBF，可能低估持续机动。所有候选均预测失守时，修正后的评分在相同失败类别内优先推迟失败；此修正也应用于短时对照。

第四轮只新增一个间隔规则：每次开始规划时，若根据历史观测估计的机动系数小于1.75，执行区间为2秒，否则为1秒。区间内继续0.2秒反馈，不中途改变区间长度；初始估计2.25，因此首次仍为1秒。阈值取原两档机动系数的中点，未搜索其他阈值。其依据是前两批旧数据中低机动场景在2秒重选时均更快。匹配短时消融使用完全相同的估计和间隔规则，仅移除20秒后续推演。

固定2秒短时方案是早先开发数据中较强的短时对照，继续作为主要对照；第三轮固定1秒长时版本作为次级版本对照。网络权重、真实环境、噪声、捕获和失守判定均未更改。预测捕获半径保留0.5米裕度，偏离CBF基线的候选保留0.15秒代价。

速度优化使用批量Actor、数组观测与Boids、JIT双精度动力学积分，以及五候选并行推演。与源实现的数值误差已单独核验；同一加速算法的串行和并行输出要求完全相同。

## 第四轮：全新六艇场景

机动系数1.5、2、2.5、3各{n//4}场。原任务把捕获和坚持至80秒都算作成功，表中分列捕获与超时。效率指标把所有未捕获回合计为80秒，含超时成功。

| 方法 | 成功 / {n} | 捕获 | 碰撞 | 源码失守 | 超时成功 | 封顶耗时 / 秒 |
|---|---:|---:|---:|---:|---:|---:|''']
    for r in table:
        sections.append(f"| {NAMES[r['method']]} | {r['success']} | {r['capture']} | {r['collision']} | {r['source_loss']} | {r['timeout']} | {float(r['capped_capture_time']):.3f} |")
    sections.append('''
| 最终版本的主要对照 | 平均耗时降低 | 配对耗时差 / 秒，98.75% CI | 成功率差 / 百分点 | 成功率差保守下界 / 百分点 | 全部判据 |
|---|---:|---|---:|---:|---|''')
    for method in result['primary_references']:
        r = comparison[method]
        decision = '通过' if r['meets_strong_criteria']=='True' else '未通过'
        sections.append(f"| {NAMES[method]} | {float(r['improvement_percent']):.2f}% | {float(r['delta_capture_seconds']):+.3f} [{float(r['delta_ci_low']):+.3f}, {float(r['delta_ci_high']):+.3f}] | {float(r['delta_success_pp']):+.3f} | {float(r['success_conservative_lower_pp']):+.3f} | {decision} |")
    previous = comparison['previous']
    sections.append(f'''
耗时差为“最终方法−对照”，负值更好。预定判据要求对四项主要对照全部同时达到：平均封顶耗时至少降低10%，配对分层10000次自举区间排除零改善，观察成功数不少且碰撞数不多，配对成功率差保守下界不低于−3个百分点。本轮使用98.75%区间（单轮α=0.05/4），前三轮按原协议作为历史记录，不能把各轮区间混合解释为一个统一的总体错误率保证。

相对第三轮固定1秒长时版本，次级比较的耗时差为{float(previous['delta_capture_seconds']):+.3f}秒，98.75% CI [{float(previous['delta_ci_low']):+.3f}, {float(previous['delta_ci_high']):+.3f}]，耗时相对变化为{-float(previous['improvement_percent']):+.2f}%（正值表示更慢）。本轮结果不支持自适应间隔带来额外收益；主要贡献的证据来自长时反馈推演与匹配短时消融的比较。冻结主方案及其全部预定判据保持原样报告。

![第四轮成功与效率]({BASE.as_posix()}/confirmation/figures/fig_adaptive_primary.png)

![第四轮配对耗时]({BASE.as_posix()}/confirmation/figures/fig_adaptive_paired.png)

| 机动系数 | CBF成功 / {n//4} | 最终版成功 / {n//4} | 最终版捕获 / {n//4} | CBF耗时 / 秒 | 最终版 / 秒 | 较强2秒短时 / 秒 |
|---|---:|---:|---:|---:|---:|---:|''')
    for cell in sorted({r['cell'] for r in cells}):
        group = {r['method']:r for r in cells if r['cell']==cell}
        a, b, c = group['predictive'], group['cbf'], group['strongest_short']
        sections.append(f"| {cell.split('-a')[1]} | {b['success']} | {a['success']} | {a['capture']} | {float(b['capped_capture_time']):.3f} | {float(a['capped_capture_time']):.3f} | {float(c['capped_capture_time']):.3f} |")
    plans = {h:sum(int(r[f'{name}_second_plans']) for r in episodes if r['method']=='predictive')
             for h, name in ((1,'one'), (2,'two'))}
    sections.append(f"\n最终方法共执行{plans[1]}次1秒间隔规划和{plans[2]}次2秒间隔规划；间隔完全由运行时估计决定。\n")
    append_runtime(sections, runtime)
    append_audit(sections, result, spec, execution, development)
    sections.extend(['\n'+HISTORY, historical, '\n## 参考文献与软件\n', bibliography])
    (REPO/'results.md').write_text('\n'.join(sections), encoding='utf-8')
    print(json.dumps(dict(passed=True,report=str(REPO/'results.md'),
                         all_requirements_met=result['all_requirements_met']),ensure_ascii=False))


def append_runtime(sections, runtime):
    sections.append('''
## 最终版本的独立时延

其他实验工作进程退出后，用32个旧六艇场景运行完整回合，每档机动系数8场。计时包含观测适配、全部候选推演、安全过滤、选择和动作映射，不含真实环境积分或外部通信。串行与并行交替先后计时，每步只推进一次真实环境。每个工作进程完成预热后才开始稳定运行计时。

| 实现与调用 | 样本数 | 中位数 / ms | p95 / ms | p99 / ms | 最大值 / ms | 超过200 ms |
|---|---:|---:|---:|---:|---:|---:|''')
    names = dict(serial='串行', parallel='五进程并行')
    phases = dict(all='全部控制', planning='完整规划', feedback='普通反馈')
    for r in runtime['groups']:
        sections.append(f"| {names[r['engine']]}·{phases[r['phase']]} | {r['samples']} | {r['median_ms']:.2f} | {r['p95_ms']:.2f} | {r['p99_ms']:.2f} | {r['max_ms']:.2f} | {r['deadline_misses']} |")
    sections.append(f'''
Actor初始化{runtime['policy_initialization_ms']/1000.:.2f}秒；串行首次规划{runtime['cold'][0]['first_plan_ms']/1000.:.2f}秒；五进程启动、全部进程预热和首次规划{runtime['cold'][1]['first_plan_ms']/1000.:.2f}秒。稳定运行预算需要在部署前完成预热。设备为AMD Ryzen 9 7940HX、Windows 11，每候选进程单线程。

共{runtime['exact_steps']}次实际控制输出、{runtime['exact_plans']}个选中计划逐元素一致，首次全部五个候选推演也逐元素一致。该基准验证同一最终算法的并行加速与计算时间，不构成通信、传感或执行器在内的端到端实时保证。

![第四轮最终版本时延]({BASE.as_posix()}/confirmation/figures/fig_adaptive_runtime.png)
''')


def append_audit(sections, result, spec, execution, development):
    sections.append('''
## 开发与验证依据

第四轮开发使用已经观察过的768个六艇场景（128＋128＋512）；新增两种自适应方法实际运行1536个回合，对照复用冻结的3072个回合并记录输入哈希。下表全部是开发结果，不计入独立确认。

| 开发方法 | 成功 / 768 | 捕获 | 碰撞 | 超时成功 | 封顶耗时 / 秒 |
|---|---:|---:|---:|---:|---:|''')
    for r in development['summary']:
        if r['group']=='all':
            sections.append(f"| {NAMES[r['method']]} | {r['success']} | {r['capture']} | {r['collision']} | {r['timeout']} | {r['capped_capture_time']:.3f} |")
    sections.append(f'''
本轮1024个确认场景与既有{spec['historical_unique_scenes']}个唯一场景不重叠，种子为203160000＋机动档索引×10000＋重复编号，噪声种子加1000000；每场随机打乱七种方法的运行顺序。确认完整执行{execution['rows']}个方法回合，不依中间结果调参、筛选场景或提前停止。每次原始ARBoids运行均另用直接源码循环核对完整轨迹哈希；执行前后核对算法、协议、分析、环境、权重和开发依据哈希。

执行核验为`passed={str(execution['passed']).lower()}`，最终主判据为`primary_strong_advantage={str(result['primary_strong_advantage']).lower()}`，时延判据为`meets_runtime={str(result['meets_runtime']).lower()}`，全部要求为`all_requirements_met={str(result['all_requirements_met']).lower()}`。

间隔规则的回归检查覆盖：退化为固定1秒时与第三轮逐步控制一致、运动观测每步只更新一次、未执行完的区间不受估计变化打断。模型加速已通过2–6艇动力学和观测数值检查，以及2艇、6艇全部候选闭环预测比较；状态、观测、推力和评分容差见第三轮记录。最终版本完整回合的串行/并行相同性由独立时延基准再次验证。

沿用本地按原始配置训练的三艇Actor检查点`main-seed42-20260921/adares1.pth`，本轮未训练新网络。原始源码版本、物理尺度和任务判定见历史设置。

## 文件与复现

- [最终自适应控制器](train/adaptive_interval_rollout.py)、[并行控制器](train/parallel_adaptive_prediction.py)、[长时反馈推演](train/jit_rollout.py)、[数组评分](train/array_rollout.py)、[编译动力学](train/jit_nominal_environment.py)。
- [第四轮预定协议](docs/adaptive-prediction-study.md)、[冻结配置与哈希]({BASE.as_posix()}/confirmation/specification.json)。
- [全部回合]({BASE.as_posix()}/confirmation/episodes.csv)、[汇总]({BASE.as_posix()}/confirmation/outcome_summary.csv)、[配对比较]({BASE.as_posix()}/confirmation/paired_comparisons.csv)、[逐格结果]({BASE.as_posix()}/confirmation/cell_summary.csv)、[判据]({BASE.as_posix()}/confirmation/analysis.json)。
- [独立时延汇总]({BASE.as_posix()}/runtime-adaptive.json)、[逐调用计时]({BASE.as_posix()}/runtime-adaptive.csv)、[开发数据]({BASE.as_posix()}/development/episodes.csv)。

在仓库根目录使用项目Python环境执行；数据运行器拒绝覆盖已经完成的数据，绘图和报告读取现有结果。

```powershell
& 'D:/ARBoids/.venv/Scripts/python.exe' -X utf8 train/develop_adaptive_prediction.py --workers 8
& 'D:/ARBoids/.venv/Scripts/python.exe' -X utf8 train/benchmark_adaptive_prediction.py
& 'D:/ARBoids/.venv/Scripts/python.exe' -X utf8 train/evaluate_adaptive_prediction.py --stage freeze
& 'D:/ARBoids/.venv/Scripts/python.exe' -X utf8 train/evaluate_adaptive_prediction.py --stage confirm --workers 8
& 'D:/ARBoids/.venv/Scripts/python.exe' -X utf8 train/figures/gen_fig_adaptive_prediction.py
& 'D:/ARBoids/.venv/Scripts/python.exe' -X utf8 train/figures/write_adaptive_report.py
```

三组图均保存300 dpi PNG及矢量PDF。
''')


if __name__=='__main__':
    main()
