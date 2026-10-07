# 融合权重空间诊断

这一入口完成第一阶段开发：完整仿真状态恢复、固定候选动作的权重搜索、配对闭环验证。使用冻结的原始 ARBoids Actor/Adapter，不训练网络，也不调用现有 MAPPO 门控或自适应碰撞乘子。

## 实验定义

- 环境：`paper-parameters-v1`，默认三艘防守艇、60 秒任务时限、0.2 秒控制周期，APF 攻击者。奖励直接使用环境返回的任务、编队、碰撞三项之和，再对艇求平均。
- 候选：在采样时刻保存原策略的学习动作、原门控及环境的 Boids 推力。保留动作的原始浮点精度。
- 干预：只改变当前一个控制周期的门控。下一周期恢复冻结的原策略，重新计算候选动作和门控。攻击者每步根据该分支局势重新计算 APF，保留 APF 噪声和海流扰动。
- 网格：默认每艇 `[0, 0.25, 0.5, 0.75, 1]`，三艇有 125 个组合，另加原始整组权重；原权重已在网格中时去重。候选 0 始终是原始权重。
- 预测：零海流名义三自由度模型，保持当前融合推力 2 秒，积分步长 0.05 秒。风险取未来 `k=1..H` 所有艇对的最大安全距离侵入平方；另报当前距离。默认预警距离 7 米，实际碰撞半径沿用环境的 5 米。2 秒和 7 米是开发初值。
- 实际距离：从干预时刻起记录每个 0.2 秒控制端点的最小艇间距，包含起点；短时距离在预测窗口或终止时刻截止。它与保持指令的预测不是同一实验，也不代表连续时间安全性。

## 状态与配对

`SimulationSnapshot` 深复制整个 `TADEnv`，包括防守艇/攻击艇的内部速度、海流、推力、Boids 缓存、时间、距离缓存和历史轨迹；同时保存 NumPy、Python、Torch CPU 和已初始化 CUDA 的随机数状态。每次恢复返回独立环境。

`future_seed=None` 恢复原随机数状态。搜索前会续跑原动作，并核对完整轨迹摘要是否与采集回合一致。指定未来种子时，物理状态不变，只更换后续扰动；同一轮所有候选使用同一个未来种子。分支运行结束恢复调用者的随机数状态。

采集完成指定数量的源回合后，分别在碰撞回合和普通成功回合中做储水池抽样。碰撞回合保留指定提前量的状态；不足该提前量的状态跳过，不夹到初始时刻。成功回合在全部决策时刻中均匀抽取一个状态。抽样生成器与仿真随机数独立，配额不会提前停止源回合采集。

## 运行

在仓库根目录执行。以下是有明确数量上限的开发示例；种子应使用自己的开发集合。先小规模检查执行时间，再扩大源回合及独立扰动次数。

```powershell
$python = 'D:\ARBoids\.venv\Scripts\python.exe'
$baseline = 'D:\ARBoids\train\experiments\paper-parameters-seed42-20260921\adares1.pth'
$env:MPLBACKEND = 'Agg'

& $python -X utf8 -u train/diagnose_gate_space.py collect --checkpoint $baseline --output-dir train/experiments/gate-space-dev/capture --first-seed 97000000 --episodes 100 --collision-episodes 5 --normal-episodes 5 --lookbacks 1 2 3 --sampling-seed 42

& $python -X utf8 -u train/diagnose_gate_space.py search --cases train/experiments/gate-space-dev/capture/cases.pkl --output-dir train/experiments/gate-space-dev/search --continuation-seed 98000000 --search-repeats 4 --validation-repeats 8
```

采集阶段输出 `sources.csv`（所有源回合的真实事件）、`cases.pkl`（完整状态及固定候选）和 `summary.json`（采样条件、选中状态及检查点/源码摘要）。没有选中状态时 `diagnosis_ready=false`；这与程序执行是否通过分别记录。

搜索阶段输出：

| 文件 | 内容 |
|---|---|
| `branches.csv` | 每个状态、候选、未来种子和阶段的预测风险、实际成功/碰撞/突破、团队回报、距离及相对同种子原动作的差值 |
| `cases.csv` | 每个状态选中的门控、搜索和独立扰动验证结果、原轨迹重放是否一致 |
| `summary.json` | 固定实验设置；按普通/碰撞状态分组、先对同源回合的状态求平均再对源回合求平均的配对差值 |

目录已有输出时拒绝覆盖。搜索可通过 `--checkpoint` 指向内容完全相同但位置变化的权重。完整快照是 Python pickle，只读取本入口生成的可信本地文件；搜索要求采集与诊断使用相同源码。修改核心代码后重新采集。

## 怎样选权重与解释结果

先在搜索扰动集合上筛选：观察到的成功率不低于原动作，突破率不高于原动作。在合格组合中优先降低碰撞率，再按成功、突破、回报、短时距离、修正幅度打破平局；原权重始终可选。这是固定的开发样本筛选规则，不是统计非劣检验。

选好整组权重后，只在另一组不重叠的未来种子上比较该权重和原动作，禁止用验证结果再次选权重。`validation_observed_improvement` 表示该状态在这些新扰动中碰撞减少，且成功/突破未向不利方向变化。没有改善也原样输出。

这是同一状态下的离线机会诊断。独立未来扰动不等于独立的新初始场景，最佳网格动作也不是可部署策略。同一次碰撞前的多个状态保留共同的 `source_seed`，不能计作独立重复。有限网格中没有改善，不证明连续空间中没有改善。

## 验证

```powershell
& 'D:\ARBoids\.venv\Scripts\python.exe' -X utf8 -m unittest discover -s train/tests -p test_gate_diagnostics.py -v
```

测试覆盖完整状态与随机数恢复、序列化往返、零修正轨迹一致、单周期干预、相同海流配对、攻击者使用各分支新局势、原动作推力精度、源轨迹复现、独立扰动验证和按源回合汇总。

## 扩展诊断：介入时机、预测一致性与连续修正

`diagnose_gate_followup.py` 从已记录的 `sources.csv` 重新运行全部源回合，逐一核对完整轨迹摘要，然后采集碰撞前 1、2、3 秒的状态及每个成功回合的一个普通状态。它完成以下三组对照：

- **单次门控搜索**：每个状态搜索 125 个网格组合及原权重，使用 2 组搜索扰动选权，另用 8 组扰动复测。选权规则沿用上一节。
- **预测与执行一致性**：在 1、2、3 秒窗口内，分别对原权重和该窗口预测间距最大的组合施加固定物理推力。先用零扰动、与预测器相同的初始速度核对动力学实现，再加入实际扰动。轨迹及首次越过碰撞阈值的比较只使用双方共有的 0.2 秒端点，并在仿真终止时截断。另对全部组合运行“只修正一个周期，随后恢复原策略”的零扰动反馈预测，避免把持续推力预测当作单次修正的预测。
- **连续修正**：每周期重新观测、生成学习/Boids 候选、预测并选权。2 秒预测窗口下比较启用 1、5、10 个控制周期；连续启用 10 周期时再比较 1、2、3 秒预测窗口。有限启用时长从采集状态开始计时，不从第一次预警开始计时。

连续修正使用固定的非学习规则。若原权重的预测最小间距不小于 7 米，则精确保留原权重；否则在同一网格及原权重中最小化：

\[
\left[\max\left(0,\frac{7-d_{\min}}{7}\right)\right]^2
+0.02\,\operatorname{mean}\left[\left(\frac{u-u_0}{u_{\max}-u_{\min}}\right)^2\right].
\]

其中第二项是归一化的物理推力改变量。每次预测假设该推力在窗口中保持不变，但实际只执行下一周期，随后重新计算。与事后读取完整分支结局的离线搜索相比，这条规则只使用当前可获得的状态；两者的单次修正结果也应分别报告。

```powershell
& $python -X utf8 -u train/diagnose_gate_followup.py --checkpoint $baseline --source-episodes train/experiments/gate-space-dev/capture/sources.csv --output-dir train/experiments/gate-space-followup --workers 4 --search-repeats 2 --validation-repeats 8 --audit-repeats 4

& $python -X utf8 -u train/evaluate_rolling_gate.py --cases train/experiments/gate-space-followup/cases.pkl --output-dir train/experiments/gate-space-followup/persistent --workers 4
```

第二条命令补测持续启用到回合结束的规则，既从采集状态开始，也从全部源场景的初始状态开始。后者依靠在线预测触发，无需知道未来碰撞时刻。它复用同一冻结模型、场景和 8 组配对扰动；这仍是开发集内的对照。第一次运行也可添加 `--entire-episode`，直接包含采集状态上的持续启用设置。

扩展诊断将扰动序列对齐到绝对控制步：同一源场景、同一未来种子在相同任务时刻使用相同的外生随机抽样。不同提前量之前的物理历史仍是各自保存的原始历史，因此它们不是同一条新扰动轨迹的不同截面。起点实验和采集状态实验分开汇总；同源状态及其未来扰动不作独立场景计数。

第一条命令输出 `timing_cases.csv`、`timing_branches.csv`、`model_audits.csv`、`pulse_forecasts.csv`、`rolling_branches.csv` 和 `rolling_summary.csv`。第二条命令输出 `branches.csv` 与 `summary.csv`。两个入口都保存实验设置和最终 `summary.json`，校验运行期间源码及检查点未变化，并拒绝覆盖已有目录。`passed=true` 表示执行与一致性检查通过，不表示策略达到部署要求。

```powershell
& $python -X utf8 -m unittest discover -s train/tests -p 'test_gate*.py' -v
& $python -X utf8 -m unittest discover -s train/tests -p test_paper_protocol.py -v
```

扩展测试还覆盖固定物理推力、匹配控制的预测轨迹、单次修正后恢复反馈、有限启用窗口停止、无预警时保留原权重、共同前缀比较及绝对时间扰动配对。

## 论文统一评测与图表

`evaluate_gate_paper.py` 从不同场景的起点运行完整回合，默认 128 个场景。8 个对照包括原始 ARBoids、仅 Boids、仅学习动作、固定 0.5 门控、首次预警单周期修正（3 秒预测），以及 1、2、3 秒预测的持续修正。场景种子从 `108000000` 开始，未来扰动种子从 `118000000` 开始；各方法逐场景配对，完成后重复基线并核对整个轨迹。

`evaluate_gate_ablation.py` 复用同一组场景与扰动，增加 2 秒预测的首次预警单周期修正，以便与 2 秒持续修正直接比较。单次规则会等待首个预测预警，执行一个控制周期后恢复原门控；固定门控对照每周期重新生成候选动作，但保持门控系数不变。

```powershell
& $python -X utf8 -u train/evaluate_gate_paper.py --checkpoint $baseline --output-dir train/experiments/gate-paper-results --episodes 128 --workers 4

& $python -X utf8 -u train/evaluate_gate_ablation.py --source-run train/experiments/gate-paper-results --output-dir train/experiments/gate-paper-results/single_h2 --workers 4

& $python -X utf8 train/figures/gen_fig_gate_paper.py --run-dir train/experiments/gate-paper-results --diagnostic-dir train/experiments/gate-space-followup-20261006
```

图表入口读取已完成的配对评测，以及此前诊断目录中的 `control_timing.csv`，生成 `main_table.csv`、中英文 `paper_results.md` 和 3 幅 PDF/PNG 图件。成功率使用精确 McNemar 检验，并对相对原门控的 8 项成功率比较做 Holm 校正；配对效果区间按场景进行 20,000 次自助重采样，成功率图使用 Wilson 区间。轨迹图选取碰撞恢复回合中回报增益居中的示例，并再次验证两条轨迹。

2026-10-06 的 128 场景结果中，2 秒持续修正为 127/128 成功、0 碰撞、1 突破；原始 ARBoids 为 115/128 成功、12 碰撞、1 突破。相同 2 秒窗口下的单次修正为 116/128 成功、11 碰撞、1 突破。当前开发推荐窗口据此调整为 2 秒；所有窗口结果保留在完整结果表中。
