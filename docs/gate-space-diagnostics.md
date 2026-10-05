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
