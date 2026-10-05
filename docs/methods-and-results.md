# 已实施的方法改进与已获得结果

数据截至 **2026-10-05 22:48:50（北京时间）**。范围包括原论文基础上的实现与部署、预测增强 MAPPO 的全部已保存开发分支，以及本轮正式对照已经完成的结果。本文的成功、碰撞、突破均来自实际回合事件；训练预测风险不当作实验结局。

当前最明确的成果是：本研究实现与部署的 ARBoids 在本轮 100 个 VRX 码头场景中取得 **100% 成功、0% 碰撞、0% 突破**；历史冻结的预测增强 MAPPO 在 6,000 个独立 2D 场景中取得 **98.50% 成功、0.517% 碰撞、0.983% 突破**，相对同场景 ARBoids 明显减少碰撞，但增加突破。新正式训练的种子 101 完整模型为 **98.60% / 0.90% / 0.50%**；其他训练种子尚未齐全。

## 1. 方法与结果的归属

| 名称 | 已实际做了什么 | 本文如何使用 |
|---|---|---|
| 原论文 ARBoids | Boids 与学习动作自适应融合，以原 SAC 训练；作者报告 VRX 85% 成功、11% 碰撞、4% 突破 | 文献报告值，单列引用 |
| ARBoids（本研究实现与部署） | 本研究独立训练；任务、奖励、参数按论文对齐；完成 VRX 控制、推力映射及反馈同步适配 | 属于本研究获得的结果；代码中的 `baseline` 是内部方法标识 |
| 预测增强 MAPPO（历史冻结模型） | 在 seed-42 ARBoids Actor 上初始化，经过预测、联合动作与学习增益等开发；最终冻结第 265 次更新 | 6,000 场景独立复测、冻结模型泛化与 VRX 的受测模型 |
| 预测增强 MAPPO（本轮正式模型） | 种子 101/202/303/404/505 各自预训练，再按同一冻结配方续训和消融 | 评估学习机制贡献及训练种子稳定性；目前主要消融仅完成种子 101 |

ARBoids 版本的 SAC、回放池及 Actor/Critic 核心保留自作者公开实现。已做源码比对确认，新增工作主要在协议、训练入口、运行与部署链路；该版本未启用 MAPPO、联合预测或学习增益。因此，可以把它的实测性能列为本研究结果，但不能把这部分性能归因给尚未启用的新机制。论文未提供可重建其原 100 次试验的完整初值，本文与论文的数字比较属于跨试验参考。

主要依据：[论文参数对齐说明](paper-parameters.md)、[预测增强 MAPPO 设计](predictive-mappo.md)、[正式实验方案](formal-evidence.md)。原论文：[ARBoids, IEEE RA-L, DOI: 10.1109/LRA.2026.3662620](https://doi.org/10.1109/LRA.2026.3662620)，VRX 结果见印刷页 3642。

## 2. 已实施的改进设计

### 2.1 ARBoids 的实现、训练与部署

| 改进 | 具体实现及目的 | 状态与证据 |
|---|---|---|
| 任务时限和课程对齐 | 训练时限由发布代码的 80 秒调整为论文的 60 秒；课程灵活度扰动半宽由 0.25 调整为 0.5 | 已实现并完成 seed-42 独立训练 |
| 终止条件与边界对齐 | 攻击艇须实际进入目标区域才判突破；去掉“比所有防御艇更靠近目标即获胜”的额外条件；对齐目标、碰撞与捕获边界 | 已用于后续论文参数协议 |
| 奖励重构 | 按论文叠加任务、编队、碰撞分量；编队奖励去除双倍计算并覆盖终止转移；涉碰艇每次转移处罚 -50；守时成功不再冒用捕获奖励 | 已实现；后续真实事件判定与奖励分开记录 |
| Adapter 探索修正 | 由共享均匀噪声改为独立高斯噪声，标准差 0.1 | 已用于论文参数训练 |
| 可恢复的训练和评估 | 保存模型、优化器、随机状态及配置；分离训练、确定性评估、VRX；保存真实终止事件和模型哈希 | 已运行；不把基础设施异常计为任务结局 |
| VRX 推进器和航向适配 | 对齐 ENU 航向与左右推进器映射：策略动作 0 对应右舷、动作 1 对应左舷 | 已用于两类控制器部署 |
| VRX 反馈同步 | 根据反馈估计航向角速度，将各艇状态外推到共同时间戳；训练、评估、VRX 复用策略入口 | 本轮两种方法采用相同同步处理 |
| 真实仿真闭环 | ROS 2/Gazebo 中记录轨迹、控制耗时、反馈时间差；控制计算期间仿真时钟继续推进 | 已完成本轮 600 个 VRX 回合 |

保留了作者的 Boids、APF、动力学和 18 维观测实现。论文未明确的观测噪声未自行补造；这些实现差异与缺失信息见参数说明。上述改进作为一个整体得到现有结果，尚未逐项消融出各自带来的增益。

### 2.2 预测增强 MAPPO 的核心结构

执行链为：

`局部观测 → Boids/学习候选 → 同步交换候选和状态 → 名义轨迹预测 → 艇对关系与联合动作选择 → 学习门控 → 融合推力`

最终推进器推力仍为 `F = theta × F_learning + (1 − theta) × F_Boids`；归一化动作按 `F = 750a + 250` 换算，预测器和执行器使用同一规则。

| 编号 | 已实现设计 | 解决的问题与实现要点 | 当前采用情况 |
|---|---|---|---|
| M1 | SAC Actor 到两阶段 MAPPO 的初始化 | 迁移状态编码、学习候选及原 Adapter；未加入预测修正时对齐原确定性行为，不迁移 SAC 的 Q 梯度、回放更新或温度优化 | 历史模型与正式配方均采用；启用预测修正后动作可发生变化 |
| M2 | 提案—门控两阶段随机策略 | 先生成学习动作，再在候选及预测关系条件下生成融合门控；所有防御艇共享 Actor，Critic 集中训练 | 已实现并训练 |
| M3 | 短时名义动力学预测 | 以实测地速初始化零海流三自由度模型，预测 2 秒、子步 0.05 秒；不读取未来扰动，不推进真实环境 | 当前采用 |
| M4 | 艇对关系编码 | BB/BL/LB/LL 四类端点候选组合的最近距离、最近时刻、接近趋势，加相对位姿速度，共 18 个有向特征；关系编码和 masked attention 汇总交互 | 当前采用 |
| M5 | 互补的成对协调信号 | 共享成对网络由正反方向关系生成协调强度，再送入 Adapter | 当前采用；互补信号本身不等于全队安全 |
| M6 | 预测风险直接影响门控 | 将两类候选的风险差作为门控修正；7 米名义提前响应距离，实际碰撞阈值仍为 5 米；安全区修正为零 | 从关系特征扩展为显式物理先验，已开发验证 |
| M7 | 队友临时融合意向 | 一轮消息中交换原始 Actor 观测，重算队友尚未风险修正的门控意向，替代两个端点各占一半的假设 | 当前采用；不使用 Critic 信息或最终门控样本 |
| M8 | 面向拦截分工的调整优先级 | 更靠近攻击艇的艇具有更高动作改动代价，同伴承担更多避碰调整；优先级温度默认 5 米 | 当前采用；仅表达当前几何优先级 |
| M9 | 实际混合动作轨迹预测 | 从仅预测两个端点扩展到 5 点、9 点融合网格；九点间隔 0.125 | 当前采用九点网格 |
| M10 | 全队联合组合选择 | 枚举每艇网格动作及原意向，以艇间风险与门控改变量选择全队组合 | 三艇 1,000 组合；四艇 10,000；五艇超上限 |
| M11 | 全局学习增益 | 风险/联合修正进入门控分布均值，由 PPO 学习增益，避免始终以固定强度干预 | 已实现；正式实验含固定增益对照 |
| M12 | 上下文学习增益 | 以 Adapter 特征学习乘法因子 `2·sigmoid(h(features))`，零初始化时因子为 1 | 历史最终阶段只训上下文头；正式完整组训练全部 Actor |
| M13 | 攻击艇轨迹及拦截机会预测 | 用当前攻击艇位置、地速作匀速外推，惩罚最佳拦截机会退化 | 已实现并做训练场景探针；历史冻结与正式配方未启用 |
| M14 | 终止感知预测 | 估算预计捕获时间，缩短捕获后不再影响任务的预测窗口，另加 0.4 秒余量 | 已实现并做探针；历史冻结与正式配方未启用 |

联合网格选出的修正经过学习增益、门控采样和裁剪后才执行，实际执行动作可能与被检查的网格组合不同；名义模型也存在海流、攻击运动及保持推力近似。因此，当前实现支持“经验上减少碰撞”的表述，尚不支持“实际执行动作有严格安全保证”。联合枚举同代价时按固定艇序选首项，也不能把所有退化情形宣称为严格置换等变。

代码入口：[策略](../train/policy/mappo.py)、[网络和联合选择](../train/policy/mappo_networks.py)、[预测器](../train/policy/interaction_prediction.py)、[VRX 控制适配](../vrx/mappo_controller.py)。

### 2.3 训练目标、概率计算与优化改进

下表同时列出主配方和开发可选分支；并非所有开关都在当前正式实验中启用。

| 编号 | 已实施改进 | 作用及当前证据 |
|---|---|---|
| T1 | 真实碰撞事件成本与集中价值估计 | 首次实际队内碰撞计团队成本 1；任务奖励与事件成本分别建价值头，先组合优势再统一标准化 |
| T2 | 拉格朗日约束与完整回合采样 | 一个 PPO 批次内固定乘子，结束后用正常起始分布的完整回合事件率更新；分支课程不混入预算统计 |
| T3 | 分因子 PPO 与全队联合 ratio | 实现两种概率规则；当前正式配方对每个控制时刻所有艇两阶段对数概率差求和，再作一次 PPO 裁剪 |
| T4 | 未折扣事件优势 | 成本 Monte Carlo 标签配合成本 GAE=1，使碰撞优势直接基于本回合真实结局；任务、碰撞折扣均为 1 |
| T5 | 突破约束 | 增加突破价值头和乘子，在原任务收益上同时约束碰撞和突破 |
| T6 | 纯事件目标分支 | 尝试以减少碰撞为主目标，并约束成功率和突破率；另做 99% 成功下限及恢复全 Actor 训练 |
| T7 | theta 空间门控及裁剪分布概率 | 对裁剪到 0/1 的门控使用端点概率质量，内部使用高斯密度；训练与执行概率对应 |
| T8 | 价值预热和奖励尺度处理 | 先拟合完整回合回报；任务价值内部缩放 100 倍，在优势计算时恢复原单位 |
| T9 | 预测参数独立优化设置 | 试验学习率、Adam epsilon、门控标准差学习率、探索范围和 PPO 数据复用轮数，保留历史优化器状态 |
| T10 | KL 提前停止与整批更新保护 | 更新后重算整批 KL 和 PPO surrogate，保留合格轮次；不合格时恢复 Actor 与完整 Adam 状态 |
| T11 | 完整梯度回溯 | 常规更新均被拒绝时，最多试八个减半步长；拒绝的试探不累计优化器状态 |
| T12 | 任务/成本 surrogate 联合保护 | 在正常训练回合上检查任务和负成本 surrogate，避免只优化加权和掩盖其中一项退化 |
| T13 | 风险前缀恢复课程 | 从真实失败轨迹回退，精确重放已执行前缀，采集独立续跑；扩大门控探索并保存真实探索尺度 |
| T14 | 早期突破恢复与独立续跑基线 | 在首次明显预测干预之前回退；用排除自身未来的其他续跑估计任务、碰撞及突破基线 |
| T15 | 先验 KL 校准与有界方向微调 | 新增风险分支时校准增益；另有标量增益、门控末层的有界方向/Fisher 预调节工具，只在训练数据上检查 |

T5 在冻结模型和正式配方中启用。T6 的事件目标已完成开发比较，未取代当前主配方。T13/T14 属于历史开发，当前正式实验取消恢复分支，使用正常完整回合。优化目标和 KL 的训练批改善均需由真实评估验证，不能直接替代成功率、碰撞率和突破率。

当前正式配方的碰撞训练预算为 **5%**、突破预算为 **0.25%**；用户期望的性能目标仍为 **成功率至少 99%、碰撞率至多 1%，并检查突破不恶化**。训练内部预算与最终验收目标是两个量，现有数据未达成全部目标；本轮保持冻结设置执行。

### 2.4 计算、通信和证据流程

| 已实施工作 | 实际作用 |
|---|---|
| CPU 多进程完整回合采样、GPU 单学习器更新 | 同批 rollout 使用同一不可变策略；正式任务每采样进程内批量处理 4 个环境 |
| 共享一次状态编码、向量化预测、动力学常量缓存 | 减少重复计算；保留原动力学和概率链路。确定性验收继续逐场景执行 |
| checkpoint、RNG 与失败前缀恢复 | 保存配置、Actor/Critic、优化器、乘子、随机状态；支撑续跑和实际动作重算 |
| 交替对抗训练入口与集成检查 | 保留原有防御/攻击交替训练入口及短程 smoke 配置；现有攻击者测试权重不作为训练充分的学习型对手 |
| 同场景配对、模型冻结与种子隔离 | 开发、独立验收、正式实验分开；已查看的验收数据转作开发后，不再宣称全新 |
| 五训练种子、同环境步数对照及消融 | 统一预训练来源和新增网络共有张量，分别检查额外训练、预测、联合选择和学习增益 |
| 泛化条件 | 灵活度、2/4 艇、直接攻击和扩大排斥范围 APF；五艇单列“不支持” |
| 通信扰动注入 | 测试 200/400 ms 延迟、10%/30% 丢包；可靠启动快照后使用缓存消息，无未来信息 |
| 服务器执行适配 | 在不改算法、模型哈希和预算的前提下迁移；三个训练、两个 VRX、两个评估槽位；仿真通信分区隔离 |

通信试验是消息级故障注入，未包含真实无线链路或协议栈。受损消息按各艇接收缓存分别计算，耗时与可靠通信的一次团队计算不同。三艇当前消息载荷为广播 **504 字节/控制步**，逐邻艇单播 **1,008 字节/控制步**，均不含协议头；0.2 秒周期对应约 2.52/5.04 kB/s。

## 3. 已取得的主要结果

### 3.1 VRX：原论文参考与本研究结果

三艇防御、码头、攻击艇灵活度 2.25。论文值与本研究使用不同试验；本轮两模型则使用相同 100 个初始场景、相同反馈同步和任务规则。

| 来源/模型 | 回合数 | 成功率 | 碰撞率 | 突破率 |
|---|---:|---:|---:|---:|
| 原论文报告 | 100 | 85.00% | 11.00% | 4.00% |
| 本研究早期论文参数 ARBoids，旧场景组 | 100 | 98.00% | 0.00% | 2.00% |
| 本轮 ARBoids（本研究实现与部署） | 100 | **100.00%** | **0.00%** | **0.00%** |
| 本轮预测增强 MAPPO，历史冻结模型 | 100 | 99.00% | 0.00% | 1.00% |

本轮 ARBoids 的 100 次成功均为捕获；MAPPO 为 99 次成功、1 次突破。MAPPO 在该 VRX 条件下未超过本研究 ARBoids。两轮 ARBoids 的场景及部署时点不同，98% 到 100% 不能归因为某一项单独修正。

本轮 100/100 成功的 Wilson 95% 区间约为 **96.30%–100%**；0/100 碰撞或突破的上限约为 **3.70%**。这是已观察到的有限样本结果。

### 3.2 历史冻结模型：6,000 个独立 2D 场景

三艇、灵活度 2.0、60 秒、原 APF 与随机海流；两模型场景完全配对，冻结后不再更新。ARBoids 训练 100 万步，历史 MAPPO 另经过自适应开发和累计 7,538,623 步；因此这组结果说明受测模型之间的性能差异，尚不能单独归因于算法而排除额外训练。

| 模型 | 成功 | 碰撞 | 突破 |
|---|---:|---:|---:|
| ARBoids（本研究实现） | 5,553/6,000，92.550% | 427/6,000，7.117% | 20/6,000，0.333% |
| 预测增强 MAPPO | **5,910/6,000，98.500%** | **31/6,000，0.517%** | 59/6,000，0.983% |
| MAPPO 相对变化 | **+5.950 个百分点** | **−6.600 个百分点** | **+0.650 个百分点** |

MAPPO 成功率的 Wilson 95% 区间为 **98.160%–98.778%**，碰撞率为 **0.364%–0.732%**，突破率为 **0.763%–1.266%**。严格验收记录仍为 `performance_passed=false`：成功率未到 99%，突破率高于同场景 ARBoids。

| 预先固定的场景种子 | ARBoids 成功/碰撞/突破（%） | MAPPO 成功/碰撞/突破（%） |
|---|---|---|
| 74000–75999，2,000 回合 | 92.10 / 7.70 / 0.20 | 98.80 / 0.25 / 0.95 |
| 76000–77999，2,000 回合 | 92.70 / 7.00 / 0.30 | 98.65 / 0.45 / 0.90 |
| 78000–79999，2,000 回合 | 92.85 / 6.65 / 0.50 | 98.05 / 0.85 / 1.10 |

三批均有成功提升、碰撞下降，同时均有突破上升。它们描述同一冻结模型的跨场景表现，不等于三个独立训练重复。

结果来源：[冻结复测记录](../train/experiments/predictive-mappo-performance-stable-best-42/qualification.json)、[逐项统计](../train/experiments/predictive-mappo-performance-stable-best-42/audit-candidate.json)。

### 3.3 统一 4,800 场景开发比较

所有数字均用于开发选模，不能再作独立验收。固定按成功率优先、再比较碰撞与突破选择上下文候选。

| 候选 | 更新次数 | 成功率 | 碰撞率 | 突破率 |
|---|---:|---:|---:|---:|
| ARBoids（本研究实现） | — | 91.875% | 7.875% | 0.250% |
| 九点联合预测，7 米余量 | 255 | 98.625% | **0.583%** | 0.792% |
| 上下文学习增益，原任务约束 | 265 | **98.688%** | 0.625% | 0.688% |
| 99% 成功约束的真实事件目标 | 273 | 98.583% | 0.750% | **0.667%** |
| 事件目标，恢复全 Actor 训练 | 281 | 98.313% | 0.917% | 0.771% |
| 九点联合预测，6.5 米余量 | 255 | 98.396% | 0.896% | 0.708% |
| 五点联合预测，7 米余量 | 255 | 98.542% | 0.771% | 0.688% |
| 拦截优先级温度改为 1 米 | 255 | 98.500% | 0.729% | 0.771% |

上下文候选为 4,737 次成功、30 次碰撞、33 次突破。相对 ARBoids，367 个失败场景转为成功、40 个成功场景退步，净增加 327 次成功。事件目标相对上下文候选少 1 次突破，但多 6 次碰撞、少 5 次成功；恢复全 Actor、缩小余量、减少网格和增强优先级均未提高这组开发成功率。有限开发搜索不能据此证明全局最优。

### 3.4 冻结模型泛化：每条件 500 回合

下表各单元为“成功 / 碰撞 / 突破”，单位均为百分比。沿用同一三艇训练 Actor，2/4 艇不重训。

| 条件 | ARBoids（本研究） | 预测增强 MAPPO |
| --- | --- | --- |
| 三艇，灵活度 1.5 | 93.40 / 6.60 / 0.00 | 98.80 / 0.80 / 0.40 |
| 三艇，灵活度 2.25 | 93.60 / 6.00 / 0.40 | 97.60 / 1.20 / 1.20 |
| 三艇，灵活度 2.5 | 89.00 / 9.20 / 1.80 | 97.40 / 0.60 / 2.00 |
| 三艇，灵活度 3.0 | 90.20 / 8.20 / 1.60 | 97.40 / 0.60 / 2.00 |
| 两艇，灵活度 2.0 | 64.60 / 12.00 / 23.40 | 69.00 / 2.40 / 28.60 |
| 四艇，灵活度 2.0 | 66.20 / 33.40 / 0.40 | 95.80 / 2.80 / 1.40 |
| 三艇，直接趋近目标 | 91.20 / 6.40 / 2.40 | 97.80 / 0.20 / 2.00 |
| 三艇，扩大排斥范围 APF | 66.00 / 34.00 / 0.00 | 98.60 / 0.80 / 0.60 |

八个条件均观察到成功率上升和碰撞率下降；其中七个条件突破率升高，直接趋近目标条件的突破率从 2.40% 降到 2.00%。两艇条件的 MAPPO 仍有 **28.60% 突破**，说明减少艇数后的防御能力不足。四艇的碰撞从 33.40% 降至 2.80%，但仍高于 1% 目标。

扩大排斥范围 APF 中，MAPPO 的 493 次成功由 **178 次捕获、315 次守时成功**组成；ARBoids 的 330 次成功为 160 次捕获、170 次守时成功。这里的高成功率不能全部解读为捕获能力。

五艇九点联合组合为 100,000，超过实现的 10,000 上限；本轮记录为“不支持”，没有虚构成功率。

### 3.5 冻结模型的通信扰动

每个条件使用相同场景种子；2D 每条件 500 回合，VRX 每条件 100 回合。受测模型均为历史冻结 MAPPO。

| 条件 | 2D 成功/碰撞/突破（%） | VRX 成功/碰撞/突破（%） |
| --- | --- | --- |
| 可靠通信 | 98.40 / 0.80 / 0.80 | 99.00 / 0.00 / 1.00 |
| 延迟 200 ms | 98.20 / 0.60 / 1.20 | 97.00 / 1.00 / 2.00 |
| 延迟 400 ms | 98.20 / 1.40 / 0.40 | 99.00 / 0.00 / 1.00 |
| 丢包 10% | 98.20 / 0.80 / 1.00 | 98.00 / 0.00 / 2.00 |
| 丢包 30% | 98.60 / 0.80 / 0.60 | 99.00 / 0.00 / 1.00 |

这些有限样本没有呈现随延迟或丢包比例严格单调恶化的关系，也不能据此说丢包有益。2D 400 ms 延迟的碰撞率为 1.40%，超过 1% 目标；VRX 200 ms 延迟的成功率降到 97%。

### 3.6 推理与控制耗时

下表均值按控制步数加权，单位毫秒；控制耗时包含推理及控制处理。跨平台分别报告，不混成同一硬件性能。截止期限为 200 ms。

| 执行平台/方法 | 回合 | 控制步 | 推理均值 | 控制均值 | 最大控制耗时 | 超期次数 |
| --- | --- | --- | --- | --- | --- | --- |
| 本机 / MAPPO | 88 | 5151 | 6.35 | 7.45 | 149.94 | 0 |
| 本机 / ARBoids | 88 | 5483 | 1.23 | 3.55 | 173.70 | 0 |
| 服务器并发 / ARBoids | 12 | 716 | 1.29 | 3.42 | 14.09 | 0 |
| 服务器并发 / MAPPO | 12 | 711 | 8.90 | 10.84 | 21.36 | 0 |
| 服务器并发 / MAPPO，丢包 30% | 100 | 5840 | 23.24 | 24.88 | 40.89 | 0 |
| 服务器并发 / MAPPO，丢包 10% | 100 | 5933 | 23.24 | 24.94 | 353.30 | 1 |
| 服务器并发 / MAPPO，延迟 400 ms | 100 | 5686 | 23.24 | 24.86 | 50.00 | 0 |
| 服务器并发 / MAPPO，延迟 200 ms | 100 | 5952 | 23.32 | 24.96 | 39.17 | 0 |

服务器并发环境中，10% 丢包组共 5,933 个控制步出现 1 次超期，最大完整控制耗时约 353.30 ms；不能写为所有条件均零超时。可靠通信 2D 三艇的每回合平均推理耗时，ARBoids 约 0.57 ms、MAPPO 约 7.4 ms；四艇 MAPPO 约 9.15 ms，反映联合预测带来的计算开销。

## 4. 本轮正式同预算实验：已完成的阶段结果

### 4.1 固定实验设计

训练种子为 101、202、303、404、505。每个种子先独立训练 **1,000,000 步 ARBoids**，再派生下列对照；完整回合边界允许最后一回合最多超出 299 步。

| 方法 | 预训练之后的处理 | 检验的问题 |
|---|---|---|
| ARBoids 同预算续训 | SAC 再训练 7,538,623 步，保留全部训练状态 | 额外训练本身能否取得收益 |
| 无预测 MAPPO | 同步数续训，关闭预测分支及相应条件 | 预测分支整体作用 |
| 固定规则 | 同预算 SAC Actor 加九点联合预测与固定增益 1，不再学习 | 规则是否已经足够 |
| 无联合选择 | 保留网格预测，改为局部风险调整 | 全队联合选择的贡献 |
| 固定增益 | 联合预测增益固定 1、取消上下文头，其他网络照常训练 | 学习增益的贡献 |
| 完整模型 | 联合预测、上下文增益、两阶段约束 PPO | 主要方法 |

各 MAPPO 组均按固定配方训练全部 Actor，不重演历史 seed-42 的自适应调参过程。无预测消融同时减少信息交互条件，尚不能完全分开“额外通信”与“预测结构”的贡献。环境交互步数相同也不等于计算量相同。

### 4.2 种子 101：每方法 2,000 个相同场景

| 方法 | 成功率 | 碰撞率 | 突破率 | 成功/碰撞/突破次数 |
| --- | --- | --- | --- | --- |
| ARBoids，100 万步起点 | 92.85% | 7.10% | 0.05% | 1857 / 142 / 1 |
| 无预测 MAPPO | 93.00% | 6.75% | 0.25% | 1860 / 135 / 5 |
| 无联合选择 | 96.70% | 2.75% | 0.55% | 1934 / 55 / 11 |
| 固定增益 | 98.15% | 0.90% | 0.95% | 1963 / 18 / 19 |
| 完整预测增强 MAPPO | 98.60% | 0.90% | 0.50% | 1972 / 18 / 10 |

同预算 ARBoids 续训尚未结束，固定规则对照等待该模型，因此两项还没有终点结果。

在同一种子上，完整模型相对无预测组成功率 **+5.60 个百分点**、碰撞率 **−5.85 个百分点**、突破率 **+0.25 个百分点**；相对无联合组成功率 **+1.90 个百分点**、碰撞率 **−1.85 个百分点**；相对固定增益组多 9 次成功、少 9 次突破，碰撞次数相同。这是有价值的单种子信号，尚不支持五种子稳定性结论。

完整模型仍未达到 99% 成功；相对自身 100 万步起点，突破从 1/2,000 增至 10/2,000。是否优于同预算 ARBoids、固定规则，以及其他种子能否复现，仍待当前队列给出结果。

### 4.3 五个预训练起点

| 训练种子 | 成功率 | 碰撞率 | 突破率 |
| --- | --- | --- | --- |
| 101 | 92.85% | 7.10% | 0.05% |
| 202 | 91.70% | 8.05% | 0.25% |
| 303 | 41.35% | 58.60% | 0.05% |
| 404 | 56.45% | 43.45% | 0.10% |
| 505 | 91.00% | 8.75% | 0.25% |
| 五种子均值 | 74.67% | 25.19% | 0.14% |

预训练成功率范围为 **41.35%–92.85%**，训练种子间的样本标准差约 **24.13 个百分点**。这反映预训练阶段差异很大，不能把 seed-42 或种子 101 的良好表现直接推到所有种子；也不能把这些起点指标当作同预算续训的最终结果。

### 4.4 种子 101 完整模型的泛化

每条件 500 回合；对应的同预算 ARBoids 泛化尚未完成。不同于第 3.4 节的历史冻结模型及场景组，下面不作跨表逐回合配对比较。

| 条件 | 成功率 | 碰撞率 | 突破率 |
| --- | --- | --- | --- |
| 三艇，灵活度 1.5 | 98.80% | 1.00% | 0.20% |
| 三艇，灵活度 2.25 | 98.20% | 0.80% | 1.00% |
| 三艇，灵活度 2.5 | 97.80% | 1.00% | 1.20% |
| 三艇，灵活度 3.0 | 97.20% | 1.00% | 1.80% |
| 两艇，灵活度 2.0 | 71.00% | 2.20% | 26.80% |
| 四艇，灵活度 2.0 | 92.80% | 6.40% | 0.80% |
| 三艇，直接趋近目标 | 98.20% | 0.20% | 1.60% |
| 三艇，扩大排斥范围 APF | 93.60% | 6.40% | 0.00% |

两艇突破率仍为 26.80%；四艇和扩大排斥 APF 的碰撞率均为 6.40%。扩大排斥 APF 的 468 次成功中，282 次为捕获、186 次为守时成功。当前结果已经指出规模变化与任务保持仍是主要缺口。

### 4.5 完成进度

截至上述时间，调度器运行中，**648/772 项完成，未记录失败任务**。

| 任务类型 | 已完成 | 计划 | 剩余 |
| --- | --- | --- | --- |
| VRX 回合 | 600 | 600 | 0 |
| 2D 评估任务 | 38 | 136 | 98 |
| 训练/固定规则构建 | 9 | 35 | 26 |
| 五艇能力限制记录 | 1 | 1 | 0 |

35 项训练/构建中，已完成 5 个预训练和种子 101 的 4 个 MAPPO 续训；固定规则的 5 项属于模型构建，不能算新增训练。主要方法除预训练 5/5 外，无预测、无联合、固定增益、完整模型均为 **1/5**，同预算 ARBoids 和固定规则均为 **0/5**。

当前运行：`train-101-arboids`、`train-202-fixed_gain`、`train-202-no_prediction`。600 个 VRX 短任务占了大量任务数，648/772 不是训练计算量已完成 84%。遵照既定安排，本轮配方与预算保持不变，全部完成后再集中修改方法。

## 5. 早期实现与部署结果

这些记录发生在当前正式实验之前；源代码协议、论文参数协议及部署版本存在差别，不能与本轮场景合并作一个样本。早期表中的碰撞/突破按终止类别统计。

| 运行/条件 | 回合 | 成功率 | 碰撞率 | 突破率 |
| --- | --- | --- | --- | --- |
| 早期代码协议 / VRX 码头，灵活度 2.25 | 10 | 100.00% | 0.00% | 0.00% |
| 早期代码协议 / VRX 开阔水域，灵活度 2.25 | 10 | 40.00% | 0.00% | 60.00% |
| 早期代码协议 / 2D，灵活度 2.0 | 100 | 93.00% | 6.00% | 1.00% |
| 早期代码协议 / 2D，灵活度 2.25 | 100 | 93.00% | 5.00% | 2.00% |
| 论文参数协议 / VRX 码头，灵活度 2.25 | 100 | 98.00% | 0.00% | 2.00% |
| 论文参数协议 / VRX 开阔水域，灵活度 2.25 | 10 | 70.00% | 0.00% | 30.00% |
| 论文参数协议 / 2D，灵活度 2.0 | 100 | 95.00% | 4.00% | 1.00% |
| 论文参数协议 / 2D，灵活度 2.25 | 100 | 96.00% | 4.00% | 0.00% |

早期开阔水域的成功率分别为 4/10 和 7/10，显示码头中的高成功率不能自动推广到开阔水域。当前 600 回合正式 VRX 使用码头条件，未对新 MAPPO 补做正式开阔水域实验。原始证据位于 [早期运行](D:/ARBoids/train/experiments/main-seed42-20260921) 和 [论文参数运行](D:/ARBoids/train/experiments/paper-parameters-seed42-20260921)。

## 6. 目前已能支持的结论与仍未解决的问题

1. **实现与部署已得到较强结果。** 本研究 ARBoids 的本轮 VRX 点估计超过原论文报告，但具体实现改动的独立作用还未逐项证明。
2. **联合预测在减少碰撞方面有重复出现的实证信号。** 历史 6,000 场景、八种泛化条件和新种子 101 均有相应证据；五种子同预算实验负责继续核验算法归因。
3. **避碰与任务保持尚未同时解决。** 历史冻结模型多数条件突破上升，新种子 101 也未消除这一取舍；当前任务优先级不等于对未来拦截机会的充分约束。
4. **当前没有任何一份记录证明所有目标同时达成。** 99% 成功、1% 碰撞、突破不恶化与跨训练种子稳定性需要分别检查，已有严格验收失败如实保留。
5. **规模和通信依赖存在明确边界。** 两艇防御较弱、五艇组合超限，通信试验仍是消息故障注入；实际动作的名义安全没有理论保证。
6. **下一轮提出但尚未完成的工作** 是任务保持的联合残差协调：约束拦截机会退化、检查最终实际动作、处理预测误差及可扩展联合选择。现有攻击轨迹/终止预测探针只是相关尝试，不能写成这条研究路线已经完成。

这些材料已经足以撰写方法与阶段性结果部分；它们尚未构成“全面超越原论文并满足独立发表”的完整证明。独立论文的核心应落到可识别的新机制及其证据，不能仅由某一个更高的成功率决定。

## 附录 A. 全部已保存 MAPPO 开发记录

共检索到 **73 个**相关开发/检查目录，其中包括候选、重复检查、参照及诊断记录，**不是 73 种独立算法**。下表逐个列出目录的主要结果，不将不同样本集、不同训练阶段混排为同条件算法排名。

`M/` 指 `D:/ARBoids/train/experiments/`，`W/` 指本工作树 `train/experiments/`；目录均带前缀 `predictive-mappo-performance-`。三项数值顺序为 **成功 / 碰撞 / 突破，单位 %**。表中的“通过”只在明确相应范围时使用；旧开发门槛通过不等于最终严格验收通过。

| 序号 | 目录 | 类型 | 回合 | 成功/碰撞/突破 | 更新/累计步数 | 记录结论 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | [W/backtrack42](../train/experiments/predictive-mappo-performance-backtrack42/validation.json) | 开发评估 | 800 | 98.750 / 0.625 / 0.625 | 270 / 7,433,827 | 最终门槛未通过 |
| 2 | [W/context4800-42](../train/experiments/predictive-mappo-performance-context4800-42/validation.json) | 开发评估 | 4800 | 98.688 / 0.625 / 0.688 | 265 / 7,538,623 | 最终门槛未通过 |
| 3 | [W/context9-42](../train/experiments/predictive-mappo-performance-context9-42/validation.json) | 开发评估 | 800 | 98.875 / 0.875 / 0.250 | 263 / 7,099,675 | 最终门槛未通过 |
| 4 | [W/development4800](../train/experiments/predictive-mappo-performance-development4800/summary.json) | ARBoids 开发参照 | 4800 | 91.875 / 7.875 / 0.250 | — | 开发参照 |
| 5 | [W/event99-42](../train/experiments/predictive-mappo-performance-event99-42/validation.json) | 开发评估 | 4800 | 98.583 / 0.750 / 0.667 | 273 / 8,485,630 | 最终门槛未通过 |
| 6 | [W/eventfull-42](../train/experiments/predictive-mappo-performance-eventfull-42/validation.json) | 开发评估 | 4800 | 98.313 / 0.917 / 0.771 | 281 / 9,451,767 | 最终门槛未通过 |
| 7 | [W/grid9-margin65-42](../train/experiments/predictive-mappo-performance-grid9-margin65-42/training-candidate.json) | 训练场景探针 | 512 | 98.633 / 1.367 / 0.000 | — | 不赋予验收资格 |
| 8 | [W/grid9-margin65-check42](../train/experiments/predictive-mappo-performance-grid9-margin65-check42/validation.json) | 开发评估 | 800 | 99.250 / 0.500 / 0.250 | — | 最终门槛未通过 |
| 9 | [W/grid9-margin65-development4800](../train/experiments/predictive-mappo-performance-grid9-margin65-development4800/validation.json) | 开发评估 | 4800 | 98.396 / 0.896 / 0.708 | — | 最终门槛未通过 |
| 10 | [W/grid9-task10-42](../train/experiments/predictive-mappo-performance-grid9-task10-42/training-candidate.json) | 训练场景探针 | 512 | 98.438 / 0.977 / 0.586 | — | 不赋予验收资格 |
| 11 | [W/interceptgrid42](../train/experiments/predictive-mappo-performance-interceptgrid42/training-candidate.json) | 训练场景探针 | 512 | 98.242 / 0.977 / 0.781 | — | 不赋予验收资格 |
| 12 | [W/interceptgrid6-42](../train/experiments/predictive-mappo-performance-interceptgrid6-42/training-candidate.json) | 训练场景探针 | 512 | 98.242 / 1.367 / 0.391 | — | 不赋予验收资格 |
| 13 | [W/joint3s-42](../train/experiments/predictive-mappo-performance-joint3s-42/training-candidate.json) | 训练场景探针 | 512 | 98.242 / 0.977 / 0.781 | — | 不赋予验收资格 |
| 14 | [W/jointgoal42](../train/experiments/predictive-mappo-performance-jointgoal42/validation.json) | 开发评估 | 800 | 98.750 / 1.000 / 0.250 | 260 / 6,833,526 | 最终门槛未通过 |
| 15 | [W/jointgrid5-development4800](../train/experiments/predictive-mappo-performance-jointgrid5-development4800/validation.json) | 开发评估 | 4800 | 98.542 / 0.771 / 0.688 | — | 最终门槛未通过 |
| 16 | [W/jointgrid6-42](../train/experiments/predictive-mappo-performance-jointgrid6-42/training-candidate.json) | 训练场景探针 | 512 | 98.242 / 1.367 / 0.391 | — | 不赋予验收资格 |
| 17 | [W/jointgrid9-42](../train/experiments/predictive-mappo-performance-jointgrid9-42/training-candidate.json) | 训练场景探针 | 512 | 98.438 / 0.977 / 0.586 | — | 不赋予验收资格 |
| 18 | [W/jointgrid9-check42](../train/experiments/predictive-mappo-performance-jointgrid9-check42/validation.json) | 开发评估 | 800 | 99.250 / 0.625 / 0.125 | — | 最终门槛未通过 |
| 19 | [W/jointgrid9-development4800](../train/experiments/predictive-mappo-performance-jointgrid9-development4800/validation.json) | 开发评估 | 4800 | 98.625 / 0.583 / 0.792 | — | 最终门槛未通过 |
| 20 | [W/jointmix-holdout42](../train/experiments/predictive-mappo-performance-jointmix-holdout42/training-candidate.json) | 训练场景探针 | 1024 | 98.438 / 0.586 / 0.977 | — | 不赋予验收资格 |
| 21 | [W/jointmix42](../train/experiments/predictive-mappo-performance-jointmix42/training-candidate.json) | 训练场景探针 | 256 | 98.828 / 0.391 / 0.781 | — | 不赋予验收资格 |
| 22 | [W/jointmixlearn42](../train/experiments/predictive-mappo-performance-jointmixlearn42/validation.json) | 开发评估 | 800 | 98.750 / 1.000 / 0.250 | 255 / 6,531,369 | 最终门槛未通过 |
| 23 | [W/nativejoint42](../train/experiments/predictive-mappo-performance-nativejoint42/training-candidate.json) | 训练场景探针 | 512 | 97.266 / 1.172 / 1.563 | — | 不赋予验收资格 |
| 24 | [W/priority1-42](../train/experiments/predictive-mappo-performance-priority1-42/training-candidate.json) | 训练场景探针 | 512 | 98.438 / 1.172 / 0.391 | — | 不赋予验收资格 |
| 25 | [W/priority1-development4800](../train/experiments/predictive-mappo-performance-priority1-development4800/validation.json) | 开发评估 | 4800 | 98.500 / 0.729 / 0.771 | — | 最终门槛未通过 |
| 26 | [W/stable-best-42](../train/experiments/predictive-mappo-performance-stable-best-42/validation.json) | 开发评估 | 4800 | 98.688 / 0.625 / 0.688 | — | 最终门槛未通过 |
| 27 | [W/terminal9-42](../train/experiments/predictive-mappo-performance-terminal9-42/training-candidate.json) | 训练场景探针 | 512 | 98.242 / 1.172 / 0.586 | — | 不赋予验收资格 |
| 28 | [M/baseline](D:/ARBoids/train/experiments/predictive-mappo-performance-baseline/summary.json) | ARBoids 开发参照 | 200 | 93.000 / 6.500 / 0.500 | — | 开发参照 |
| 29 | [M/batch42](D:/ARBoids/train/experiments/predictive-mappo-performance-batch42/validation.json) | 开发评估 | 200 | 93.500 / 6.500 / 0.000 | 100 / 510,604 | 最终门槛未通过 |
| 30 | [M/bounded-explore42](D:/ARBoids/train/experiments/predictive-mappo-performance-bounded-explore42/validation.json) | 开发评估 | 200 | 93.000 / 7.000 / 0.000 | 125 / 835,260 | 最终门槛未通过 |
| 31 | [M/bounded42](D:/ARBoids/train/experiments/predictive-mappo-performance-bounded42/validation.json) | 开发评估 | 200 | 91.000 / 9.000 / 0.000 | 115 / 692,991 | 最终门槛未通过 |
| 32 | [M/breach42](D:/ARBoids/train/experiments/predictive-mappo-performance-breach42/one-step.pth) | 单次校准/更新 | — | — / — / — | 251 / 6,409,480 | 无独立性能汇总 |
| 33 | [M/calibrated42](D:/ARBoids/train/experiments/predictive-mappo-performance-calibrated42/validation.json) | 开发评估 | 200 | 91.500 / 8.000 / 0.500 | 80 / 217,870 | 最终门槛未通过 |
| 34 | [M/censored42](D:/ARBoids/train/experiments/predictive-mappo-performance-censored42/one-step.pth) | 单次校准/更新 | — | — / — / — | 251 / 6,409,480 | 无独立性能汇总 |
| 35 | [M/compact42](D:/ARBoids/train/experiments/predictive-mappo-performance-compact42/validation.json) | 开发评估 | 800 | 98.000 / 1.125 / 0.875 | — | 最终门槛未通过 |
| 36 | [M/compatibility42](D:/ARBoids/train/experiments/predictive-mappo-performance-compatibility42/validation.json) | 开发评估 | 800 | 97.875 / 1.250 / 0.875 | — | 最终门槛未通过 |
| 37 | [M/coordinated42](D:/ARBoids/train/experiments/predictive-mappo-performance-coordinated42/validation.json) | 开发评估 | 800 | 98.625 / 1.125 / 0.250 | — | 最终门槛未通过 |
| 38 | [M/coordination42](D:/ARBoids/train/experiments/predictive-mappo-performance-coordination42/validation.json) | 开发评估 | 800 | 94.375 / 5.500 / 0.125 | 135 / 902,575 | 最终门槛未通过 |
| 39 | [M/cost42](D:/ARBoids/train/experiments/predictive-mappo-performance-cost42/validation.json) | 开发评估 | 200 | 94.000 / 6.000 / 0.000 | 100 / 266,435 | 最终门槛未通过 |
| 40 | [M/counterfactual42](D:/ARBoids/train/experiments/predictive-mappo-performance-counterfactual42/one-step.pth) | 单次校准/更新 | — | — / — / — | 251 / 6,403,996 | 无独立性能汇总 |
| 41 | [M/coverage42](D:/ARBoids/train/experiments/predictive-mappo-performance-coverage42/validation.json) | 开发评估 | 200 | 93.000 / 6.500 / 0.500 | 88 / 234,482 | 最终门槛未通过 |
| 42 | [M/credit42](D:/ARBoids/train/experiments/predictive-mappo-performance-credit42/validation.json) | 开发评估 | 800 | 93.375 / 6.250 / 0.375 | 120 / 735,172 | 最终门槛未通过 |
| 43 | [M/development800](D:/ARBoids/train/experiments/predictive-mappo-performance-development800/summary.json) | ARBoids 开发参照 | 800 | 93.000 / 6.625 / 0.375 | — | 开发参照 |
| 44 | [M/earlyrecovery42](D:/ARBoids/train/experiments/predictive-mappo-performance-earlyrecovery42/one-step.pth) | 单次校准/更新 | — | — / — / — | 251 / 6,409,480 | 无独立性能汇总 |
| 45 | [M/eventcredit42](D:/ARBoids/train/experiments/predictive-mappo-performance-eventcredit42/validation.json) | 开发评估 | 800 | 95.000 / 4.375 / 0.625 | 261 / 7,081,751 | 最终门槛未通过 |
| 46 | [M/expectation42](D:/ARBoids/train/experiments/predictive-mappo-performance-expectation42/validation.json) | 开发评估 | 800 | 97.875 / 1.375 / 0.750 | — | 最终门槛未通过 |
| 47 | [M/exploration42](D:/ARBoids/train/experiments/predictive-mappo-performance-exploration42/validation.json) | 开发评估 | 200 | 94.500 / 5.000 / 0.500 | 50 / 132,563 | 最终门槛未通过 |
| 48 | [M/frozen150-audit40000](D:/ARBoids/train/experiments/predictive-mappo-performance-frozen150-audit40000/audit-candidate.json) | 当时冻结检查 | 1000 | 93.800 / 6.100 / 0.100 | 150 / 1,725,850 | 验收未通过 |
| 49 | [M/fullcost42](D:/ARBoids/train/experiments/predictive-mappo-performance-fullcost42/validation.json) | 开发评估 | 800 | 94.000 / 5.625 / 0.375 | 170 / 2,601,584 | 最终门槛未通过 |
| 50 | [M/gate-step42](D:/ARBoids/train/experiments/predictive-mappo-performance-gate-step42/validation.json) | 开发评估 | 200 | 93.000 / 6.500 / 0.500 | 100 / 521,114 | 最终门槛未通过 |
| 51 | [M/gate42](D:/ARBoids/train/experiments/predictive-mappo-performance-gate42/validation.json) | 开发评估 | 200 | 93.500 / 6.500 / 0.000 | 115 / 687,216 | 最终门槛未通过 |
| 52 | [M/intent42](D:/ARBoids/train/experiments/predictive-mappo-performance-intent42/validation.json) | 开发评估 | 800 | 97.875 / 1.500 / 0.625 | — | 最终门槛未通过 |
| 53 | [M/interceptor42](D:/ARBoids/train/experiments/predictive-mappo-performance-interceptor42/validation.json) | 开发评估 | 800 | 98.000 / 1.375 / 0.625 | — | 最终门槛未通过 |
| 54 | [M/joint42](D:/ARBoids/train/experiments/predictive-mappo-performance-joint42/validation.json) | 开发评估 | 200 | 95.500 / 4.000 / 0.500 | 95 / 464,403 | 最终门槛未通过 |
| 55 | [M/mincollision42](D:/ARBoids/train/experiments/predictive-mappo-performance-mincollision42/validation.json) | 开发评估 | 800 | 94.750 / 5.000 / 0.250 | 270 / 7,325,263 | 最终门槛未通过 |
| 56 | [M/mixture42](D:/ARBoids/train/experiments/predictive-mappo-performance-mixture42/training-candidate.json) | 训练场景探针 | 256 | 94.922 / 4.297 / 0.781 | — | 不赋予验收资格 |
| 57 | [M/nativeanchor42](D:/ARBoids/train/experiments/predictive-mappo-performance-nativeanchor42/validation.json) | 开发评估 | 800 | 97.625 / 1.125 / 1.250 | — | 最终门槛未通过 |
| 58 | [M/newgate42](D:/ARBoids/train/experiments/predictive-mappo-performance-newgate42/validation.json) | 开发评估 | 800 | 71.250 / 22.125 / 6.625 | 125 / 989,159 | 最终门槛未通过 |
| 59 | [M/prediction42](D:/ARBoids/train/experiments/predictive-mappo-performance-prediction42/validation.json) | 开发评估 | 200 | 94.500 / 5.500 / 0.000 | 40 / 106,132 | 最终门槛未通过 |
| 60 | [M/recovery42](D:/ARBoids/train/experiments/predictive-mappo-performance-recovery42/validation.json) | 开发评估 | 800 | 95.250 / 4.750 / 0.000 | 250 / 6,355,005 | 最终门槛未通过 |
| 61 | [M/refine42](D:/ARBoids/train/experiments/predictive-mappo-performance-refine42/metrics.csv) | 仅训练记录 | — | — / — / — | — | 无独立性能汇总 |
| 62 | [M/relation42](D:/ARBoids/train/experiments/predictive-mappo-performance-relation42/validation.json) | 开发评估 | 200 | 93.500 / 5.000 / 1.500 | 95 / 253,733 | 最终门槛未通过 |
| 63 | [M/reuse42](D:/ARBoids/train/experiments/predictive-mappo-performance-reuse42/validation.json) | 开发评估 | 200 | 93.500 / 6.000 / 0.500 | 135 / 964,217 | 最终门槛未通过 |
| 64 | [M/seed42](D:/ARBoids/train/experiments/predictive-mappo-performance-seed42/metrics.csv) | 仅训练记录 | — | — / — / — | — | 无独立性能汇总 |
| 65 | [M/stability42](D:/ARBoids/train/experiments/predictive-mappo-performance-stability42/validation.json) | 开发评估 | 800 | 93.625 / 6.125 / 0.250 | 150 / 1,725,850 | 最终门槛未通过 |
| 66 | [M/stable42](D:/ARBoids/train/experiments/predictive-mappo-performance-stable42/validation.json) | 开发评估 | 200 | 93.000 / 6.000 / 1.000 | 35 / 93,078 | 最终门槛未通过 |
| 67 | [M/step42](D:/ARBoids/train/experiments/predictive-mappo-performance-step42/validation.json) | 开发评估 | 200 | 93.000 / 7.000 / 0.000 | 80 / 212,854 | 最终门槛未通过 |
| 68 | [M/strongprior42](D:/ARBoids/train/experiments/predictive-mappo-performance-strongprior42/validation.json) | 开发评估 | 800 | 98.375 / 1.375 / 0.250 | 255 / 6,515,023 | 最终门槛未通过 |
| 69 | [M/thetaspace42](D:/ARBoids/train/experiments/predictive-mappo-performance-thetaspace42/validation.json) | 开发评估 | 800 | 94.875 / 4.875 / 0.250 | 190 / 3,438,099 | 最终门槛未通过 |
| 70 | [M/trustlearn42](D:/ARBoids/train/experiments/predictive-mappo-performance-trustlearn42/validation.json) | 开发评估 | 800 | 95.375 / 4.250 / 0.375 | 265 / 6,993,954 | 最终门槛未通过 |
| 71 | [M/trustprior42](D:/ARBoids/train/experiments/predictive-mappo-performance-trustprior42/calibrated.pth) | 单次校准/更新 | — | — / — / — | 250 / 6,355,005 | 无独立性能汇总 |
| 72 | [M/variance42](D:/ARBoids/train/experiments/predictive-mappo-performance-variance42/validation.json) | 开发评估 | 200 | 92.000 / 8.000 / 0.000 | 135 / 963,092 | 最终门槛未通过 |
| 73 | [M/warmstart42](D:/ARBoids/train/experiments/predictive-mappo-performance-warmstart42/validation.json) | 开发评估 | 200 | 94.000 / 5.500 / 0.500 | 15 / 38,922 | 最终门槛未通过 |

上述更新次数和步数是各自保存记录中的累计值，部分分支继承同一 checkpoint，不能相加当作独立训练总量。只有训练指标的记录不把训练碰撞率当作泛化表现。

### 当时冻结检查的成对结果

以下保留所有已保存的 audit 结果。后续已经查看或纳入开发的场景，不再具有新一轮独立验收资格；最终 74000–79999 场景是第 3.2 节的冻结复测。

| 记录 | 起始场景种子 | 回合 | ARBoids 成功/碰撞/突破（%） | 候选成功/碰撞/突破（%） | 最终验收 |
| --- | --- | --- | --- | --- | --- |
| [grid9-margin65-check42](../train/experiments/predictive-mappo-performance-grid9-margin65-check42/audit-candidate.json) | 72000 | 2000 | 92.250 / 7.550 / 0.200 | 98.650 / 0.750 / 0.600 | 未通过 |
| [jointgrid9-check42](../train/experiments/predictive-mappo-performance-jointgrid9-check42/audit-candidate.json) | 60000 | 2000 | 91.050 / 8.700 / 0.250 | 98.500 / 0.600 / 0.900 | 未通过 |
| [stable-best-42](../train/experiments/predictive-mappo-performance-stable-best-42/audit-candidate.json) | 74000 | 6000 | 92.550 / 7.117 / 0.333 | 98.500 / 0.517 / 0.983 | 未通过 |
| [coordinated42](D:/ARBoids/train/experiments/predictive-mappo-performance-coordinated42/audit-candidate.json) | 50000 | 1000 | 92.700 / 6.900 / 0.400 | 97.700 / 1.200 / 1.100 | 未通过 |
| [exploration42](D:/ARBoids/train/experiments/predictive-mappo-performance-exploration42/audit-candidate.json) | 30000 | 200 | 91.500 / 8.000 / 0.500 | 92.000 / 7.000 / 1.000 | 未通过 |
| [frozen150-audit40000](D:/ARBoids/train/experiments/predictive-mappo-performance-frozen150-audit40000/audit-candidate.json) | 40000 | 1000 | 93.200 / 6.500 / 0.300 | 93.800 / 6.100 / 0.100 | 未通过 |
| [joint42](D:/ARBoids/train/experiments/predictive-mappo-performance-joint42/audit-candidate.json) | 32000 | 200 | 94.500 / 5.000 / 0.500 | 93.500 / 6.000 / 0.500 | 未通过 |
| [relation42](D:/ARBoids/train/experiments/predictive-mappo-performance-relation42/audit-candidate.json) | 31000 | 200 | 93.000 / 7.000 / 0.000 | 93.000 / 7.000 / 0.000 | 未通过 |

### 额外开发评估及成对探针

`gate42` 的额外 800 场景记录（更新 80）为 **93.750 / 6.125 / 0.125**；`joint42` 的额外 800 场景记录（更新 95）为 **93.750 / 6.000 / 0.250**，两者开发检查均未通过。它们与上表各目录的主要 validation 记录使用的样本量/快照不同。

| 训练场景探针 | 回合 | 参照成功/碰撞/突破（%） | 候选成功/碰撞/突破（%） |
| --- | --- | --- | --- |
| interceptgrid42 | 512 | 98.242 / 1.172 / 0.586 | 98.242 / 0.977 / 0.781 |
| jointmix-holdout42 | 1024 | 98.145 / 1.074 / 0.781 | 98.438 / 0.586 / 0.977 |
| jointmix42 | 256 | 98.828 / 0.391 / 0.781 | 98.828 / 0.391 / 0.781 |
| nativejoint42 | 512 | 96.289 / 2.734 / 0.977 | 97.266 / 1.172 / 1.563 |
| terminal9-42 | 512 | 98.438 / 0.977 / 0.586 | 98.242 / 1.172 / 0.586 |
| mixture42 | 256 | 98.828 / 0.391 / 0.781 | 94.922 / 4.297 / 0.781 |

局部混合修正 `mixture42` 在 256 个训练场景上由 98.828% 成功退化至 94.922%，碰撞从 0.391% 升至 4.297%。联合混合、加密网格、改变余量/优先级、终止感知和 3 秒预测均保留其真实探针记录；512 或 1,024 场景的训练探针不代替第 3.3 节的统一开发比较。

`trustprior42` 的先验校准将增益设为 **0.04109**，校准批联合 KL 从单位增益对应的 5.9235 缩放到目标 0.01；另 256 个训练比较场景中，成功 **242→243**、碰撞 **14→13**、突破 **0→0**。只是少 1 次碰撞，碰撞率仍为 5.078%，记录明确标为未达预算、未获验收资格。

`counterfactual42`、`earlyrecovery42`、`censored42`、`breach42` 保存了单步更新模型；后三者 checkpoint 记录了早期突破恢复及独立任务基线。目录内未保存完整独立评估汇总，因此本文只确认实施和更新，不补造成功率。`seed42`、`refine42` 仅有训练指标和模型，没有可用的冻结性能汇总。

## 附录 B. 证据定位与模型标识

| 证据 | 位置 |
|---|---|
| 方法说明与历史开发结果 | [predictive-mappo.md](predictive-mappo.md) |
| 论文协议对齐与实现差异 | [paper-parameters.md](paper-parameters.md) |
| 正式方案、各对照定义及迁移说明 | [formal-evidence.md](formal-evidence.md) |
| 当前正式算法配方 | [formal-mappo.yaml](../train/configs/formal-mappo.yaml) |
| 正式训练入口 | [formal_study_training.py](../train/formal_study_training.py) |
| 真实事件评估入口 | [formal_study_evaluation.py](../train/formal_study_evaluation.py) |
| 消息级延迟/丢包实现 | [message_stress.py](../train/policy/message_stress.py) |
| 历史冻结 6,000 场景模型及验收 | [stable-best-42](../train/experiments/predictive-mappo-performance-stable-best-42) |

ARBoids seed-42 权重 SHA-256：`b8ec0679815ec28149ba8de1321e06f420a8df00bd74e22a1abfff0fb84fe03b`。历史冻结 MAPPO 权重 SHA-256：`3d0d531ddc9314682543b60a599e32f0f06610824dac2b16805eadd502e80ebb`。正式种子 101 完整模型 SHA-256：`b344ec617a6b4980bd062598d8b87b02561f2fe6f9fbe97ee91f66b0f97e524a`。

服务器当前原始结果根目录为 `/root/arboids-formal-20261005/train/experiments/formal-evidence-20261004/`。本文件是上述时间的读取汇总；服务器任务继续运行。远端原始数据定位如下：

| 结果 | 相对上述根目录的路径 |
|---|---|
| 状态与任务清单 | `status.json`、`manifest.remote.json` |
| VRX 成对结果与事件次数 | `vrx-paired-summary.json` |
| 分平台逐回合计时 | `vrx-timing-by-platform.json` |
| 冻结模型泛化 | `frozen-generalization/<条件>/<baseline或candidate>/summary.json` |
| 2D 通信扰动 | `communication/<条件>/summary.json` |
| 正式主要结果 | `seed-<种子>/primary/<方法>/summary.json` |
| 正式泛化结果 | `seed-<种子>/generalization/<条件>/<方法>/summary.json` |
| 跨训练种子统计 | `training-seed-statistics.json` |

本次整理只创建本汇总文档，未改训练源码、实验配方、模型或运行中的任务。
