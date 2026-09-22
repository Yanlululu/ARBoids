# 推进—转向分通道 MAPPO

本实现位于 `train/RL`，通过 `algorithm: channel_mappo` 选择。原 SAC 入口与旧模型格式继续可用。新模型格式是 `arboids-channel-mappo-v1`，旧的三输出 `ActorAdap` 权重不能直接作为新模型加载。

## 已实现的决策过程

每个控制周期先生成全部艇的 Boids 推力候选与随机学习候选，再进行一次同步候选交换，最后分别采样共同推力和差动推力门控。关系输入包括本艇坐标系下的相对位置、相对速度、相对航向、面向攻击者的方向关系、队友候选和消息年龄。两个独立的关系聚合模块分别服务推进与转向。

第一版假设可靠同步交换。二维训练与当前 VRX 单进程控制器在内存中完成该交换；没有实现跨进程 ROS 候选通信协议。消息年龄为零，VRX 会拒绝跨越一个动作周期的艇状态。交换对象是候选，不是同一步最终门控或最终控制。

`control.py` 定义统一物理约定：防守艇归一化动作对应 `u = 750 a + 250` N，`c = (u0 + u1)/2`，`d = (u1 - u0)/2`。分通道融合后恢复推进器命令并分别裁剪到 `[-500,1000]` N；这等价于通道空间的欧氏投影。底层船模接受的命令用于动作日志和 Q 监督。PPO 保存和评价两阶段潜变量的概率，不尝试给多对一投影后的命令反推概率。

策略输入支持有效艇掩码、队友掩码和人数。空队友集合安全返回零关系表示。Actor 不读取中央节点状态。中央 V/Q 在关系信息传递后汇总，没有固定艇编号嵌入。确定性 Actor 随艇顺序等变，团队价值不随艇顺序改变。

## 训练与指导

团队回报为原单艇奖励的均值。按时间保存整队 rollout，GAE 在真实终止时停止，在 rollout 截断时 bootstrap。当前任务的到时拒止是任务终止，不能当成继续任务的普通时间截断。

学习候选使用 `Normal + tanh`；门控使用 `Normal + sigmoid`。Rollout 保存实际学习候选、Boids 候选、接收消息、门控潜变量、两阶段旧概率、真实执行动作及前后中央状态。更新前自动检查历史概率一致性。PPO 使用逐艇的两阶段联合概率比裁剪，再按有效人数平均，使用共享团队优势。

V 使用在策略回报目标。Q 使用 `r_team + gamma * (1-terminal) * V_target(next_state)`。搜索固定两组候选，围绕冻结策略的确定性门控代表值，逐艇评价小幅门控变化；后续艇从已经改进的全队方案继续。候选评分始终包含完整联合动作，邻域与动作惩罚始终相对于初始方案。

对少量改善候选，模拟器从完整内存快照做配对短分支；只有第一步替换动作，后续使用同一冻结随机策略。两分支共享初始外部随机状态，验证后恢复训练随机状态，不影响主轨迹。分支目标单独校准 Q，不加入 PPO 或 V 的行为数据。主轨迹始终执行分散式策略自身采样的动作。

指导比较投影后实际推力，固定候选与教师目标，并阻断对候选编码器的梯度。只有得到分支支持的目标有正权重；近期验证支持比例不足时暂停指导。组合更新后的全 rollout KL 包含辅助损失影响，达到阈值停止后续 PPO 小批次更新。阈值与搜索步长是初始工程设置，尚需通过正式实验选择。

## 运行

在仓库根目录，Windows PowerShell：

```powershell
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -X utf8 -u train/train.py --config train/configs/channel-mappo-smoke.yaml --device cpu --seed 42
```

短配置是 256 步、64 维隐藏层、6 秒任务时限，以及 1/3/4 艇混合采样，专门验证调用链。正式配置采用 128 维隐藏层、60 秒任务、固定 3 艇和 1,000,000 步：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u train/train.py --config train/configs/channel-mappo.yaml --device cuda:0 --seed 42
```

也可直接使用 `train/train_mappo.py`，参数一致。训练目录中包含：

- `config.yaml`：该次运行的完整配置。
- `channel-mappo1.pth`：Actor、V、Q、目标 V、优化器和特征版本。
- `metrics1.csv`：PPO/KL、Q/V 损失、预测改善、分支改善、指导启停、分支步数、搜索评分次数和任务结果。
- `actions.csv`：每步每艇的两组候选、门控、真实执行推力、两阶段概率和结果码。

独立评估，`<run>` 替换成训练目录名：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u train/evaluate_policy.py --checkpoint train/experiments/<run>/channel-mappo1.pth --config train/experiments/<run>/config.yaml --episodes 100 --duration 60 --device cpu --output-dir train/experiments/<run>/evaluation
```

`--defenders 4` 可覆盖测试艇数。增加 `--teacher-search` 会运行中央搜索教师诊断；结果明确标记为 `teacher_search`，不能作为分散部署结果。默认评估只调用 Actor。结果分别报告突破、碰撞、捕获、到时拒止、最小间距，以及额外搜索次数。

VRX 使用同一策略、特征函数和融合函数，不加载训练搜索器：

```powershell
.\scripts\run_vrx.ps1 -Checkpoint 'train\experiments\<run>\channel-mappo1.pth' -Controller ChannelMAPPO -Setting 0 -TerminationRule paper -Headless
```

已激活 ROS/VRX 环境的 Linux 可直接调用：

```bash
python -X utf8 -u vrx/run_experiment.py --checkpoint train/experiments/<run>/channel-mappo1.pth --controller ChannelMAPPO --setting 0 --termination-rule paper --headless
```

VRX 保留训练动作第 0 项到 starboard、第 1 项到 port 的映射。轨迹附带两组候选、双门控、候选消息、两组注意力及实际发布命令。当前接入 APF 攻击者训练；旧的交替 SAC 对抗训练脚本不接受该配置。

## 对照开关与协议

| 设置 | 用途 |
| --- | --- |
| `policy.fusion: scalar` | 一个门控同时融合整组推力 |
| `policy.fusion: channels` | 共同推力/差动推力各一个门控 |
| `policy.fusion: thrusters` | 两推进器分别门控，检验物理组织方式 |
| `policy.aggregation: mean` | 使用同样关系输入但均值聚合 |
| `policy.share_candidates: false` | 屏蔽队友候选，保留本艇候选 |
| `policy.motion_features: false` | 屏蔽新增本艇速度/角速度和队友相对速度/航向；原 Boids 状态仍保留 |
| `guidance.enabled: false` | 无团队搜索指导的 MAPPO |
| `guidance.search_mode: independent` | 同一 Q，各艇对原始方案独立选择后组合 |
| `agent.training_team_sizes: [3, 4, 5]` | 同一共享策略的混合人数训练 |

这些结构消融使用新特征接口；不能把关闭若干字段后的版本宣称为原论文观测/网络的逐项复刻。

`environment.initial_min_spacing` 扩大人数较多时的初始化环半径，避免把初始碰撞误判为规模泛化失败。`canonical_agent_order` 对 APF 障碍物及随机海流分配采用几何顺序，消除已有环境中的编号依赖。新配置启用这两项，原环境默认值保持不变。比较算法时必须统一这两项以及奖励、时限和终止协议；原发布设置结果应另行标明。

## 检验与结果边界

```powershell
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s train/tests -v
```

测试覆盖融合退化/投影、真实执行一致性、两阶段概率与梯度、辅助梯度隔离、空队友与混合人数、网络与环境置换、GAE、联合/独立搜索区别、快照/随机数恢复、模型读写、VRX 公共策略及推进器映射，并保留原环境协议测试。

短训练和短 VRX 验证用于确认软件链路。短配置产生的权重尚未收敛，不能用于声称防御效果、规模泛化收益或优于原 ARBoids。性能结论需要正式训练、多随机种子和相同信息/回报/计算预算的对照。
