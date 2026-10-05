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
- `channel-mappo-best1.pth`：开发验证集上成功率最高的模型；同成功率先比较碰撞数，再比较回报，避免后续更新覆盖较好模型。
- `metrics1.csv`：PPO/KL、Q/V 损失、预测改善、分支改善、指导启停、分支步数、搜索评分次数和任务结果。
- `actions.csv`：每步每艇的两组候选、门控、真实执行推力、两阶段概率和结果码。

性能预训练应使用独立开发种子，正式测试种子不参与选模型。预训练配置为 `train/configs/channel-mappo-pretrain.yaml`。`train_mappo.py` 支持 `--max-steps` / `--max-seconds` 限制单次运行，限制到达后完成当前 rollout、验证并保存。续训使用 `--resume <checkpoint>`，恢复网络、优化器、随机数和指导验证历史，从新回合开始收集在策略数据；不保证与未中断仿真逐步相同。保持配置中的总步数和验证种子/回合数不变，避免改变课程进度或最佳模型评选口径。

预训练把 8 个独立环境的决策合并成一次 GPU 推理，每次采集 1,024 个团队时间步，即每环境 128 步。GAE 分别沿各环境的时间轴计算，再统一归一化优势；终止、重置和不足整批的末尾样本都保持独立。动作日志的 `env_id` 标识采样环境。验证同样批量推理，但每个场景保存独立随机数状态，避免批量大小改变海流或初始条件。训练总步数是所有环境的团队时间步之和。

预训练开启 `mappo.value_normalization`：V/Q 在归一化坐标下计算回归损失，并同步重标定末层权重，保持统计更新前后的原始价值输出一致。GAE、TD 目标、团队搜索和分支验证仍使用原始回报单位；目标 V 在参数软更新前先转换到相同坐标。还开启 `critic_layer_norm` 防止隐藏层 Tanh 饱和，Critic 学习率为 0.0003，Actor 保持 0.0001。`--initialize-actor <checkpoint>` 用已有 Actor 权重开始一个新训练阶段，重置 Critic 和优化器；来源及哈希写入配置，这类预训练成本应计入后续比较，不能当成随机初始化结果。

船模仍使用原三自由度方程及 0.05 秒积分步长，只缓存常量质量算子并减少矩阵分配。回归测试将带海流、推力限幅和非零耦合参数的长轨迹与原矩阵实现逐步对照。

经明确选择，也可用 `train/pretrain_actor.py --config train/configs/channel-mappo-pretrain.yaml --teacher <原ARBoids权重> --run-id <新目录>` 做动作示范预训练。脚本复用仓库 `RL.Networks.ActorAdap` 作为离线教师，先收集教师状态，再加入学生访问的状态，始终用教师实际融合推力作监督，优化新 Actor 的最终执行推力误差。示范数据只用于初始化 Actor，不进入 PPO rollout，中央 V/Q 保持待训练状态；部署文件不包含 SAC 教师。完成后用 `train_mappo.py --resume <目录>/channel-mappo-best1.pth --config <目录>/config.yaml --run-id <同一目录>` 继续完整 MAPPO。教师来源、哈希、示范数量、更新次数和耗时均记录在训练输出中。正式比较必须给对应对照相同的示范条件并计入原策略训练成本。

`train/refine_actor.py --checkpoint <已训练模型> --run-id <新目录>` 提供可选的门控示范初始化：仅在训练种子上对照完整回合，保留“原策略碰撞、推进门控修正后防守成功”的轨迹，结合普通成功轨迹监督门控；候选生成网络保持冻结。示范器按相对运动筛选接近事件，它只用于训练数据收集，不进入部署，也不等同于原 Q 搜索器。示范标签不加入 PPO，初始化后仍需继续完整 MAPPO，并以 Actor 独立执行通过连续验证。模型记录来源、训练种子、采集步数和监督更新次数，这些额外训练成本及示范条件必须计入正式对照。

当前性能准入要求是：固定 3 艇、60 秒，同协议和验证场景下，分散式 Actor 的成功率不低于原 ARBoids，同时碰撞率严格降低；两项持平不算达标。连续开发验证稳定后，再检查独立资格验证集，正式实验种子保持隔离。仿真执行完成、单次峰值或中央搜索器提高 Q 分数都不能代替性能达标。

PPO 与指导损失合并更新后，在整个 rollout 上检查 KL。超限时恢复 Actor 参数及 Adam 状态，用同一梯度缩小步长重试；最多 `mappo.max_kl_backtracks` 次（默认 8），仍超限则拒绝该步。缩步后结束当前 rollout 的 Actor 更新，下一次使用新收集的数据；基础学习率不被永久缩小。日志中的 `max_actor_kl`、`actor_kl_backtracks` 和 `actor_updates_rejected` 记录实际接受的更新及回退，尤其用于避免低方差预训练策略在第一次续训时跳变。

`mappo.actor_minibatch_steps` 可独立设置 Actor 批量，默认沿用 Critic 的 `minibatch_steps`。`channel-mappo-fullbatch.yaml` 在每个 Actor 更新中使用整个 1,024 步 rollout，使首次更新就覆盖所有状态，避免小批量在 KL 提前停止前遗漏稀少的碰撞轨迹；Critic 仍用 64 步批量。`actor_samples_processed` 统计包含重复 epoch 在内的使用次数，不能视为新增环境样本。该设置保持原有奖励、概率比、指导目标及 KL 上限；性能是否改善仍由连续验证决定。

`train/qualify_policy.py --checkpoint <模型> --baseline <同协议基线.json> --metrics <metrics1.csv> --output <qualification.json>` 检查默认连续 3 个 MAPPO 验证点均满足门槛，随后仅用分散式 Actor 重跑基线的开发集和独立资格集。脚本核对环境、艇数、攻击者敏捷度、基线权重哈希及配对种子，并在两组均满足成功率不降低、碰撞率严格降低时设置 `performance_qualified: true`。它不会启动正式实验。该准入判断不等同于多训练种子的统计显著性结论。

继续降低碰撞率时，可使用 `channel-mappo-refine.yaml` 或带终止事件校准的 `channel-mappo-calibrated.yaml`，通过 `--initialize-model <已训练模型>` 继承 Actor、V、Q 和目标 V，重新建立优化器并记录来源。接近冲突的优先查询只影响训练指导样本：根据相对位置、速度的短时最近接近距离，在每个 rollout 中保留有界数量的状态快照；主轨迹、奖励和行为概率保持原样。`guidance.batch_mode: validated` 让每次 Actor 更新使用当前 rollout 中通过分支检验的全部指导目标，按有效权重归一化，避免稀少指导样本在 KL 提前停止前完全未被抽中。指导仍受完整 rollout 的 KL 回退约束。

`guidance.priority_lead_time` 可把优先查询移到预计进入碰撞半径之前。例如 `channel-mappo-anticipatory.yaml` 使用 1.5 秒提前量；排名依据相对运动的直线预测，进入半径时间区别于最近接近时间。这只是训练查询启发式，真实效果仍用原非线性船模检验，不是执行时的避碰约束。默认值 0 保留原查询顺序。

`channel-mappo-collision-curriculum.yaml` 增加有界的训练初始场景重访：仅在训练回合发生碰撞时保存其初始化种子，随后以 50% 概率重访，其他回合重新抽样。重新执行时使用当前 Actor、新的动作采样和海流；保存的不是轨迹、概率或优势，不把旧经验插入 PPO。连续三次成功防御后移除该场景，突破会中断这个成功序列。种子和计数随检查点保存，`collision_revisit_episodes`、`collision_start_cases` 记录训练分布；默认关闭时保持原抽样过程。验证不使用这个场景池。正式算法对照应统一这种训练课程及预算，不能把课程收益全部归因于网络结构。

`mappo.terminal_q_updates` 开启仅供 Q 使用的终止事件校准。分别有界保存突破、碰撞、捕获和到时拒止的真实状态、实际执行动作及团队平均奖励，按终止类型平衡抽样，并混合当前在策略 Q 样本约束其他状态。这些已观测终止转移的回报标签为 `r`，没有后续自举项；海流具有随机性，因此单个标签不等于精确的期望 `Q(s,u)`。按结果类型重采样也是额外校准目标，不能称为原 Bellman 目标的无偏估计。它们不进入 PPO 或 V 的时间序列。`guidance.require_terminal_calibration` 要求已见到碰撞和捕获样本且校准误差达标后再启用指导。训练标签拟合达标仍需分支检验与独立 Actor 验证，不代表 Q 在未见状态上必然准确。

`channel-mappo-collision-refine.yaml` 在普通 Q 更新中以 `terminal_q_interleave_weight: 0.25` 穿插同一类终止回报样本，减少两个训练阶段交替造成的遗忘；`terminal_q_full_probe: true` 使用全部有界终止标签、各结果类型相同权重计算校准误差，避免小样本探针随机决定指导的启停。这些仍是训练误差，后续指导必须通过模拟分支验证。该配置还开启 `selection_objective: collision`：开发验证成功率达到原 ARBoids 的 120/128 后，先按碰撞数选检查点，再比较成功率及回报；其他协议须重新指定对应的 `selection_success_floor`。

进一步降低碰撞的 `channel-mappo-collision-cost.yaml` 增加可关闭的训练成本项。真实回合碰撞终止为成本 1，其余时间步为 0；单独的团队成本 V 估计当前策略在剩余回合的碰撞概率。成本 GAE 沿各环境独立计算，折扣率为 1、`cost_gae_lambda` 为 0.99，真实终止清零自举，rollout 边界使用成本 V 自举。原始任务奖励、任务 V/Q 的回归目标和环境终止规则不变。

Actor 在原 PPO、团队指导和熵损失之外，加入 `collision_cost_coefficient` 乘以成本 PPO 损失。成本损失使用 `max(ratio*A_cost, clip(ratio)*A_cost)` 的保守上界，最小化真实碰撞成本；成本优势保留概率单位，不套用任务优势的归一化。当前试验系数为 20，默认 0 关闭。全部 Actor 更新仍受原 KL 限制；成本网络仅参与训练，分散执行仍是原双通道 Actor。它是额外的训练目标，不是硬安全保证，也不应作为原方法结构的贡献。正式关键对照须统一成本目标、课程和预训练条件；成功率门槛与独立验证继续适用。

这些配置是训练试验选项，不能因代码已实现而视为性能已达标。首次 1,024 步采样的成本训练在开发集上没有稳定降低碰撞。`channel-mappo-collision-batch.yaml` 改用每轮 8,192 步采样及 Actor 批量、256 步 Critic 批量，每 32,768 步验证，目的是降低已观测到的策略梯度方差；是否保留仍取决于完整回合结果。不要用这些未通过资格验证的模型替换此前已达标权重。

已用于分析或调参的资格集应转为开发信息。下一阶段固定新的资格种子，并可在准入脚本增加 `--reference-checkpoint <上次达标模型>`：除成功率不低于原 ARBoids、碰撞率低于原 ARBoids 外，进一步要求碰撞率低于上次模型，并检查连续训练点。上次模型的成功率仍完整报告，但不替换用户指定的原 ARBoids 成功率门槛。评估摘要报告碰撞率的 Wilson 95% 区间，有限样本中的零碰撞也不被解释为真实碰撞概率为零。

`--max-collision-rate 0.01` 进一步要求连续开发检查和两组最终评估的观察碰撞率均不超过 1%。开发检查超限时不读取新模型在资格集上的表现；这仍是有限样本的准入门槛，不能解释为真实风险的确定上限。

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
