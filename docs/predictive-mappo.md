# 预测驱动的两阶段 MAPPO

此实现使用固定三自由度模型预测候选动作的交互，通过可训练关系编码和协调头影响随机 Adapter，再用任务优势与碰撞成本优势训练提案和门控。防御艇共享 Actor；Critic 集中训练，部署不使用 Critic。

## 控制链路

一次控制周期为：观测 → 各艇生成 Boids/学习候选 → 同步交换消息 → 名义运动推演 → 艇对关系编码及协调 → 采样门控 → 执行融合推力。

- `train/policy/mappo.py::PredictiveMAPPO.act` 是训练、评估和 VRX 的共同策略入口。
- 原始学习动作 `v` 服从二维高斯，归一化候选为 `tanh(v)`。
- 消息包括艇 ID、时间戳、`[x,y,yaw,vx,vy,yaw_rate]`、两个归一化候选。第一版采用同步可靠的全队交换，不含延迟、丢包或带宽优化。
- `InteractionPredictor` 从测得的地速初始化零海流名义模型，每艇保持两个候选推力分别推演。默认窗口 2 秒，子步 0.05 秒；积分顺序与原 `WAMV` 相同。不会读取未来扰动，也不会推进真实环境。
- 每个有向艇对有 18 个原始特征：BB、BL、LB、LL 各自的最近距离、最近时刻、接近趋势，加本艇坐标系下的相对位置、相对速度、相对航向正弦/余弦。距离、时间和速度使用固定尺度。
- 关系编码和 masked attention 产生 `z_pred`。共享成对网络产生 `eta_ij = sigmoid(g(e_ij)-g(e_ji))`，再汇总为 `z_coord`。双方协调信号互补；这不是全队无碰撞保证。
- 默认 Adapter 输入为状态特征 512 维、动作编码 128 维、关系特征 64 维、协调特征 16 维。输出门控高斯的均值和对数标准差。
- `w` 为门控原始采样，`theta = sigmoid(w)`。测试用两个高斯的均值作确定性决策。

推进器执行保持原公式：`F_exec = theta * F_L + (1-theta) * F_B`。
网络动作与推力的换算为 `F = 750*a + 250`，因此归一化动作零不是零推力。预测器与执行器使用相同换算。

## 概率梯度和训练目标

明确采用**分因子裁剪的 MAPPO 扩展**，并非未经修改的标准 MAPPO：提案与条件门控分别构造 PPO ratio，各自裁剪，再求和平均；二者使用相同的团队综合优势。这是局部因子 surrogate，不等价于对全队联合概率比进行裁剪。

PPO 数据保存原始采样 `v,w`、旧对数概率、观测、候选消息、原始艇对特征与 mask。更新时固定这些采样和消息，只重新计算当前状态编码、关系编码、协调头和门控分布；不重新采样候选，不复用旧的神经网络隐特征。

概率统一定义在原始高斯变量上；固定 tanh/sigmoid 变换的雅可比在新旧 ratio 中抵消。探索正则使用原始高斯的熵，不将其称为有界动作或 theta 的熵。没有 SAC 的 Q 动作梯度、温度更新、target Q、旧数据回放或额外 Adapter 噪声。

集中式 Critic 输入所有艇的位置、航向、相对水流速度、攻击艇机动参数与剩余时间，不接收本次提案或门控。它有独立的任务/成本价值头，与 Actor 不共享参数。

任务奖励是各艇 `(main + formation)` 的均值，移除原固定碰撞惩罚。首次实际队内碰撞为团队成本 1，一个回合最多一次；独立读取物理事件，因而同时发生突破与碰撞时仍计成本。

默认 `reward_gamma=cost_gamma=1`，对应有限回合任务收益和碰撞事件概率。两套 GAE 使用各自价值输出，先计算 `A = A_reward - lambda*A_cost`，然后统一标准化 A，不分别标准化两个优势。GAE 的 lambda 与碰撞约束乘子是不同参数。

一个 rollout 收集固定行为策略下的一批完整回合，保留每一帧的全队维度。所有 PPO epoch 使用同一个碰撞乘子。更新结束后仅执行一次：

```text
collision_rate = 碰撞回合数 / 完整回合数
lambda_next = max(0, lambda + lagrange_lr * (collision_rate - collision_budget))
```

没有使用预测风险替代真实成本，没有按艇数重复累计成本。`done=4` 是任务时限到达，在当前任务中计守时成功，并按真实终点令 bootstrap 为零。第一版不在回合中间切断采样，因此总步数可能超出设定上限一个完整 rollout。

## 代码位置

| 文件 | 职责 |
|---|---|
| `train/envs/mappo_env.py` | 任务奖励、独立事件成本和终点接口，保留旧 SAC/攻击艇接口 |
| `train/envs/TADgame.py` | 运动快照、集中状态、奖励分量、独立物理事件 |
| `train/policy/interaction_prediction.py` | 同步候选消息、向量化名义动力学和有向艇对特征 |
| `train/policy/mappo_networks.py` | 提案、关系、协调、随机 Adapter、集中式双价值头 |
| `train/policy/rollout_buffer.py` | 完整回合缓存和两套 GAE |
| `train/policy/mappo.py` | 采样、旧动作评价、分因子 PPO、乘子、完整 checkpoint |
| `train/train_mappo.py` | 训练、回合采样和确定性评估 |
| `vrx/mappo_controller.py` | 复用相同策略的部署适配，不依赖 ROS 即可测试 |

此分支移除了未被当前入口使用的旧 `train/Utils` 重复索引项，统一使用 `train/utils`，解决 Windows 无法区分仅大小写不同目录的检出冲突。

## 运行

从当前工作树根目录执行，使用已有 Python 环境即可。独立 worktree 可使用原仓库的虚拟环境解释器；源代码和输出仍来自当前工作树。

```powershell
& 'D:\ARBoids\.venv\Scripts\python.exe' -X utf8 -u train/train_mappo.py --device cpu --seed 42 --max-updates 2 --episodes-per-update 2 --eval-episodes 2 --run-id mappo-smoke
```

正式训练使用 `train/configs/mappo-prediction.yaml`，去掉 smoke 限制。配置中的 5% 碰撞预算只是可运行示例，不是实验选择的最优值；正式比较前应按任务要求固定预算，并在所有对应 MAPPO 对照中统一任务奖励、折扣、环境协议和成功定义。

```bash
python -X utf8 -u train/train_mappo.py --device cuda:0 --seed 42 --run-id mappo-seed42
```

checkpoint 包含网络、优化器、乘子、配置、步数、更新次数和随机数状态。恢复时使用保存的配置，允许修改步数/回合数等运行上限，不允许悄悄改变约束预算。`--max-updates` 是包含已完成更新在内的总更新数。

```bash
python -X utf8 -u train/train_mappo.py --resume train/experiments/mappo-smoke/predictive-mappo.pth --device cpu --max-updates 3 --run-id mappo-smoke
python -X utf8 -u train/evaluate_policy.py --checkpoint train/experiments/mappo-smoke/predictive-mappo.pth --episodes 100 --seed 10000 --output-dir train/experiments/mappo-smoke/evaluation
```

评估自动识别新 checkpoint，并使用其配置。输出同时区分捕获成功、守时成功、突破、队内碰撞和任务回报；预测危险分数不参与碰撞率统计。训练内评估会恢复随机数状态，避免改变后续训练轨迹。

VRX 需要原有 ROS 2/Gazebo 环境。在激活其运行环境后：

```bash
python -X utf8 -u vrx/run_experiment.py --controller MAPPO --termination-rule paper --checkpoint train/experiments/mappo-smoke/predictive-mappo.pth --headless
```

Windows 启动器支持 `-Controller MAPPO -TerminationRule paper`；批量启动器同样支持 `--controller MAPPO`。部署保留既有推进器左右映射，增加由反馈估计的航向角速度，将反馈外推到同一时间戳后执行本地候选交换。轨迹保存反馈时间差和控制计算耗时，仿真时钟在计算时继续推进；这是同步消息的集中仿真实现，不是实体艇网络通信实现。

## 验证

```bash
python -X utf8 -m unittest discover -s train/tests -v
```

测试覆盖原有论文协议、预测与原动力学逐子步一致性、无状态/RNG副作用、艇对方向与置换、旧概率重算、关系/协调到门控的梯度、两阶段实际更新、终点 bootstrap、首次碰撞计数、融合端点、乘子方向、checkpoint 恢复和部署动作一致性。短训练用于验证程序与梯度链路，收敛和性能比较需要预先确定预算后进行完整训练及独立种子评估。
