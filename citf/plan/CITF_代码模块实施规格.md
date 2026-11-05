# CITF代码模块实施规格

## 适用方式

本文件描述模块职责与可复用接口，服务于完整方案的收益目标。源码和实际产物决定实现状态；按下一项研究实验的依赖补齐缺口，优先复用已有实现。网络、控制、训练和数据接口可随新方法调整，更新相应运行配置。

## 共同任务与产物

作业记录方法、代码、源与模型、数据、任务角色、有效协议、对手、资源和真实状态，沿用现有protocol/runtime/manifest。环境步按真实共同控制转移计数，行为访问、分支采样、策略交互、梯度更新与墙钟分别记录。

共同环境已改为`citf-defense80-v3`：APF固定影响半径并累计全部有效障碍；以实际突破、碰撞、捕获或80秒到时终止，径向劣势保留为诊断。`citf.environment`统一真实环境与归档源码适配，CPU/GPU预测按协议选择相同APF和终止规则，分支标签及完整回合回报由实际终局生成。旧`citf-source80-v2`显式保留原有规则，历史结果按原任务解释。

旧任务库通过`python -m citf.tasks --from-bank <旧任务库> --registry <布局注册表> --out <新任务库>`切换为新环境，布局、角色、随机未来与任务编号保持。新训练与比较使用同一新库；当前远端归档作业按原代码完成。

## A　源策略训练与基线适配

源训练接口可采用`train_source(config, checkpoint=None)`，输入协议、算法、种子、预算、验证任务与输出位置；保存策略和训练所需网络、优化状态、随机状态、采样恢复位置、阶段模型与验证记录。源训练预算与是否联合微调按完整方案决定。

统一控制入口返回实际物理推力与必要trace。ARBoids使用其Adapter定义，RP使用对应残差形式，SAC和Boids保留自身方法。记录观测、动作和物理单位映射。CI-original保留内部历史身份；既有CITF-Phys使用原无学习预测控制器，新机制对照随最终方案定义。

原规划路径为`baselines/source_train.py`与`baselines/legacy_controller.py`；按实际仓库中可复用入口落实，文件名可调整。正式产物为各独立运行权重、有效配置、训练/验证记录及任务结果。

现有独立源训练入口为`train/mechanism_controls.py::run_author_source_repeat`。2026-11-05配方显式记录SAC观测变体及固定验证任务，旧结果保留原配方身份。VRX从权重恢复Actor输入维度；公开算法配对统计使用`code/analysis/paired_stats.py --comparison algorithm`。

## B　MAPPO强基线

参考共享Actor、集中Critic和连续动作MAPPO。环境返回艇队观测、动作、团队终止和事件；明确任务真实终止与采样截断的bootstrap语义。tanh动作变换、对数密度、GAE、PPO和价值归一化按实际方法实现，状态随训练恢复。

原规划路径`baselines/mappo_train.py`及适配器按需要补齐。采用作者实现或现有等价实现，合理调参并记录训练成本、适配及最终任务结果。

普通MAPPO/MAAC实现集中在`citf.baselines`，原`train/policy`保留导入接口。`python -m experiments.public_baselines train --algorithm MAPPO|MAAC --project <项目> --tasks <训练库> --validation <验证库> --registry <注册表> --out <输出>`完成训练、固定验证选模及恢复；`--resume`保存和恢复优化器、价值归一化、随机状态、环境和MAAC回放。默认沿用源奖励，可用`--reward terminal`登记完整任务奖励对照。

`python -m experiments.public_baselines evaluate --project <项目> --checkpoint <权重> --tasks <共同评价库> --registry <注册表> --out <输出>`复用共同任务、逐步记录、恢复和正式冻结入口。输入为当前可测状态，直接输出双推进器推力。`train/train_mappo.py`、旧`formal_study_evaluation.py --kind mappo`和`vrx/mappo_controller.py`继续按实际PredictiveMAPPO身份使用。新入口尚无训练成绩。

## C　实验调度与恢复

`experiments/matrix.py`已实现依赖矩阵调度。现有研究作业按实际状态接续，资源随本机或远端预算配置。下游依赖对应真实产物，恢复已完成任务与未完成任务的进度。

评价沿用任务ID和原子提交记录，保留成功、失败、早停和重跑原因。源码或实验条件变化时记录有效版本。按当前研究需要扩展调度能力。

## D　联合与独立选择

`experiments/independent_select.py`已实现共同源中心上的一次并行独立选择：各艇枚举仅替换自身的完整配置，分别选优后组装。六艇9候选最多49个独特配置，实际评分量随去重记录。

新动作域或规划方式下，选择能检验联合决策作用的对应简化策略；记录相同任务、可用信息、控制条件与实际成本。

## E　候选信息与学习机制

需要定位交互信息作用时，可固定模型、目标动作及历史，替换其他艇候选及对应物理特征，比较预测、动作和任务后果。原规划`experiments/context_probe.py`可按需实现或复用等价入口。

最终核心消融覆盖实际保留的学习链。改变预测、价值或策略目标时，保留对照需要的其余训练能力；若采用教师，纳入教师的训练和机制。模块诊断与完整方法比较各自记录实际研究对象。

`continuous_episode_training`支持`--reference-mode teacher|source|direct`。`direct`用固定250 N作为归一化原点，创新策略独立控制；`--teacher`与`--warmstart-episodes`可先采集训练布局的教师轨迹，预热动作与价值，再用真实完整回合优化。教师数据、环境步及是否在线调用分别记录；失败重放是可选课程。训练恢复绑定有效数据、源、配置和实现，历史窗口按配置构造。`continuous_actor_critic_training`保存每次更新及最终权重，并支持价值预热阶段恢复。

安全投影的普通、批量差分与原生实现共用实际剩余任务时长，显式长时域参数生效。原生接口校验形状与内存维度；有限优化迭代包含原始动作，数值失败记录原因并保留最佳有限解。联合覆盖与持续角色控制复用同一修复。

## F　学习型攻击者

`citf.opponents`已支持冻结对手执行；原规划训练入口为`experiments/opponent_train.py`。原接口为8维观测到归一化双推进器动作，实际动作经agility映射进入`World.step`。原奖励调用`attacker_training_reward`，新对手设计可调整奖励、网络和训练方式并记录配置。

采样、每条真实分支与评价使用指定对手在各自状态下反馈。需要RNN或随机策略时，一并提供隐藏状态、随机状态和分支恢复。对手池、训练预算、防御适配与共同任务遵循最终研究设计。

产物为真实训练记录、阶段和最终对手、观测/动作规范、对应防御适配模型及任务结果。

## G　VRX适配

原规划入口为`vrx/adapter.py`。将观测转换为实际模型所需状态和时间，明确ENU/NED、航向、速度、推进器左右与牛顿映射。控制输入来自当前可用传感信息。

以实际仿真时钟调度，记录sensor_time、decision_start、decision_finish、actuation_time。原参考为每0.2秒决策，超期保持上一合法指令并丢弃过期控制，初始统一零推力；仿真持续推进。根据最终部署方案统一各方法的执行条件。

正式参考为验证选择的中位运行、100共同场景，保存全部任务、时延、事件、轨迹和真实画面。开发适配按当前依赖开展。

## H　统计与论文结果

`analysis/paired_stats.py`已实现配对统计。任务按实际方法、运行、布局和条件关联；缺失和早停保留实际分母。共享布局跨运行的实验使用交叉重采样，同一次布局抽样在所选运行间共享。

原规划`analysis/paper_results.py`及图表生成按最终论文需要补齐。阶段曲线使用真实阶段权重，主表使用验证选定模型。表图关联结果集合、计算公式与样本单位；方法角色由`release/method_registry.json`与最终定义更新。行为图使用真实轨迹。

## 状态与执行

已有核心训练、评价、事件记录、调度、独立选择及配对统计可直接复用。正式源训练、基线、学习型对手、VRX和结果生成按实际缺口推进。`delivery_status.json`随真实产物更新。必要核对结合当前研究完成；工程测试与自动验证遵循个人AGENTS及用户明确要求。
