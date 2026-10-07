# 原始源码上的预测与联合门控贡献实验

本轮重新评测“候选轨迹预测与全队联合门控选择”。此前论文参数版环境和
MAPPO/门控结果不并入本轮统计。所有方法从作者发布的同一个原始 Actor、
Boids 和环境出发；消融只改变风险评价和门控选择方式。

## 固定原始对照

- 作者仓库：<https://github.com/taojy687/ARBoids>。
- 发布提交：`9eb6df464808af6b9729f39644b4d658e1695e2d`。
- `third_party/arboids_release/` 中的源码逐字节取自该 Git 提交；
  `provenance.json` 固定每个文件的 SHA-256。程序启动、结束均检查。
- 使用既有 `main-seed42-20260921/adares1.pth` 原始协议重训权重，SHA-256：
  `fe0c34101379582eac83ec6e89b4090d754308b604a97babf777f5b01d6c6ca2`。
  这是本地重训权重，不称为作者发布权重。训练配置必须与作者原 YAML
  的解析内容完全相同；论文参数版和 MAPPO 配置直接被拒绝。
- 数值仿真直接调用作者原始 `TADEnv.step`、奖励和终止函数。保留 80 秒上限、
  原始随机海流、5 米碰撞/捕获阈值，以及“攻击艇比所有防守艇更靠近目标”
  的提前失守规则。另行记录实际目标突破，避免把提前失守误写为目标突破。
- 每个正式场景额外执行一遍不经过任何新增控制器的原始循环，逐步核对
  动作、运动状态、观测、奖励和终止码的完整轨迹哈希。

## 六组方法

| 名称 | 风险信息 | 决策变量与选择方式 |
|---|---|---|
| Original | 作者原始策略 | 原始 Adapter 权重；不调用新增预测器 |
| Reactive independent | 当前状态的二阶屏障残差 | 每艇单独选权，队友保持原权重；所有艇同时执行 |
| Reactive joint | 同上 | 在全队组合上联合最小化同一目标 |
| Predictive independent | 2 秒名义候选轨迹的最小艇间距 | 每艇单独选权，队友保持原权重；所有艇同时执行 |
| Predictive joint | 同上 | 在全队组合上联合最小化同一目标 |
| CBFpy | 同一当前运动状态与动力学 | 官方 CBFpy 直接修正全队推力，不使用候选门控 |

所有门控组每 0.2 秒重新计算。每艇候选为原权重与 `{0, .25, .5, .75, 1}`，
保留原权重作为第一个候选，代价相同时不额外干预。预测组采用作者三自由度
积分器对应的零海流模型，步长 0.05 秒；用当前地速初始化，候选推力在预测
窗口内保持不变，只执行下一控制周期。

令 `p`、`v`、`a` 为艇对的相对位置、速度和候选瞬时加速度，预警距离
`d_s=7 m`。预测风险为 `[max(0, 1-d_min/d_s)]²`。瞬时消融不推演轨迹，使用
`h=(p·p-d_s²)/(2d_s²)`、`h_dot=p·v/d_s²`、
`h_ddot=(v·v+p·a)/d_s²`，风险为
`[max(0, -(h_ddot+2αh_dot+α²h)/α²)]²`，`α=1 s⁻¹`。

四个门控组均最小化最危险艇对的风险，加 `0.02` 倍归一化推力变化平方均值。
无风险时原样保留原始指令。独立组是同一全队目标的并行单艇最优响应，
不迭代、不获知队友最终改动；联合组穷举完整组合。
候选轨迹按单艇复用，三艇时是 18 条单艇轨迹和 216 个权重组合。

## 直接竞争基线

使用未修改的 [CBFpy 官方实现](https://github.com/StanfordASL/cbfpy)，版本
`0.0.4`，`qpax 0.1.4`，`jax/jaxlib 0.4.30`。依赖安装在工作区专用目录，
Windows 与 Linux 使用相同版本；完整依赖列表在
`train/requirements-gate-study.txt`。

任务适配仅实现官方 `CBFConfig` 接口：作者无人艇动力学、相同的全队运动
状态、7 米艇间屏障、增益 1、原始推力界限。控制变量按 1000 N 缩放；
目标是最小化相对原 ARBoids 指令的改动。自动 Lie 导数、CBF 转换和 QP
求解全部调用官方包；屏障/控制松弛惩罚分别为 1000/100000，求解容差
`1e-6`。记录饱和和约束残差。所有方法面对同一真实原始环境。
这是标准 CBF 安全滤波基线，不冒称为某篇 USV 论文的完整复现或 MPC。

CBFpy 在初始化时以所有状态分量为 1 检查 `Lgh`，这会使所有艇重合而提示
零梯度。测试另外验证实际分离状态下的有效约束、有限有界输出，以及 CBF
动力学与候选预测动力学相同；不修改官方检查或求解器来隐藏该提示。

## 固定的数值实验安排

- 机制/方法比较：三艇、敏捷度 2.25，256 个新场景，种子从 `129000000` 开始。
- 场景变化：艇数 `{2,3,4,5}` × 敏捷度 `{1.5,2,2.5,3}`，每格 32 个新场景，
  起始种子 `131000000`，不同配置相隔 10000。
- 所有六组都运行每个场景；配对相同初态与随机扰动流，场景内方法次序随机。
  主指标为原始任务成功、实际碰撞和团队回报；同时给出源码失守、实际突破、
  干预周期和耗时。
- 配对单位是场景。成功差使用精确 McNemar 检验；五项“完整方法对其他组”
  比较使用 Holm 校正。回报及成功差给出场景自助法区间。场景变化按配置分别
  报告，避免平均数掩盖艇数或敏捷度下的退化。

## VRX 源码与接口

VRX 的策略、Boids/APF、观测构造、初始化分布和终止规则直接调用作者发布版。
保留原始观测中的额外零填充邻居槽及 5.5 米捕获半径，默认原始 100 秒上限。
共同的 ROS/Gazebo 启动、资源路径解析、推力话题桥接、终止进程与记录属于
实验接口，所有方法一致；不把接口适配称为算法改进。传感速度处理也保留
作者原始回调。新增控制器仅额外使用同一组位姿消息计算的角速度。

原始控制器、新方法和两实例隔离的真实 Gazebo 接口验证已完成。
正式 VRX 固定 32 个初态：三艇码头 16 个、三艇开阔水域 8 个、四艇码头
8 个；前两种敏捷度 2.25，四艇配置敏捷度 2.5。比较原始 ARBoids、预测逐艇、
预测联合和 CBFpy，共 128 回合，种子从 `139000000` 开始，配置之间相隔
10000。每个方法在 Gazebo 启动前完成相同的推理预热，统一在仿真第 2 秒后
开始控制；同时核对相同采样初态和第一次控制时的实际艇位置。两套实例使用
独立 ROS domain 与 Gazebo partition；正式批次不启用相机以免改变启动条件。
数值仿真结果和 VRX 结果分别统计，不用历史其他控制器的 VRX 结果替代本轮。

实验设计工作流使用 Scientific Agent Skills 的 `experimental-design` 技能。
软件参考：Kassis, T., Agarwal, V., He, Y., Patel, D. & Brueckner, A. M. (2026).
Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents.
<https://doi.org/10.48550/arXiv.2609.00065>。

## 入口

```powershell
& 'D:\ARBoids\.venv\Scripts\python.exe' -X utf8 -u train/evaluate_gate_contribution.py `
  --checkpoint 'D:\ARBoids\train\experiments\main-seed42-20260921\adares1.pth' `
  --suite mechanism --episodes 256 --workers 4 --seed 129000000 `
  --output-dir train/experiments/gate-contribution-source-20261006/mechanism
```

`--suite generalization --episodes 32 --seed 131000000` 对应上面的场景变化安排。
运行目录保存具体配置、原始逐回合数据、统计表和 `passed` 校验结果。

VRX 在既有 `ARBoids-22.04` WSL 环境内运行，先加载
`/opt/arboids-runtime/vrx_ws/activate.bash`，再执行：

```bash
python -X utf8 -u vrx/evaluate_gate_contribution.py \
  --checkpoint /mnt/d/ARBoids/train/experiments/main-seed42-20260921/adares1.pth \
  --assets-dir /mnt/d/ARBoids/.vrx-assets \
  --output-dir train/experiments/gate-contribution-source-20261006/vrx \
  --workers 2 --dock-scenes 16 --ocean-scenes 8 --team-scenes 8 --seed 139000000
```

`--resume` 只接续相同固定配置的批次，复用已经完成的有效回合。仅当 Gazebo
在执行第一个控制指令之前启动失败时，允许同一方法、同一初态再启动一次；
不会因为碰撞或失守结局重跑。失败的启动结果保留在原运行目录。

批次另行检查每组四种方法第一次控制时的实际位置差是否小于 `1e-7 m`。
如该固定标准未通过，`--resume --repair-pairing` 以相同初态和参数重跑整组
四种方法，最多尝试两组；接受标准只使用起点一致性，不使用任务结局。
原回合目录保留，最终逐回合表注明替换关系，汇总注明被替换的初态组。
最小艇间距包含终止瞬间的位置，避免控制日志最后一帧早于实际碰撞时刻。

数值两部分通过校验且 VRX 原定回合运行完毕后，生成报告、四张统计图及
配对检验；VRX 的严格配对子集由相同的起点标准确定：

```powershell
& 'D:\ARBoids\.venv\Scripts\python.exe' -X utf8 train/figures/gen_fig_gate_contribution.py `
  --root train/experiments/gate-contribution-source-20261006 --include-vrx
```

本次实际完成 32 个 VRX 初态、128 个方法回合。4 组首次控制位置差为
3.83–8.43 mm，完整批次的严格配对校验未通过。对首组进行两次原方式整组
恢复后，另通过 `--synchronized-start-repair` 验证“所有模型生成后统一启动”
的共享接口；偏差降至 0.976 μm，仍高于预设阈值，所有恢复回合均未入统计。

最终报告并列展示完整 32 组的描述性结果，以及通过原定标准的 28 组严格
配对结果。图表脚本验证每个回合的源码和权重后，以起点条件筛选完整四方法
组，在 `episodes.csv` 标注是否纳入；`paired_summary.json` 只表示该子集的
校验结果。完整批次的 `summary.json` 保留失败状态。筛选不使用任务结局，
两种统计范围内各方法的成功数均相同。

可额外传入 `--case-examples` 重放两组说明性案例并生成艇间距曲线。案例按
碰撞转成功集合内的回报改善中位数选取，每条轨迹必须与正式评测的完整哈希
相同。它们只用于图示，不增加统计样本量。
