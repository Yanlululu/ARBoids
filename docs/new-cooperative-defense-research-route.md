> 当前研究方案按用户指定版本恢复：候选控制交互条件化的协同 Adapter，以及单艇融合干预驱动的联合条件价值学习。后续分层组成学生、双层 Actor–Critic 和预测直接监督 Actor 的方案不再作为研发主线。预测相关代码、控制器和实验结果暂时全部保留，作为独立工具与对照。
>
> 下文保留指定方案的完整内容。文中的既有数值继续归属于各自方法与原实验协议；当前 60 秒协议及后续正、负实验事实以 `results.md` 为准。恢复方案不代表尚未完成的学习证据已经成立，也不撤销已确认的执行一致性修复。
>
> 对照口径：原论文基线是未加 CBF 的 ARBoids。CBF 最初为早期预测门控方案的额外竞争基线，后来被沿用为执行层；它不是原文模块，也不是当前两项创新的内部学习机制。现有带 CBF 的实现与消融保持同一执行层，CBF 对照单独标明用途，不设为必须全面超越的唯一主基线。原法主比较衡量完整方法差异，两项创新的收益由匹配消融识别。

# 新论文的总体定位

**建议将新论文确定为：**

## 《面向高机动目标防御的多无人艇交互感知协同残差强化学习》

**Interaction-Aware Cooperative Residual Reinforcement Learning for Multi-USV Target Defense**

论文首先研究一个独立的任务问题：

> **在攻击者具有机动优势、艇间交互随拦截过程增强的条件下，如何学习兼顾区域防护、机动协调与捕获效率的多无人艇联合控制策略，使防守方的数量与空间分布优势转化为有效的团队防御能力？**

随后提出方法：**通过交互感知的自适应融合，协调规则控制与学习控制在不同防守艇上的作用；通过针对融合决策的条件价值学习，使这种协调能够从任务反馈中学出来。**

这与ARBoids的研究组织方式一致：**先提出任务中的控制需求，再选择能够回应这些需求的学习机制，最后以对应实验支撑方法。**新文章的主语是“多无人艇协同防御与策略学习”，ARBoids是重要相关工作和对照，而不是每一节默认的叙述前提。

下面将原文分析、新方法公式、完整文章大纲、结果安排及引言文献统一起来。文中新增网络和训练公式属于拟定方法；已有实验仍归属于其实际运行的方法，不改变结果名称。

---

# 一、ARBoids究竟是怎样形成一篇论文的？

能够从正式期刊版还原的是作者公开呈现的论证过程，而不是未披露的实际研发顺序。其论证并非“找到一个缺失模块，然后补上”，而是完成了五次逐步收敛。

## 1. 从应用领域收敛到具有挑战性的任务条件

原文首先定义目标防御任务：防守艇在攻击艇进入保护区域之前实施拦截。随后突出攻击艇具有更强机动能力这一条件，由此提出防守策略需要同时具备协作与动态适应能力。2026_ARBoids_Adaptive_Residual\_… 2026_ARBoids_Adaptive_Residual\_…

它并没有以“现有Actor少了一个输入”开篇。**高机动对抗是问题背景，合作与适应是控制需求。**

新论文对应地突出：高机动对抗中，多艇需要通过联合机动弥补个体能力不足；同时，拦截汇聚导致艇间行动相互影响，因此需要将局部控制组织为有效的团队行为。

## 2. 从众多已有方法收敛到一组互补能力

原文依次讨论解析方法、虚拟力方法、强化学习和残差强化学习。它并没有否定这些方法，而是提取其中的互补性：规则方法提供结构化的协调先验，强化学习提供根据交互经验调整行为的能力，残差框架则提供二者结合的入口。2026_ARBoids_Adaptive_Residual\_…

因此，新文章不需要证明此前所有方法都无效。应当说明：

> 规则控制和学习控制已经提供了有效基础，研究重点是如何使这些控制能力在团队交互中得到协调使用。

## 3. 从控制需求收敛到明确的决策机制

原文到了III-A，才将技术问题落实到两类控制的组成方式：

\[
u=u^L+u^B
\]

改为：

\[
u=\theta(s)u^L+[1-\theta(s)]u^B.
\]

其改变是允许控制组成随状态调整，融合权重由网络学习，而不是预先固定。2026_ARBoids_Adaptive_Residual\_…

**这一步是ARBoids的核心：把前面提出的“合作与适应”需求，落实为一个具有明确控制含义的可学习变量。**

新论文也应这样做，但变量的学术意义不再停留于“本艇使用多少规则控制”，而是：

> **本艇的融合比例如何依赖队友候选控制及运动关系，从而共同形成有利的拦截行为。**

## 4. 围绕变量补全信息、评价与训练

原文第4页图2说明：Adapter读取本艇两类动作及决策隐特征；Critic把学习动作与融合权重共同纳入评价；训练阶段通过权重扰动探索不同融合方式。2026_ARBoids_Adaptive_Residual\_… 2026_ARBoids_Adaptive_Residual\_…

因此，原文创新不是孤立的一个公式，而是：

\[
\text{控制组成变量}
\rightarrow
\text{变量所需信息}
\rightarrow
\text{变量的任务评价}
\rightarrow
\text{更新变量生成机制}.
\]

新机制也必须形成这条完整链条。

## 5. 实验分别回答方法主张

原文图4展示协同行为，图5展示学习和消融，图6考察敏捷度与艇数，图7考察学习型攻击者。实验不是按开发时间排列，而是按论点排列。2026_ARBoids_Adaptive_Residual\_… 2026_ARBoids_Adaptive_Residual\_…

可以概括为：

| ARBoids的研究推进 | 新论文的对应推进 |
|---|---|
| 高机动攻击者使目标防御困难 | 高机动对抗与密集交互共同增加联合控制难度 |
| 规则与学习需要动态结合 | 不同艇对规则与学习的使用需要相互协调 |
| 提出状态相关的融合变量 | 提出候选动作交互条件下的协同融合变量 |
| 通过Actor–Critic训练Adapter | 通过联合价值与融合干预评价训练协同Adapter |
| 验证学习、协同和任务适应能力 | 验证交互建模、条件价值学习和联合防御效果 |

---

# 二、研究范围与原文保持同一层级

原文采用SAC、Boids、均值嵌入和课程学习；没有证明策略全局最优；忽略风浪；VRX实验仍有碰撞；超过四艇后性能下降；真实艇实验放在未来工作。2026_ARBoids_Adaptive_Residual\_… 2026_ARBoids_Adaptive_Residual\_… 2026_ARBoids_Adaptive_Residual\_…

新论文据此限定即可：

| 范围问题 | 原文情况 | 新论文安排 |
|---|---|---|
| 使用已有基础算法 | 大量使用，创新集中在融合与训练 | 保留Boids和SAC，不要求重新发明基础算法 |
| 全局最优及严格无碰撞证明 | 没有给出 | 不作为本文额外研究任务 |
| 风浪、复杂海况 | 忽略 | 沿用相同范围 |
| 实艇验证 | 未完成 | 以数值训练和VRX验证为主体 |
| 任意艇数和所有对手 | 未覆盖，部分条件退化 | 按原文相同类型检验，不承诺无限外推 |
| 所有组件都必须单独原创 | 不是原文标准 | 评价完整新机制，而不是要求每个编码器原创 |

**这些问题不再扩展成全面补强清单。**本文需要完成的是自己的策略机制、训练方法与实验结论之间的对应关系。

---

# 三、拟定核心机制：究竟替换什么，保留什么？

## 1. 不改变任务，也不把“单艇权重”简单变成“多艇权重”

ARBoids本来就是多艇方法。它采用共享网络参数，各艇分别输出权重；共享参数不等于所有艇权重相同。原状态中也已经包含队友信息。2026_ARBoids_Adaptive_Residual\_… 2026_ARBoids_Adaptive_Residual\_…

所以，新机制不是：

\[
\text{一个权重}\rightarrow n\text{个权重}.
\]

而是：

\[
\boxed{
\text{群体状态下的个体融合}
\rightarrow
\text{候选动作交互条件下的协同融合学习}.
}
\]

## 2. 保留两类控制来源，改造融合策略及其训练依据

仍然使用：

\[
u_i=\theta_i a_i^L+(1-\theta_i)a_i^B.
\]

改造两处。

**策略侧：**让\(\theta_i\)显式依赖本艇与队友候选动作之间的关系，而不只直接读取本艇候选动作。

**训练侧：**评价整组融合权重的联合效果，并通过“只改变某艘艇当前权重”的成对后果，训练价值函数辨识该调整的条件作用。

因此，完整方法是：

\[
\boxed{
\text{候选控制交互表示}
\rightarrow
\text{协同自适应门控}
\rightarrow
\text{联合价值学习}
\rightarrow
\text{融合干预的后果监督}.
}
\]

其中，**协同自适应融合是主机制，后果推演是训练手段**。不再将“联合后果引导”单独用作任务名称。

## 3. 三项贡献按照原文层级组织

**第一项：交互感知的协同残差控制框架。**

面向高机动对抗条件下的目标防御，将队友候选控制与相对运动关系纳入自适应融合，使不同防守艇对规则控制和学习控制的使用服务于统一的团队任务。

**第二项：融合决策的条件价值学习机制。**

在联合Critic中评价整组融合决策，并利用局部门控干预的长期后果校准价值差，使训练能够区分各艇融合调整在队友动作条件下的任务作用。

**第三项：机制与任务表现的对应验证。**

通过同信息学习对照、模型辅助对照、短长时消融，以及敏捷度、艇数和学习型攻击者实验，评价协同融合的学习效果与防御能力。

这里不将通信、Boids、SAC、并行计算和安全过滤器分别包装成独立创新。

---

# 四、引言文献：与原文相近数量、相近刊物层级的21篇期刊文章

原文引言使用[1]—[21]，全文共28篇参考文献；引言文献本身包含期刊和会议，并非全部期刊。新稿可配置**21篇期刊文章作为引言主体文献**，方法部分再补SAC、反事实评价等基础来源。原文引用结构见其引言和参考文献。2026_ARBoids_Adaptive_Residual\_… 2026_ARBoids_Adaptive_Residual\_…

下面按实际刊物对齐，主要覆盖RA-L、T-RO、T-Cybernetics、TNNLS、TSMC、T-Mechatronics、IJRR、Automatica、JMLR等，不将未经核验的统一“一区”标签代替刊物信息。

## A. 任务背景、对抗几何与协同拦截：[1]—[7]

| 编号 | 期刊文章 | 在新引言中的作用 |
|---|---|---|
| [1] | **Qiao Y., et al.** “Survey of Deep Learning for Autonomous Surface Vehicles in Marine Environments.” *IEEE Transactions on Intelligent Transportation Systems*, 24(4):3678–3701, 2023. DOI: **10.1109/TITS.2023.3235911**。[IEEE Xplore](https://ieeexplore.ieee.org/document/10021250/) | 支撑海面自主系统中感知、决策与学习控制的研究背景 |
| [2] | **Fu H., Liu H. H.-T.** “Guarding a Territory Against an Intelligent Intruder: Strategy Design and Experimental Verification.” *IEEE/ASME Transactions on Mechatronics*, 25(4):1765–1772, 2020. DOI: **10.1109/TMECH.2020.2996901**。[飞行系统与控制](https://flight.utias.utoronto.ca/fsc/wp-content/uploads/2023/06/TMech2020FF.pdf) | 支撑目标防御、智能入侵者及团队拦截的任务定义 |
| [3] | **Chen Y., et al.** “A Distributed Pipeline for Collaborative Pursuit in the Target Guarding Problem.” *IEEE Robotics and Automation Letters*, 9(3):2064–2071, 2024. DOI: **10.1109/LRA.2024.3349977**。[IEEE Xplore](https://ieeexplore.ieee.org/document/10380689/) | 支撑目标守卫中的协同决策与分布式预测控制 |
| [4] | **Yan R., et al.** “Multiplayer Homicidal Chauffeur Reach-Avoid Games: A Pursuit Enclosure Function Approach.” *Automatica*, 167:111770, 2024. DOI: **10.1016/j.automatica.2024.111770**。[科学直通车](https://www.sciencedirect.com/science/article/pii/S0005109824002644) | 支撑受运动约束的多参与者到达—规避与围堵分析 |
| [5] | **Fang X., Wang C., Xie L., Chen J.** “Cooperative Pursuit With Multi-Pursuer and One Faster Free-Moving Evader.” *IEEE Transactions on Cybernetics*, 52(3):1405–1414, 2022。[卡内基梅隆大学机器人研究所出版物库](https://publications.ri.cmu.edu/cooperative-pursuit-with-multi-pursuer-and-one-faster-free-moving-evader) 2026_ARBoids_Adaptive_Residual\_… | 直接支持多追捕者对抗更快逃逸者这一任务条件 |
| [6] | **Guo X., Cui R., Yan W.** “Pursuit-Evasion Games of Marine Surface Vessels Using Neural Network-Based Control.” *IEEE Transactions on Systems, Man, and Cybernetics: Systems*, 55(1):18–27, 2025。2026_ARBoids_Adaptive_Residual\_… | 支撑海面艇追逃与神经网络控制研究 |
| [7] | **Zhou M., Wang Z., Wang J., Cao Z.** “Multi-Robot Collaborative Hunting in Cluttered Environments With Obstacle-Avoiding Voronoi Cells.” *IEEE/CAA Journal of Automatica Sinica*, 11(7):1643–1655, 2024. DOI: **10.1109/JAS.2023.124041**。[ieee-jas.net](https://www.ieee-jas.net/en/article/doi/10.1109/JAS.2023.124041) | 支撑空间组织、避碰与协同捕获之间的联系 |

注意，[2]和[5]已经研究更快的入侵者或逃逸者，不能在新引言中笼统写成“已有研究都假设防守方更快”。应当强调本文关注的**动力学控制、密集交互与策略学习的结合**。

## B. 强化学习追捕与海面艇决策：[8]—[11]

| 编号 | 期刊文章 | 在新引言中的作用 |
|---|---|---|
| [8] | **de Souza C., et al.** “Decentralized Multi-Agent Pursuit Using Deep Reinforcement Learning.” *IEEE Robotics and Automation Letters*, 6(3):4552–4559, 2021。[IEEE Xplore](https://ieeexplore.ieee.org/document/9387125) | 支撑共享经验、分散策略与协同追捕 |
| [9] | **Zhang R., et al.** “Game of Drones: Multi-UAV Pursuit-Evasion Game With Online Motion Planning by Deep Reinforcement Learning.” *IEEE Transactions on Neural Networks and Learning Systems*, 34(10):7900–7909, 2023. DOI: **10.1109/TNNLS.2022.3146976**。[IEEE Xplore](https://ieeexplore.ieee.org/document/9712866/) | 支撑多主体对抗中的在线决策和运动预测 |
| [10] | **Xue S., et al.** “Multi-agent Self-Attention Reinforcement Learning for Multi-USV Hunting Target.” *Neural Networks*, 189:107574, 2025. DOI: **10.1016/j.neunet.2025.107574**。[科学直通车](https://www.sciencedirect.com/science/article/pii/S0893608025004538) | 与多无人艇学习型围捕直接相关，支持交互表示的重要性 |
| [11] | **Lin X., Szenher P., Huang Y., Englot B.** “Distributional Reinforcement Learning Based Integrated Decision Making and Control for Autonomous Surface Vehicles.” *IEEE Robotics and Automation Letters*, 10(2):1194–1201, 2025. DOI: **10.1109/LRA.2024.3518839**。[IEEE Xplore](https://ieeexplore.ieee.org/document/10804093/) | 支撑海面艇决策与连续控制一体化 |

## C. 群体表示、联合价值和通信：[12]—[15]

| 编号 | 期刊文章 | 在新引言中的作用 |
|---|---|---|
| [12] | **Hüttenrauch M., Šošić A., Neumann G.** “Deep Reinforcement Learning for Swarm Systems.” *Journal of Machine Learning Research*, 20(54):1–31, 2019。[机器学习研究期刊](https://jmlr.org/papers/v20/18-476.html) | 支撑可变规模队伍的表示、共享策略与群体学习 |
| [13] | **Rashid T., et al.** “Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning.” *Journal of Machine Learning Research*, 21(178):1–51, 2020。[机器学习研究期刊](https://jmlr.org/papers/v21/20-081.html) | 支撑联合任务价值与多智能体策略学习 |
| [14] | **Ma Z., Luo Y., Pan J.** “Learning Selective Communication for Multi-Agent Path Finding.” *IEEE Robotics and Automation Letters*, 7(2):1455–1462, 2022. DOI: **10.1109/LRA.2021.3139145**。[HKU Scholars Hub](https://hub.hku.hk/handle/10722/319777) | 支撑根据决策需求组织队友信息，而非通信本身作为目的 |
| [15] | **Li Q., Lin W., Liu Z., Prorok A.** “Message-Aware Graph Attention Networks for Large-Scale Multi-Robot Path Planning.” *IEEE Robotics and Automation Letters*, 6(3):5533–5540, 2021. DOI: **10.1109/LRA.2021.3077863**。[IEEE Xplore](https://ieeexplore.ieee.org/document/9424371/) | 支撑消息内容和交互关系进入多机器人策略表示 |

## D. 控制先验与学习融合：[16]—[18]

| 编号 | 期刊文章 | 在新引言中的作用 |
|---|---|---|
| [16] | **Rana K., et al.** “Bayesian Controller Fusion: Leveraging Control Priors in Deep Reinforcement Learning for Robotics.” *The International Journal of Robotics Research*, 42(3):123–146, 2023. DOI: **10.1177/02783649231167210**。[Sage Journals](https://journals.sagepub.com/doi/full/10.1177/02783649231167210) | 支撑利用控制先验与学习策略互补的混合控制路线 |
| [17] | **Jeon S. H., Lee H. J., Hong S., Kim S.** “Residual MPC: Blending Reinforcement Learning With GPU-Parallelized Model Predictive Control.” *IEEE Transactions on Robotics*, 42, 2026。[Xplore](https://xplorestaging.ieee.org/author/37537397200) | 支撑残差学习与模型控制结合，而不是要求完全抛弃已有控制器 |
| [18] | **Tao J., Shen T., Zhao D., Zhang F.** “ARBoids: Adaptive Residual Reinforcement Learning With Boids Model for Cooperative Multi-USV Target Defense.” *IEEE Robotics and Automation Letters*, 11(3):3637–3644, 2026. DOI: **10.1109/LRA.2026.3662620**。2026_ARBoids_Adaptive_Residual\_… 2026_ARBoids_Adaptive_Residual\_… | 作为自适应融合目标防御的代表性工作，以及主要学习基线 |

## E. 长期评价、安全学习与海上预测：[19]—[21]

| 编号 | 期刊文章 | 在新引言中的作用 |
|---|---|---|
| [19] | **Bhattacharya S., et al.** “Multiagent Reinforcement Learning: Rollout and Policy Iteration for POMDP With Application to Multirobot Problems.” *IEEE Transactions on Robotics*, 40:2003–2023, 2024. DOI: **10.1109/TRO.2023.3347128**。[Arizona State University](https://asu.elsevierpure.com/en/publications/multiagent-reinforcement-learning-rollout-and-policy-iteration-fo/) | 为后续推演与学习策略的结合提供方法基础 |
| [20] | **Brunke L., et al.** “Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning.” *Annual Review of Control, Robotics, and Autonomous Systems*, 5:411–444, 2022. DOI: **10.1146/annurev-control-042920-020211**。[Annual Reviews](https://www.annualreviews.org/content/journals/10.1146/annurev-control-042920-020211) | 支撑任务学习与安全控制的分层组织 |
| [21] | **Yu J., et al.** “A Cooperative Hunting Method for Multi-USVs Based on Trajectory Prediction by OR-LSTM.” *IEEE Transactions on Vehicular Technology*, 73(12):18087–18101, 2024. DOI: **10.1109/TVT.2024.3432739**。[IEEE Xplore](https://ieeexplore.ieee.org/document/10607953/) | 支撑预测信息在多无人艇捕获中的作用，同时区分目标轨迹预测与本文的融合价值评价 |

这21篇足以形成与原文相近规模、相近刊物层级的引言支撑。其组织顺序是：

\[
\text{任务与对抗控制}
\rightarrow
\text{学习型协同}
\rightarrow
\text{群体信息与价值}
\rightarrow
\text{先验融合}
\rightarrow
\text{预测评价与安全执行}.
\]

方法部分另引用SAC、反事实多智能体评价和模型价值扩展，明确技术来源即可，不把它们当作否定新设计的理由。[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v80/haarnoja18b.html)

---

# 五、可直接发展为正文的引言

下面的编号与上述文献对应。

### 第一段：任务背景

多无人艇目标防御通过防守艇的联合行动阻止入侵者进入指定区域，是海上自主系统中具有代表性的对抗决策任务。相关研究已从海面艇学习控制、区域守卫和协同追捕等角度发展了多种解决方案[1]—[3]。与一般目标跟踪不同，目标防御要求控制策略同时考虑入侵者运动、保护区域位置和防守资源的空间分布，其效果取决于防守艇能否在动态过程中形成有效的联合拦截。[IEEE Xplore](https://ieeexplore.ieee.org/document/10021250/)

### 第二段：提出本文研究的困难条件

当入侵者具有机动优势时，防守方需要通过多方向接近、空间封堵与相互支持弥补个体控制能力的不足。对抗几何、到达—规避分析及协同围捕研究已经揭示了多主体空间关系对任务结果的重要作用[4]—[7]。与此同时，更多防守艇参与拦截会增加控制之间的相互影响：区域覆盖、局部避让与集中捕获需要在同一运动过程中协调完成。本文关注高机动对抗与密集交互共同作用下的协同目标防御，研究如何将多艇空间优势转化为持续、有效的团队机动。[科学直通车](https://www.sciencedirect.com/science/article/pii/S0005109824002644)

### 第三段：引入学习型协同控制

强化学习为上述问题提供了通过交互经验获得控制策略的途径。已有研究将深度强化学习用于分散追捕、多无人机对抗、多无人艇围捕及海面艇决策控制[8]—[11]。这些工作表明，复杂机动可以通过任务反馈形成，而不必完全依赖人工规定的控制序列。群体表示、联合价值学习和选择性信息交换进一步提供了处理多主体关系的技术基础[12]—[15]。[IEEE Xplore](https://ieeexplore.ieee.org/document/9387125)

### 第四段：从任务需求引出混合控制路线

规则控制能够提供具有明确结构的群体运动先验，学习控制则根据对抗状态调整局部机动。控制器融合、残差模型预测控制和自适应残差防御研究，为二者的结合提供了不同实现路径[16]—[18]。对于多艇系统，这种结合不仅需要适应个体状态，还需要协调不同防守艇对控制来源的使用，使局部自适应行为共同服务于区域防护和联合捕获。[Sage Journals](https://journals.sagepub.com/doi/full/10.1177/02783649231167210) 2026_ARBoids_Adaptive_Residual\_…

### 第五段：提出方法层的核心研究内容

为实现上述控制目标，本文将艇间候选控制关系纳入自适应融合，并研究与融合变量相对应的联合价值学习。不同于仅根据完整团队轨迹拟合策略价值，本文在相同起点下比较局部门控干预的后续任务结果，以辨识某次融合调整在队友控制条件下的作用。后续推演、模型辅助决策与安全控制已有相应研究基础[19]—[21]；本文利用这些基础构造针对融合决策的训练信息，而不将在线轨迹搜索作为最终策略的替代。[Arizona State University](https://asu.elsevierpure.com/en/publications/multiagent-reinforcement-learning-rollout-and-policy-iteration-fo/)

### 第六段：方法概述

本文提出交互感知的协同残差强化学习框架。该框架通过共享策略生成学习动作，并结合规则动作与艇间候选关系生成自适应融合权重。训练阶段使用联合Critic评价整组融合决策，通过门控干预的成对长期后果辅助价值学习；执行阶段，各艇根据可用观测和候选消息直接计算融合控制。该方法旨在统一规则先验的结构优势、学习控制的适应能力与团队机动的协调需求。

随后接前述三项贡献。**引言至此独立成立，不需要逐段重复“ARBoids未考虑什么”。**

---

# 六、完全对应原文的新文章大纲

| 原文结构 | 新文章结构 | 保留或改变的内容 |
|---|---|---|
| I. Introduction | I. Introduction | 同样从任务条件和控制需求提出研究问题 |
| II-A. USV Motion Model | II-A. USV Motion Model | 保留动力学与推进器控制 |
| II-B. USV Target Defense Problem | II-B. Cooperative Target Defense | 保留目标防御，明确成功、捕获与效率 |
| III-A. Adaptive Residual Policy | III-A. Interaction-Aware Cooperative Residual Policy | 改造融合权重的信息条件与联合决策表达 |
| III-B. Soft Actor Critic | III-B. Joint Actor–Critic Learning | 保留SAC，评价整组候选动作与权重 |
| III-C. Boids Model | III-C. Rule-Based Coordination Prior | 保留Boids四类规则 |
| III-D. State Representation | III-D. State and Candidate-Interaction Representation | 增加双方候选控制关系 |
| III-E. Network Structure | III-E. Cooperative Adapter and Conditional Value Learning | 给出新网络、成对后果及训练损失 |
| III-F. Reward Function | III-F. Task Reward | 主实验保留共同任务奖励，不另加复杂奖励系统 |
| III-G. Curriculum Learning | III-G. Curriculum and Training Procedure | 保留敏捷度课程，交代辅助训练调用 |
| IV-A. Experimental Setup | IV-A. Experimental Setup | 沿用环境、基础策略及评价协议 |
| IV-B. Experimental Results | IV-B. Cooperative Defense Results | 行为轨迹、门控变化、已有机制发现 |
| IV-C. Comparison With Benchmark Methods | IV-C. Learning Performance and Ablations | 基线、同信息对照和条件价值消融 |
| IV-D. Robustness Against Attacker Agility | IV-D. Attacker Agility Evaluation | 同类型评价 |
| IV-E. Generalization to Unseen Team Size | IV-E. Team-Size Generalization | 同类型评价 |
| IV-F. Robustness Against Learning-Based Attacker | IV-F. Learning-Based Attacker | 同层级扩展 |
| V. Conclusion | V. Conclusion | 回答研究问题、机制和证据 |

原文对应结构见II—IV各节。2026_ARBoids_Adaptive_Residual\_… 2026_ARBoids_Adaptive_Residual\_… 2026_ARBoids_Adaptive_Residual\_… 2026_ARBoids_Adaptive_Residual\_…

---

# 七、II. Problem Formulation：问题建模

## A. 无人艇运动模型

沿用平面三自由度训练模型：

\[
\dot{\boldsymbol\eta}_i
=
J(\psi_i)\boldsymbol\nu_i,
\]

\[
M\dot{\boldsymbol\nu}_i
+
C(\boldsymbol\nu_i)\boldsymbol\nu_i
+
D(\boldsymbol\nu_i)\boldsymbol\nu_i
=
B\boldsymbol\tau_i.
\tag{1}
\]

其中：

\[
\boldsymbol\eta_i=[x_i,y_i,\psi_i]^\top,\qquad
\boldsymbol\nu_i=[u_i,v_i,r_i]^\top,
\]

\[
\boldsymbol\tau_i=
[\tau_{i,L},\tau_{i,R}]^\top.
\]

\(B\)将左右推进器推力映射为纵向力和艏摇力矩。

该节属于共同物理基础。原文同样以WAM-V为对象，并在数值训练中保留纵荡、横荡和艏摇。2026_ARBoids_Adaptive_Residual\_… 2026_ARBoids_Adaptive_Residual\_…

攻击者机动系数继续采用推力极限之比：

\[
L_{\mathrm{agi}}
=
\frac{\tau_A^{\min}}{\tau_D^{\min}}
=
\frac{\tau_A^{\max}}{\tau_D^{\max}}.
\tag{2}
\]

它不是最大速度的直接比值。2026_ARBoids_Adaptive_Residual\_…

## B. 协同目标防御

设保护区域中心为原点。定义：

\[
d_{iA}=\|\boldsymbol p_i-\boldsymbol p_A\|,
\qquad
d_{ij}=\|\boldsymbol p_i-\boldsymbol p_j\|.
\]

捕获条件、目标区域和艇间碰撞规则继续采用共同任务协议。学习目标为：

\[
\max_{\Omega}
J(\Omega)
=
\mathbb E_{\Pi_\Omega}
\left[
\sum_{t=0}^{T-1}\gamma^t r_t^{\mathrm{team}}
\right].
\tag{3}
\]

任务表现分别统计防御成功率、捕获率、碰撞率和封顶耗时：

\[
T_e^{\mathrm{cap}}
=
\begin{cases}
t_e^{\mathrm{capture}},&\text{发生捕获},\\
T_{\max},&\text{未发生捕获}.
\end{cases}
\tag{4}
\]

原文正文采用60秒时限，当前报告采用80秒及源码提前失守条件。新稿在设置表中明确采用哪套协议即可；已有数值结果继续对应80秒协议，不将“源码失守”统一改称实际突破。2026_ARBoids_Adaptive_Residual\_… results

---

# 八、III. Method：新的创新机制及完整公式

## A. 交互感知的协同残差策略

### 1. 保留控制组成

每艇生成学习动作\(\boldsymbol a_i^L\)与规则动作\(\boldsymbol a_i^B\)：

\[
\boxed{
\boldsymbol\tau_i^{\mathrm{nom}}
=
\theta_i\boldsymbol a_i^L
+
(1-\theta_i)\boldsymbol a_i^B,
\quad \theta_i\in[0,1].
}
\tag{5}
\]

这是继承的混合控制形式，来源需要引用，不作为重复提出的公式创新。2026_ARBoids_Adaptive_Residual\_…

### 2. 改变权重的条件信息

新门控为：

\[
\boxed{
\theta_i
=
g_\psi
\left(
h_i,\boldsymbol a_i^L,\boldsymbol a_i^B,\boldsymbol m_i
\right),
}
\tag{6}
\]

其中，\(\boldsymbol m_i\)是由本艇与队友候选控制构成的交互表示。

原文图2直接连接本艇动作与本艇隐特征；新结构将队友本次候选动作的关系显式加入门控。**两者都有多个权重，改变的是权重生成的依赖关系。**2026_ARBoids_Adaptive_Residual\_…

### 3. 以联合价值定义权重的任务作用

记：

\[
\boldsymbol A^L=(\boldsymbol a_1^L,\ldots,\boldsymbol a_n^L),
\qquad
\boldsymbol\Theta=(\theta_1,\ldots,\theta_n).
\]

联合决策价值为：

\[
Q^\Pi(X,\boldsymbol A^L,\boldsymbol\Theta).
\]

本艇权重的作用以其他艇的选择为条件：

\[
\theta_i^\star(X;\boldsymbol\Theta_{-i})
\in
\arg\max_{\theta_i\in[0,1]}
Q^\Pi
\left(
X,\boldsymbol A^L,\theta_i,\boldsymbol\Theta_{-i}
\right).
\tag{7}
\]

式（7）用于说明学习对象，不表示执行阶段逐次求解全局优化。

当保留当前CBF时，实际控制为：

\[
\boldsymbol\tau^{\mathrm{exec}}
=
\mathcal F_{\mathrm{safe}}
\left(
X,\boldsymbol\tau^{\mathrm{nom}}
\right).
\tag{8}
\]

当前带 CBF 的完整方法及其机制消融使用相同过滤层。原始 ARBoids 对照不加 CBF；ARBoids＋CBF 单独列示，若还进行了继续训练，须同时标明训练预算。CBF 的独立作用不能归给两项学习创新，也不能将带 CBF 的版本称为原论文基线。式（8）描述现有实现采用的附加执行层，不构成两项创新必须依赖 CBF 的要求。

---

## B. 联合Actor–Critic学习

保留SAC作为基础，不再另行切换MAPPO。SAC提供连续动作、熵正则与离策略学习框架。[Proceedings of Machine Learning Research](https://proceedings.mlr.press/v80/haarnoja18b.html)

令一个控制周期的可学习决策为：

\[
z=(\boldsymbol A^L,\boldsymbol\Theta).
\]

策略分两阶段生成：

\[
\Pi_\Omega(z\mid X)
=
\prod_{i=1}^{n}
\pi_\omega^L(\boldsymbol a_i^L\mid o_i)
\prod_{i=1}^{n}
\pi_\psi^G(\theta_i\mid I_i).
\tag{9}
\]

先生成学习候选动作并交换允许的候选信息，再生成权重。这样不要求任何艇先知道队友尚未生成的最终权重。

采用联合软价值：

\[
V_{\bar\zeta}^{\Pi}(X)
=
\mathbb E_{z\sim\Pi}
\left[
\min_{\ell=1,2}Q_{\bar\zeta_\ell}(X,z)
-
\alpha\log\Pi_\Omega(z\mid X)
\right].
\tag{10}
\]

真实环境转移的目标值为：

\[
y_t
=
r_t^{\mathrm{team}}
+
\gamma(1-d_t)
V_{\bar\zeta}^{\Pi}(X_{t+1}).
\tag{11}
\]

策略损失为：

\[
\mathcal L_\Pi
=
\mathbb E
\left[
\alpha\log\Pi_\Omega(z\mid X)
-
\min_{\ell=1,2}Q_{\zeta_\ell}(X,z)
\right].
\tag{12}
\]

这里的概率对应“学习候选动作＋门控”的采样决策，不是未经推导的最终推力概率。

---

## C. Boids规则先验

四种规则保留：

\[
\begin{aligned}
\boldsymbol F_{i,\mathrm{sep}}
&=
-k_{\mathrm{sep}}
\sum_{j\in\mathcal N_i}
\frac{\boldsymbol p_j-\boldsymbol p_i}
{\|\boldsymbol p_j-\boldsymbol p_i\|},\\
\boldsymbol F_{i,\mathrm{ali}}
&=
\frac{k_{\mathrm{ali}}}{|\mathcal N_i|}
\sum_{j\in\mathcal N_i}\boldsymbol v_j-\boldsymbol v_i,\\
\boldsymbol F_{i,\mathrm{coh}}
&=
\frac{k_{\mathrm{coh}}}{|\mathcal N_i|}
\sum_{j\in\mathcal N_i}\boldsymbol p_j-\boldsymbol p_i,\\
\boldsymbol F_{i,\mathrm{att}}
&=
k_{\mathrm{att}}(\boldsymbol p_A-\boldsymbol p_i).
\end{aligned}
\tag{13}
\]

对齐和凝聚项在这里保留原文的书写方式，实际共同基线也保持一致。总力映射为推进器动作：

\[
\boldsymbol a_i^B
=
\mathcal M_B
\left(
\sum_{\ell}\boldsymbol F_{i,\ell},o_i
\right).
\tag{14}
\]

原文对应式（9）—（13）及其推力映射说明。2026_ARBoids_Adaptive_Residual\_…

这一节的结论是：

> 规则控制提供一致的协同先验；本文学习这种先验与学习动作在团队交互中的使用方式，而不改写规则控制本身。

---

## D. 状态与候选动作交互表示

原状态结构继续采用：

\[
s_i=[s_{i,D},s_{i,AT},s_{i,\mathrm{Boids}}].
\]

原文分别编码队友、攻击者与目标、Boids信息，并通过均值嵌入适配不同队友数量。2026_ARBoids_Adaptive_Residual\_…

新加入的关系向量为：

\[
\xi_{ij}
=
\left[
r_{ij},
\boldsymbol a_i^L,\boldsymbol a_i^B,
\boldsymbol a_j^L,\boldsymbol a_j^B
\right],
\tag{15}
\]

其中，\(r_{ij}\)由允许获取的相对位置、航向和运动估计组成。

先进行成对编码，再聚合：

\[
e_{ij}=f_\kappa(\xi_{ij}),
\qquad
\boldsymbol m_i
=
\frac{1}{|\mathcal N_i|}
\sum_{j\in\mathcal N_i}e_{ij}.
\tag{16}
\]

关键不在于是否使用均值，而在于：

**聚合前已经建立“特定队友的运动关系—双方候选动作”的对应。**

这是与普通队友位置编码的区别。信息交换采用固定协议，不再同时增加可学习通信拓扑、角色分配和优先级模块。

---

## E. 协同Adapter与条件价值学习

这是新文章对应原文III-E和图2的重点。

### 1. 协同Adapter

使用可重参数化的随机权重：

\[
z_i^\theta
=
\mu_\psi(I_i)+
\sigma_\psi(I_i)\epsilon_i,
\qquad
\epsilon_i\sim\mathcal N(0,1),
\]

\[
\boxed{
\theta_i
=
\frac{\tanh z_i^\theta+1}{2}.
}
\tag{17}
\]

训练时探索不同融合方式，评价时采用确定性输出。策略密度计算包含该变换对应的雅可比项。

网络可保留原图2的状态编码和三层512单元决策网络，只新增候选关系编码，并替换门控输入。**原文已有网络尺度可以直接作为初始配置，不需要为每一层重新创造理论。**2026_ARBoids_Adaptive_Residual\_…

### 2. 联合Critic

Critic接收：

\[
Q_\zeta(X,\boldsymbol A^L,\boldsymbol\Theta).
\]

可以采用共享节点编码与艇对编码：

\[
Q_\zeta
=
f_Q
\left(
\operatorname{Pool}_i z_i^Q,\,
\operatorname{Pool}_{i<j}z_{ij}^Q,\,
X_{\mathrm{task}}
\right).
\tag{18}
\]

艇对编码包含双方动作和权重，使价值函数能够表达权重间的相互影响，而不强制分解为若干独立单艇价值之和。

### 3. 核心训练信息：局部门控干预

在相同状态下，当前联合决策为：

\[
z=(\boldsymbol A^L,\boldsymbol\Theta).
\]

仅替换第\(i\)艘艇当前权重：

\[
z^{-i}
=
\left(
\boldsymbol A^L,
\theta_1,\ldots,\vartheta_i,\ldots,\theta_n
\right),
\tag{19}
\]

其中，\(\vartheta_i\)是参考权重，例如从固定集合\(\{0,0.5,1\}\)中抽取。最终策略仍输出连续权重。

两条推演的第一步仅在该权重上不同；从第二步开始，都使用当前策略根据各自预测状态重新生成动作。**不把某个门控固定20秒，也不固定其他艇未来的反馈。**

设推演在第\(h\)步达到时域或终止，构造与软价值定义一致的后果估计：

\[
\widehat G_H(z)
=
\sum_{k=0}^{h-1}\gamma^k
\left[
\hat r_k
-
\mathbf1_{k\ge1}
\alpha\log\Pi_{\bar\Omega}(\hat z_k\mid\hat X_k)
\right]
+
\gamma^h(1-\hat d_h)
V_{\bar\zeta}^{\Pi}(\hat X_h).
\tag{20}
\]

\(\bar\Omega\)表示生成该批监督目标时固定的策略参数。第一步动作已给定，因此后续决策的熵项从\(k=1\)开始。

成对后果差为：

\[
\boxed{
\widehat\Delta_i
=
\widehat G_H(z)
-
\widehat G_H(z^{-i}).
}
\tag{21}
\]

它直接对应：

> 在队友当前动作一致的比较条件下，本艇这次融合选择相对于参考选择，改变了多少后续团队价值？

### 4. 条件价值辅助损失

Critic对应的预测差：

\[
\Delta Q_{\zeta_\ell}
=
Q_{\zeta_\ell}(X,z)
-
Q_{\zeta_\ell}(X,z^{-i}).
\]

总损失为：

\[
\boxed{
\begin{aligned}
\mathcal L_Q
=
\sum_{\ell=1}^{2}
\Big[
&\mathbb E_{\mathcal D}
\big(Q_{\zeta_\ell}(X,z)-y\big)^2\\
&+
\lambda_\Delta
\mathbb E_{\mathcal D_\Delta}
\big(
\Delta Q_{\zeta_\ell}
-
\operatorname{sg}[\widehat\Delta_i]
\big)^2
\Big].
\end{aligned}
}
\tag{22}
\]

第一项使用实际交互，保持任务价值的基本尺度；第二项学习融合调整引起的条件变化。`sg`表示监督目标不反向传播。

成对评价和模型价值扩展有相应研究基础，本文将它们具体作用于规则—学习融合变量，而不是把这些基础概念重新命名为原创。[arXiv](https://arxiv.org/abs/1705.08926)

### 5. 为什么它直接服务于新门控？

对于局部变化：

\[
\Delta Q
\approx
(\theta_i-\vartheta_i)
\frac{\partial Q}{\partial\theta_i}.
\tag{23}
\]

因此，成对监督校准的是与门控更新直接相关的价值变化，而不是另加一个无关的风险分数。

在模型与尾部价值准确的理想条件下，成对推演的期望差与相应策略价值差一致。该说明用于解释训练目标的一致性，不扩展为全局最优或严格安全证明。

### 6. 现有长时推演怎样进入新方法？

保留已有动力学、观测生成、反馈调用和并行计算，将推演对象从：

\[
\text{五种整队策略}
\]

改为：

\[
\text{两次合法门控干预}.
\]

旧规划器的“时间评分”不能直接作为式（20）的奖励；推演统一使用新训练过程中的奖励、折扣与终止规则。

最终执行只运行协同Adapter。**长时推演服务于训练，不把新论文重新变成五候选在线规划。**

---

## F. 任务奖励

主机制实验先保留共同任务奖励：

\[
r_t^{\mathrm{team}}
=
\frac1n
\sum_{i=1}^{n}
\left(
r_{i,t}^{\mathrm{main}}
+
r_t^{\mathrm{form}}
+
r_{i,t}^{\mathrm{collision}}
\right).
\tag{24}
\]

主任务中的捕获、协助、失守和碰撞项沿用共同协议。原文分别采用捕获100、附近协助50、目标失守−100和碰撞−50。2026_ARBoids_Adaptive_Residual\_…

编队项保留：

\[
\boldsymbol g
=
\sum_i
\frac{\boldsymbol p_A-\boldsymbol p_i}
{\|\boldsymbol p_A-\boldsymbol p_i\|},
\]

\[
r^{\mathrm{form}}
=
0.5\,
\hat{\boldsymbol d}_{TA}^{\top}
\frac{\boldsymbol g}{\|\boldsymbol g\|}
-
\frac{\|\boldsymbol g\|}{n}.
\tag{25}
\]

其作用是保持与威胁方向有关的防御分布。2026_ARBoids_Adaptive_Residual\_…

**此阶段不再增加风险乘子、角色奖励和多层时间惩罚。**捕获效率继续作为独立评价指标；是否改善，由真实任务结果回答，不将奖励回报与成功概率或封顶耗时直接混称。

---

## G. 课程与训练过程

保留原文的敏捷度课程：

\[
L_{\mathrm{agi}}
\sim
U(\bar L_{\mathrm{agi}}-0.5,\bar L_{\mathrm{agi}}+0.5),
\]

\[
\bar L_{\mathrm{agi}}
=
2.0+0.25k,
\qquad k=0,1,2,3,
\tag{26}
\]

每25万训练步进入下一阶段。2026_ARBoids_Adaptive_Residual\_…

训练流程为：

1. 生成学习动作和Boids动作，构造候选交互信息。
2. 协同Adapter生成权重，融合后执行，存储真实转移。
3. 从部分训练状态抽取门控干预对，生成式（21）的后果差。
4. 用式（22）更新联合Critic，用式（12）更新策略。
5. 按SAC方式更新目标网络，继续环境交互。

首先冻结基础动作网络，只训练协同门控，用于识别新增机制；确认作用后，再进行动作网络与门控的联合训练。相应对照使用相同初始化和额外训练预算。

这两阶段服务于贡献归因，不另设一套独立的多阶段控制理论。

---

# 九、IV. Experiments：现有结果如何提升为论文证据，哪些结果需要新增？

## A. 实验设置

沿用数值训练环境和VRX平台，不重建任务。

原文的训练设置为三艇、100万步、学习率\(10^{-4}\)、批量4096、折扣0.99；图5使用五次独立运行、每5000步评价。可以采用同层级配置作为起点。2026_ARBoids_Adaptive_Residual\_… 2026_ARBoids_Adaptive_Residual\_…

新增参数集中为关系编码规模、后果推演时域\(H\)和辅助权重\(\lambda_\Delta\)，在开发场景中固定，不再搜索大量无关模块。

## B. 总体结果与协同行为

### 已有结果一：融合组织方式具有任务价值

在512个场景变化试验中：

| 方法 | 成功 | 碰撞 | 源码失守 |
|---|---:|---:|---:|
| 瞬时风险＋逐艇选择 | 403/512 | 55 | 54 |
| 瞬时风险＋联合选择 | 433/512 | 29 | 50 |

两者共享候选、评价和控制设置；成功率差为5.86个百分点，描述性配对区间为[2.54，9.18]。results results

可以上升为文章中的研究发现：

> **融合控制的组织方式能够改变团队防御结果，表明融合空间中存在可利用的联合决策价值。**

这比将结果写成“又修复了若干碰撞”更有学术意义。

### 已有结果二：持续反馈后果具有额外任务信息

最新1024个六艇场景可以保留为“显式反馈策略评价参照”。

| 已完成方法 | 成功 | 捕获 | 碰撞 | 平均封顶耗时／秒 |
|---|---:|---:|---:|---:|
| 原始ARBoids | 146 | 146 | 878 | 70.248 |
| CBF基线 | 1000 | 998 | 0 | 21.095 |
| 匹配短时评价 | 993 | 992 | 0 | 20.681 |
| 较强固定2秒短时评价 | 1008 | 1007 | 0 | 19.423 |
| 长时反馈选择参照 | 1024 | 1022 | 0 | 17.087 |
| 固定守卫 | 1020 | 954 | 0 | 27.573 |
| 固定1秒长时参照 | 1024 | 1022 | 0 | 16.907 |

上述数值来自当前报告，未捕获回合统一记为80秒。results

可形成三项论断：

**其一，安全满足没有消除控制策略之间的任务差异。**

多个方法都没有碰撞，但捕获与效率不同。

**其二，长期后续评价相对匹配短时评价具有增量价值。**

封顶耗时降低17.38%，配对差为−3.594秒，98.75%区间为[−4.806，−2.462]。results

**其三，成功率不足以单独描述防御行为。**

固定守卫虽然成功1020次，但有66次超时成功，捕获数明显少于长时参照。results

这些结果共同支持：

> **团队控制不仅需要避免相互干扰，还需要识别能够持续产生有效拦截的控制组合。**

自适应间隔没有额外收益，保留为次级结果，不进入新论文主贡献。results

### 新策略需要增加的行为结果

参照原文图4，展示一个完整的：

\[
\text{初始防护}
\rightarrow
\text{接近与交汇}
\rightarrow
\text{协调调整}
\rightarrow
\text{联合捕获}.
\]

在轨迹之外增加\(\theta_i(t)\)，把控制行为与学习变量联系起来。图中应能解释：不同艇为什么产生不同权重，以及这些变化如何形成有效机动。

已有五候选规划器的轨迹继续按原名称展示；新门控策略需要生成自己的轨迹。这只是结果归属的基本区别，不意味着已有结果不能进入文章。

---

## C. 学习性能与机制消融

### 主学习对照

| 方法 | 与完整方法的差别 | 对应的研究问题 |
|---|---|---|
| 原始 ARBoids | 保留原法自适应融合，不加 CBF，无两项新增机制 | 完整方法相对原法的任务表现如何？ |
| ARBoids＋CBF | 原法结构加固定安全过滤；正式学习对照另含匹配预算的继续训练 | 在附加执行层和训练预算下提供补充参照；不作为必须全面超越的唯一门槛 |
| 同信息普通协同学习 | 同样输入候选关系、同样联合Critic，无后果差监督 | 改善是否不只是更多信息或更大网络？ |
| 去除队友候选交互 | 保留队友状态、自身候选、条件价值学习和相同执行层 | 候选控制交互是否带来额外收益？ |
| 普通模型辅助学习 | 相近推演预算，使用普通回报监督 | 门控条件评价是否比一般模型辅助更有效？ |
| 短时条件评价 | 保留结构，缩短推演时域 | 长期任务后果是否有额外作用？ |
| 完整方法 | 候选交互＋长期条件价值学习 | 完整机制的学习效果 |

Boids、固定残差RP和纯SAC仍可按原文保留为基础对照；CBF和现有长时选择器作为补充参照，不将“在所有指标全面超过这些控制器”设为发表或开发前置条件。完整方法与原始 ARBoids 的差异可能同时包含 CBF 和额外训练的作用，不能仅凭该差异宣称两项创新成立；两项创新分别通过保留相同执行层、匹配信息和学习预算的消融验证。

### 应当得到什么结果？

不预先规定“必须提高10%”之类的发表数字，而检验三项明确关系。

**学习效果：**在相同环境步数下，完整方法获得更好的成功或捕获表现，或更早达到同等表现。

**机制作用：**完整方法优于同信息普通协同学习，并且相比普通模型辅助具有额外收益。

**长期信息作用：**短时与长时版本的差异能够对应真实任务表现，而不只是预测评分更低。

建议增加一个直接对应核心机制的指标：

\[
E_{\Delta}
=
\mathbb E
\left[
\left|
\Delta Q_\zeta-\Delta G_{\mathrm{env}}
\right|
\right],
\tag{27}
\]

其中，\(\Delta G_{\mathrm{env}}\)通过测试状态的成对环境分支得到。这样可以检查：

> 条件价值估计的改善，是否伴随融合决策和防御表现的改善？

这项实验比再添加很多外围场景更直接。

---

## D. 攻击者敏捷度

按照原文图6(a)，固定艇数，改变攻击者敏捷度。

已有六艇长时参照可保留：

| 敏捷度 | CBF耗时／秒 | 长时参照耗时／秒 | 长时参照成功 |
|---|---:|---:|---:|
| 1.5 | 15.355 | 13.005 | 256/256 |
| 2.0 | 19.480 | 15.145 | 256/256 |
| 2.5 | 22.873 | 19.458 | 256/256 |
| 3.0 | 26.671 | 20.741 | 256/256 |

results

新策略按同类设置增加曲线，检验高机动条件下交互融合的作用。原文也以这一变化作为主要任务难度实验。2026_ARBoids_Adaptive_Residual\_…

## E. 未见艇数泛化

沿用三艇训练、其他艇数测试的方式，建议保持原文图6(b)的二至七艇范围。

主要呈现：

\[
SR(n),\qquad CR(n),\qquad
\overline T^{\mathrm{cap}}(n).
\]

研究问题是：

> 随着交互增多，交互感知门控是否比普通融合保持更好的团队表现？

原文在三、四艇表现较好，超过四艇开始下降。新文章无需要求所有规模都完美，而应呈现新机制对这一变化的影响。2026_ARBoids_Adaptive_Residual\_…

## F. 学习型攻击者

保留原文五阶段攻守交替训练作为最后一项扩展实验，评价对手变化后策略能否继续调整。原文总计250万训练步，并报告防守成功、碰撞和攻击成功曲线。2026_ARBoids_Adaptive_Residual\_…

这一节不扩展成多攻击者、策略集成、通信对抗和复杂海况的综合工程。

---

# 十、图表应当如何与原文一一对应？

| 原文图 | 新论文图 | 要支撑的论点 |
|---|---|---|
| 图1：任务场景 | 高机动目标防御与密集交互示意 | 定义任务，不从已有算法缺点开篇 |
| 图2：Actor、Adapter、Critic | 学习动作、候选交互编码、协同Adapter、联合Critic及训练期成对推演 | 完整展示新可学习机制 |
| 图3：状态与队形奖励 | 状态关系及单艇门控干预示意 | 解释条件价值评价的对象 |
| 图4：VRX过程 | 新策略轨迹＋各艇融合权重 | 从变量变化解释联合机动 |
| 图5：学习曲线 | 完整方法与同信息、普通模型辅助、短时消融 | 证明学习和训练机制有效 |
| 图6：敏捷度与艇数 | 相同两类参数变化，增加捕获效率 | 证明任务适应性 |
| 图7：交替训练 | 相同类型的学习型攻击者曲线 | 评价对手变化适应 |
| 新增一张表 | 保留两批现有机制结果 | 支撑联合组织与长期评价的研究动机 |

原文第4页图2、第6页图5和第7页图6是最应对应的三处：**新机制、学习证据、任务变化。**它们共同构成文章主轴，而不是把新文章写成大量结果表的汇总。

---

# 十一、已有工作中，哪些进入正文，哪些不必继续占据主线？

| 已有内容 | 最终安排 |
|---|---|
| 三艇门控相对原方法提高5.08个百分点 | 作为融合层可改善性的前期证据 |
| 瞬时联合相对逐艇选择提高5.86个百分点 | 作为联合组织价值的直接证据 |
| 长时相对匹配短时降低17.38%封顶耗时 | 作为持续反馈评价具有额外信息的证据 |
| 长时相对CBF降低19.00%封顶耗时 | 作为安全约束之外仍存在任务决策空间的证据 |
| 固定守卫高成功、较低捕获 | 支撑成功与效率的分别评价 |
| 新增协同Adapter与条件价值训练 | 成为文章的方法中心和主学习结果 |
| 自适应重选间隔 | 保留次级结果，不再主打 |
| 进程、编译和哈希核验 | 放实验设置或补充材料 |
| 旧版本VRX结果 | 保留其原方法归属；新策略在同平台生成对应结果 |
| 原五候选长时规划器 | 保留为强参照与训练工具，不改称新强化学习策略 |

前两批实验已经建立了两项有价值的研究认识：**融合的组织方式影响团队表现；长期任务评价影响控制选择。**新学习机制的任务，是把这两项认识转化为能够直接执行的协同融合策略，而不是重复证明已有规划器运行成功。results results results

---

# 十二、文章应当形成的最终论证

整篇文章的中心不再是：

> ARBoids还有碰撞，所以增加交互模块、预测模块和辅助损失。

而是：

> **高机动对抗中的多艇防御需要把局部适应组织为有效联合机动。本文采用规则与学习互补的控制结构，通过候选动作交互条件化生成融合权重，并利用融合干预的长期后果改善联合价值学习，使训练能够辨识不同控制组合的团队作用。**

对应的证据链为：

\[
\boxed{
\begin{aligned}
&\text{任务：高机动对抗下的有效联合防御}\\
&\Downarrow\\
&\text{机制：交互感知的协同自适应融合}\\
&\Downarrow\\
&\text{变量：依赖候选控制关系的融合权重}\\
&\Downarrow\\
&\text{学习：实际交互与条件后果共同评价}\\
&\Downarrow\\
&\text{证据：学习改善、机制作用与任务表现}
\end{aligned}
}
\]

**ARBoids提供的范例，是把“规则与学习如何动态结合”组织成一套可学习控制方法。新论文对应地把“这种动态结合如何在团队交互中形成有效机动”组织成一套独立方法。**

研究路线至此可以固定：**保留物理任务、规则先验、SAC和现有实验基础；完成候选交互门控与条件价值训练这一套核心机制；围绕三项主张生成新的学习证据。**不再改成五策略在线规划论文，不再通过增加无关模块体现创新，也不把原文未完成的全部开放条件变成你必须承担的研究任务。
