# 本轮复核交付入口

以`CODEX_TASK.md`执行。论文的学术措辞已整理，方法、CITF-Phys身份、实验矩阵和核心源码保持原样。本轮重新运行了68项测试与有限端到端验收，记录在`audit/current_independent_review`。

正式数据采集前需完成`plan/CITF_短分支事件记录验收.md`所列的标签事件保存。该项尚未修改源码。正式主比较、独立训练与系统验证仍按原P0—P7规格完成。

下方为输入包原说明，其“本轮”指上传版本的修订历史；当前复核范围见`本轮独立评估.md`。

---

# CITF独立论文与Codex交付

本轮以用户上一版执行定稿为基础，修正未发表CI在预投稿稿件中的身份。CITF直接作为完整多无人艇控制方法提出，候选生成、名义运动、候选条件Transformer、联合选择及训练在正文中统一定义。内部CI名称与开发历史只保留在执行审计，不列为公开已有算法。

论文的公开算法比较包括ARBoids、RP、SAC、Boids和MAPPO。本文构造的物理预测替代CITF-Phys、两项学习消融与独立选择用于机制验证，仍在正文中给出明确算法定义。原CI与CITF-Phys不是同一个实现，历史结果禁止改名。

## 文件入口

| 文件 | 作用 |
|---|---|
| `paper/CITF_中文独立论文_修订稿.pdf` | 独立叙事的完整中文稿，正式结果红色待补 |
| `paper/CITF_中文独立论文_修订稿.md` | 可编辑正文，公式不编号 |
| `CODEX_TASK.md` | 完整本地研发与正式验证提示词 |
| `plan/CITF_最终实验与代码实施方案.md` | 完整实验协议及执行参数 |
| `plan/CITF_代码模块实施规格.md` | 尚待实现模块的明确接口 |
| `plan/CITF_独立论文叙事与实验身份修订.md` | 本轮修改与原期刊对应 |
| `release/method_registry.json` | 公开算法、本文对照和内部版本的身份登记 |
| `release/experiment_manifest.json` | 更新后的单一正式实验矩阵 |
| `release/delivery_status.json` | 实际完成与待补状态 |

文稿/实验角色修订为`CITF-independent-narrative-v4`。运行核心仍为`CITF-execution-closed-v3`，网络与输入保持`CITF-stopped-direct-v2`和`citf-stopped-aware-v2`。此次没有修改控制器代码、权重或训练结果。

## 实验身份变化

公开算法主表移除内部CI独立条目。CITF-Phys在相同主任务中继续执行，结果移入机制表；VRX和共同攻击者训练参考采用本版明确定义的CITF-Phys，替代原CI的正式实验职责。这是未来实验角色调整，不能将旧模型结果重命名继承。

六艇主任务、80秒协议、五次完整独立学习、两项学习消融、正式布局数量、机动性/跨艇数范围及七图证据职责保持。原始源码、旧开发记录与上一轮68项有限验收材料原样保存。正式主比较、泛化、对手学习和VRX仍待补。

## 核对与重建

```bash
python tools/check_paper_narrative.py
python tools/build_paper.py
```

运行代码的测试及有限真实流程仍使用原入口。上述叙事检查只核对文稿与实验身份，不代替模型训练或正式性能验证。完整研究从`CODEX_TASK.md`进入P0至P7。

本轮没有向本地Codex提交任务，没有训练或评价新控制器，也没有留下后台作业。
