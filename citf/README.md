# CITF研发与论文入口

本目录用于Candidate-Interaction Transformer Fusion（CITF）的算法研发、实验和独立论文。目标是在完整防御任务中获得稳定的闭环收益，并用匹配对照建立候选交互学习的贡献。

当前处于固定源策略的算法开发阶段。正式主比较、五次独立学习、泛化、学习型对手和VRX证据仍需完成，开发结果不能替代正式结果。继续工作时，先读取完整实验方案及其1.4节交付边界，再结合`CODEX_TASK.md`末尾的阶段结论、`release/delivery_status.json`中的实际状态和源码；保留原始问题、关键假设、实验关系及决策依据。

## 文件入口

| 文件 | 作用 |
|---|---|
| `paper/CITF_中文独立论文_修订稿.pdf` | 独立叙事的完整中文稿，正式结果红色待补 |
| `paper/CITF_中文独立论文_修订稿.md` | 可编辑正文，公式不编号 |
| `CODEX_TASK.md` | 原始任务、研发过程、阶段结论和后续决策 |
| `plan/CITF_最终实验与代码实施方案.md` | 完整实验协议及执行参数 |
| `plan/CITF_代码模块实施规格.md` | 模块接口与实现规格，完成情况以实际状态和源码为准 |
| `plan/CITF_独立论文叙事与实验身份修订.md` | 本轮修改与原期刊对应 |
| `release/method_registry.json` | 公开算法、本文对照和内部版本的身份登记 |
| `release/experiment_manifest.json` | 更新后的单一正式实验矩阵 |
| `release/delivery_status.json` | 实际完成与待补状态 |

## 方法与证据边界

CITF作为完整控制方法定义，候选生成、运动预测、候选条件Transformer、联合选择与训练共同组成方法。公开算法比较包括ARBoids、RP、SAC、Boids和MAPPO；CITF-Phys及学习、选择对照按`release/method_registry.json`登记。内部CI不列为公开已有算法，旧结果不改名继承。

六艇主任务、80秒协议、独立重复、核心消融和既定系统验证沿用完整方案。内部探索服务于最终方法，不自动增加交付项目。当前实验使用各自登记的运行身份，不能把输入包的`CITF-execution-closed-v3`当作所有后续开发版本的统一身份。

## 历史资料与文稿重建

`audit/current_independent_review`和`本轮独立评估.md`保留输入包复核时的68项测试及有限验收记录，属于历史记录。当前实现和任务进度以实际源码与状态记录为准。

当前分支已清退被完整复核取代的`audit/previous_release`、`audit/execution_v3`中的跟踪记录、旧配置与旧交付说明，以及11月2—3日开发窗口的一次性调度脚本。历史跟踪文件可从Git提交`bac5c64`回查；`audit/execution_v3_complete`和当前独立复核保留。输入包的`release/FILE_MANIFEST.json`及`audit/current_independent_review/input_FILE_MANIFEST.json`仅描述当时交付内容，不是清理后工作树的文件清单。实际运行命令、任务库、模型和结果仍以原实验目录中的记录为准。

[ARBoids原文](review_sources/ARBoids_2026_正式期刊版_用户提供.pdf)保留在`review_sources`；[前序M1方法与证据](../docs/new-cooperative-defense-research-route.md)单独作为历史参考。旧r20核心证据保留在`../train/experiments/author-source-fusion-20261011/r20-paper-evidence`。

```bash
python tools/check_paper_narrative.py
python tools/build_paper.py
```

上述命令用于文稿核对与重建，不代表已执行训练或正式评价；是否运行检查遵循当前用户指令。
