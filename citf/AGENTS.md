# CITF independent-paper workspace

用户2026-11-05要求规避过程中无意义的检查：只做会影响当前算法/实验决策、真实收益判断或排除具体异常的必要核对，优先复用现有证据并结合正常实验完成；已有结论且相关内容未变时不重复检查。旧文档的核验、验收清单不自动成为每轮或恢复后的前置任务。工程测试、烟雾测试、哈希检查和自动代码验证仍按现行用户规则，仅在明确要求时执行。原定闭环评价、必要机制对照和正式论文实验继续推进；不为本原则新增清单、审批或报告。

On context compaction or resume, reuse reliable current context. For missing recovery context, first output only the “简短恢复摘要” and “当前决策依据与下一步” sections of `CODEX_TASK.md`, selecting by heading through the next heading of the same or higher level. Do not substitute large head/tail excerpts, whole-file output or full-plan reads for this entry step. From the repository root, this PowerShell command outputs those two sections:

```powershell
$showRecoverySection = $false
Get-Content -LiteralPath 'citf/CODEX_TASK.md' | ForEach-Object {
    if ($_ -match '^#{1,3}\s') {
        $showRecoverySection = $_ -match '^### (简短恢复摘要|当前决策依据与下一步)'
    }
    if ($showRecoverySection) { $_ }
}
```

These sections are a navigation aid and current decision snapshot, not a replacement for all effective requirements or the complete plan. Reconcile them with the latest user instruction and actual run state. Read the relevant requirements, plan sections, module specifications or evidence when needed to understand a gap, conflict, method choice, experiment design or stage transition; expand to the full plan when the decision requires it. A current hypothesis or branch does not define the whole research direction or cancel later deliverables. If output is truncated, retrieve only the missing relevant portion. Restore enough context to judge the next action against the original goal, then continue authorized work without adding tests, reports, approvals or experiment gates.

The package-era architecture and parameter statements below describe the imported reference implementation. The user's later authorization for internal algorithm redesign, and which earlier instructions it supersedes, are consolidated at the start of `CODEX_TASK.md`; preserve the unchanged task criteria, method identities and final deliverables while applying that authorization.

Read release/method_registry.json before exporting any paper table or result. CI is the author’s internal unpublished draft in the supplied context, not a published baseline. Define CITF as the complete controller in the manuscript. Keep actual source citations and all needed control equations.

Published comparison: ARBoids, RP, SAC, Boids, MAPPO. Author-constructed controls: CITF-Phys and the declared learning/selection variants. CI-original is audit-only. CITF-Phys maps to the current no-model held-thrust nominal predictor, not the legacy CI implementation. Never relabel archived results.

Document revision is CITF-independent-narrative-v4; runtime code is unchanged execution-v3. VRX and common-opponent experiments now use CITF-Phys instead of legacy CI; register new experimental identities and do not reuse old results.

Use CODEX_TASK.md as the task entry and the paper, full experiment plan and module specifications as detailed references. The imported initial-reading list does not require rereading every document after each compaction. The user-provided evidence, source archive and historical weights/results are read-only. Work in a separate local branch and output directory.

This is current candidate-interaction fusion only. Do not add CV, terminal value/outcome heads, long-horizon fusion plans or PCF-MBRL. Main training size is six; any other teammate may condition a pair. No three-agent restriction.

Runtime revision is CITF-execution-closed-v3. Network and input semantics remain CITF-stopped-direct-v2 and citf-stopped-aware-v2. Actual new-controller results need their own identity; never inherit the old 46/48 result.

Keep known remaining time online and actual future termination in labels only. Direct stopped-distance heads do not use unsupervised auxiliary tails. Future attacker action traces are audit labels, never predictor inputs.

Reserve source centre inside the existing initial bank; deduplicate; log actual score count. Default 96 is a cap; six agents with 12 initial and one 9-choice coordinate neighbourhood can score at most 60 unique configurations.

Only the registered online risk_weight sweep is predictor-load compatible. Final freeze still locks the complete runtime value. Loss weights must reach objective. Preserve immutable current-stage snapshots and best/last checkpoints separately; learning curves use validation only.

The same frozen attacker must drive behavior trajectories, every candidate branch and evaluation. Feedforward deterministic actor export is supported. Stateful/stochastic actors require an explicit clone/RNG contract before use. The untrained interface actor is never an experimental opponent.

Physical geometry family governs splits. Preserve all failed trials and seeds. Record actual commands, resources, provenance, controller/opponent identity and pending/failed/completed states. No placeholder successes. Equations in the manuscript remain unnumbered, formal missing results red 待补.

Use the single P0-P7 stage map in release/phase_contract.json. Implement missing baseline trainers, real attacker learning, VRX and statistics in P2, then complete the registered experiments. Do not claim formal tests or local jobs ran without real execution evidence.
