# CITF independent-paper workspace

Read release/method_registry.json before exporting any paper table or result. CI is the author’s internal unpublished draft in the supplied context, not a published baseline. Define CITF as the complete controller in the manuscript. Keep actual source citations and all needed control equations.

Published comparison: ARBoids, RP, SAC, Boids, MAPPO. Author-constructed controls: CITF-Phys and the declared learning/selection variants. CI-original is audit-only. CITF-Phys maps to the current no-model held-thrust nominal predictor, not the legacy CI implementation. Never relabel archived results.

Document revision is CITF-independent-narrative-v4; runtime code is unchanged execution-v3. VRX and common-opponent experiments now use CITF-Phys instead of legacy CI; register new experimental identities and do not reuse old results.

Read CODEX_TASK.md, the paper, the full experiment plan and module specifications first. The user-provided evidence, source archive and historical weights/results are read-only. Work in a separate local branch and output directory.

This is current candidate-interaction fusion only. Do not add CV, terminal value/outcome heads, long-horizon fusion plans or PCF-MBRL. Main training size is six; any other teammate may condition a pair. No three-agent restriction.

Runtime revision is CITF-execution-closed-v3. Network and input semantics remain CITF-stopped-direct-v2 and citf-stopped-aware-v2. Actual new-controller results need their own identity; never inherit the old 46/48 result.

Keep known remaining time online and actual future termination in labels only. Direct stopped-distance heads do not use unsupervised auxiliary tails. Future attacker action traces are audit labels, never predictor inputs.

Reserve source centre inside the existing initial bank; deduplicate; log actual score count. Default 96 is a cap; six agents with 12 initial and one 9-choice coordinate neighbourhood can score at most 60 unique configurations.

Only the registered online risk_weight sweep is predictor-load compatible. Final freeze still locks the complete runtime value. Loss weights must reach objective. Preserve immutable current-stage snapshots and best/last checkpoints separately; learning curves use validation only.

The same frozen attacker must drive behavior trajectories, every candidate branch and evaluation. Feedforward deterministic actor export is supported. Stateful/stochastic actors require an explicit clone/RNG contract before use. The untrained interface actor is never an experimental opponent.

Physical geometry family governs splits. Preserve all failed trials and seeds. Record actual commands, resources, provenance, controller/opponent identity and pending/failed/completed states. No placeholder successes. Equations in the manuscript remain unnumbered, formal missing results red 待补.

Use the single P0-P7 stage map in release/phase_contract.json. Implement missing baseline trainers, real attacker learning, VRX and statistics in P2, then complete the registered experiments. Do not claim formal tests or local jobs ran without real execution evidence.
