# Candidate-message and inference-cost audit

Before running these tests, the six conditions are fixed: ideal messages,
independent directed packet-loss probabilities 0.1, 0.3 and 0.5, and fixed message
delays of one or two 0.2-second control cycles. The frozen channel checkpoint uses
256 common scenarios, seeds 9060000–9060255, within the pre-existing excluded
training interval. The geometry/motion observations remain current; only the
candidate-message channel is perturbed. There is no retraining or test-based choice
of replacement behavior.

Each transmitted candidate contains four float32 features, namely the Boids and
learned candidates in normalized common/differential coordinates. A receiver holds
the last received packet and supplies its age in seconds. Before the first arrival,
features are zero and age increases from reset. The policy was trained with fresh
messages, so these results measure distribution-shift sensitivity of the existing
policy. This is a simulation of message impairment, not a physical network trial.

Report every outcome and all 15 condition/outcome comparisons to ideal messages,
with exact paired McNemar tests and Holm correction. The independent unit is the
scenario for this fixed model; it is not an additional training-seed experiment.

CPU latency uses 100 warmup calls and 1,000 measured team-actor calls for the frozen
channel and ARBoids reference at team sizes 3 and 8, single PyTorch thread, batch one.
The benchmark includes tensor conversion and inference, excludes sensing and network
transport, and reports median, p95 and p99 with host/runtime metadata. Communication
accounting distinguishes candidate payload from observation traffic and packet
headers. Neither result is an end-to-end real-time or hardware guarantee.
