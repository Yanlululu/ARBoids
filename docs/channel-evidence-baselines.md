# Additional original-algorithm reproduction

This supplement is fixed before launching the new SAC runs and before opening the
reserved matrix test cohort. It adds eight independent ARBoids SAC runs, with the
same seeds as the MAPPO matrix. These are whole-algorithm comparisons; the scalar
MAPPO control remains the main control for attributing a fusion-structure effect.

`evidence_sac.py` uses the repository's published-parameter SAC architecture and
optimizer: hidden width 512, replay capacity one million individual-agent
transitions, batch 4096, learning rate 0.0001, discount 0.99, target update 0.005,
50,000 environment steps of random exploration, followed by one SAC update per
environment step. Each run receives one million environment steps and no imitation
initialization. The three-defender environment, curriculum, reserved-seed exclusions,
development episodes, evaluation interval, and checkpoint selection rule are shared
with the MAPPO matrix. Return is recomputed with the common evaluator.

Here the shared curriculum means the attacker-agility schedule. SAC retains ordinary
random training resets and its published reward/optimizer; MAPPO additionally uses
collision-start revisits and a separate collision-cost critic. Thus this comparison
tests complete training pipelines. The matched scalar MAPPO arm controls these
conditions when isolating the fusion architecture.

The difference from the archived reference checkpoint's original training script is
explicit: development/model selection and geometric environment options are
standardized. This is a controlled reproduction, not a claim to reconstruct the
published paper's exact random streams. The archived reference remains a separate
fixed-policy comparison. Common teacher training, demonstration collection and
supervised updates are additional costs of the MAPPO setup and must be reported
when comparing learning efficiency to these from-scratch SAC runs.

The eight SAC runs execute sequentially alongside the parallel MAPPO queue, limiting
GPU contention from SAC's larger, per-step minibatches. Invalid numerical results
and interrupted jobs stop this queue and are retained for diagnosis. A fresh run is
never silently substituted for a failed seed. Evaluation uses the already reserved
matrix scenarios after training and reports every seed.
