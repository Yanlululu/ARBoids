# Shared learned-attacker robustness panel

This prospective supplement evaluates transfer to learned opponents. It does not
alter any defender checkpoint or the main training matrix. Three SAC attackers
(seeds 1709, 2711, 3719) each receive 500,000 environment steps, a 50,000-step random
warmup, the repository's attacker observation and reward, and the paper-preset SAC
architecture and optimizer. Each training episode selects the frozen channel or
frozen ARBoids defender with probability one half. Defender exposure, simulator
steps, gradient updates and wall time are retained. Training uses agility 2.0 with
the same uniform perturbation as the primary experiments.

Every 100,000 steps, attackers are evaluated against both defenders on the same
64 development scenarios (630000–630063), with deterministic actions and no agility
noise. Attacker selection maximizes mean target-breach rate across the two defenders,
then mean attacker return. Defender collisions remain a separate outcome. Final
attacker testing opens only after all three attacker training runs finish.

The test cohort contains 2,048 common seeds (9050000–9052047), already within the
excluded training-seed interval. Each frozen defender faces all three selected
attackers and the ordinary APF attacker on this cohort. Newly trained full, scalar,
and ARBoids SAC policies face the same panel after their entire training queues
finish. No defender is fine-tuned on these tests. Report every attacker, every
training seed, capture, timeout denial, collision, breach, return and control effort.
Training-seed inference for full versus scalar or SAC is conditional on this fixed
attacker panel; it does not establish robustness to all learned opponents.

The panel supplies a common learned-opponent test with equal training exposure to
the two frozen defenders. It is different from reproducing the source paper's
alternating attacker/defender curriculum. Evidence for that training procedure or
a minimax robustness claim would require its own controlled experiment. Attackers
are not retrained or selected using final-test outcomes.
