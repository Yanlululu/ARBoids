# Published-parameter rerun

This is a new seed-42 run using the numerical parameters and task/reward rules
explicitly stated in the local published IEEE paper (DOI
10.1109/LRA.2026.3662620). It retains the author's released policy networks,
Boids implementation, APF attacker, observation implementation and dynamics.
It is not a claim that every implementation detail exactly matches every
formula in the paper.

The completed `main-seed42-20260921` experiment remains unchanged, including
its actor, configuration, learning history, 200 independent 2D evaluations and
20 VRX trials. Its actor SHA-256 is
`fe0c34101379582eac83ec6e89b4090d754308b604a97babf777f5b01d6c6ca2`.
The original `train/configs/train.yaml` and the environment's default `source`
protocol retain the original behavior.

## Explicit changes

| Setting | Released-code baseline | Published-parameter run | Source |
|---|---|---|---|
| Episode horizon | 80 s | 60 s in training and evaluation | IV-A, p. 3641 |
| Curriculum agility half-width | 0.25 | 0.5 | Eq. (18), p. 3641 |
| Curriculum mean | 2.0, +0.25 per 250,000 steps | Same | III-G |
| Adapter exploration | Shared uniform draw in [-0.1, 0.1] | Independent normal draws, mean 0 and standard deviation 0.1 | Eq. (14), pp. 3640–3641 |
| Attacker termination | Target breach or closer to target than every defender | Actual target breach only | II-B, p. 3638 |
| Target/collision boundary | Strict inequality | Inclusive boundary; capture remains strict as in II-B | II-B |
| Formation reward | Twice Eq. (17), only nonterminal steps | Eq. (17), also on terminal transitions | Eqs. (15)–(17) |
| Collision reward | -100 per collision partner | -50 for each involved defender, once per transition | III-F, p. 3641 |
| Main reward | Capture/helper rule also applied on timeout | -100 for breach, +100 for capture, +50 for nearby capture helpers; no timeout capture bonus | Eq. (16) |
| Combining rewards | Terminal conditions select a reward branch | Add main, formation and collision components | Eq. (15) |
| In-training validation | 50 episodes per 5,000 steps | 100 episodes per 5,000 steps | 100-trial comparisons in IV-C; the learning-curve episode count is not separately specified |

Other numerical settings remain: 3 defenders, 1,000,000 training steps,
batch size 4096, learning rate 0.0001, discount 0.99, action period 0.2 s,
capture/collision radii 5 m, target radius 15 m, target/attacker sensing ranges
60/15 m, defender thrust range [-500, 1000] N and Boids weights 10/0.1/0.1/0.5.
Warm-up 50,000 steps, hidden size 512, target-update factor 0.005, replay
implementation and optimizer details follow the released implementation.

Eq. (16) uses an inclusive capture boundary for the reward, whereas II-B uses
a strict capture boundary for task success. These are implemented as written.
When simultaneous terminal events occur, the existing target-breach, collision,
capture priority is retained. Zero-length vectors use a numerical epsilon.

## Retained implementation details and missing paper details

- The author's Boids code uses distance-dependent separation and its own
  alignment/cohesion implementation, which differ from a literal transcription
  of Eqs. (9)–(11). These are retained for this parameter rerun.
- The released observation includes attacker velocity: 18 values per defender
  for three defenders, whereas the textual state definition and Fig. 2 omit
  those two values. This run retains the released 18-value observation and its
  matching VRX deployment network.
- The paper mentions observation noise without specifying its distribution or
  magnitude. No invented observation-noise setting is introduced. The released
  random-current model remains active.
- Initial pose distributions, the dock sector [0, pi/4], APF details and the
  force-to-thrust conversion follow the author's code. Initial defender poses
  are sampled by that code; the paper does not provide complete numerical
  initialization settings for reconstructing its exact 100 trials.
- This experiment is one independently initialized seed, not the full five-run
  baseline, ablation, generalization and alternating-attacker study.

## Run and evaluate

Use `train/configs/paper-parameters.yaml`; the smoke configuration is only for
integration checks. Training and independent 2D evaluation must use the same
configuration. The queue automatically uses the saved run `config.yaml`.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
  .venv/bin/python -X utf8 train/train.py \
  --config train/configs/paper-parameters.yaml --seed 42 --device cuda:0 \
  --run-id paper-parameters-seed42-<unique-name>
```

The managed pipeline, `scripts/run_paper_reproduction.sh`, expects a fresh
`paper-parameters-*` directory under `train/experiments`. It refuses to replace
an existing experiment. It performs:

1. Fresh 1,000,000-step training, saving `exit_code`.
2. Two parallel 100-episode CPU evaluations (agility 2.0 and 2.25), seeds
   10000–10099, saving `evaluation.exit`.
3. Sequential VRX trials: 100 dock episodes and 10 open-water comparison
   episodes, agility 2.25, starting seed 20000, using `--termination-rule paper`.
   It saves `vrx.exit` and a final `pipeline.exit`.

An exit file containing zero means that stage ran successfully; the game
success rates are in each evaluation's `summary.json`. Infrastructure failures
stop the queue. Final-model snapshots, trajectories and failures are retained.

For local VRX deployment of this run:

```powershell
.\scripts\run_vrx.ps1 -Checkpoint "train\experiments\<new-run>\adares1.pth" -Setting 1 -TerminationRule paper -CaptureFrames
```
