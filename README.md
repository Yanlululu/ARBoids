# ARBoids: Adaptive Residual Reinforcement Learning With Boids Model for Cooperative Multi-USV Target Defense

## 📄 Paper
This repository contains the code implementation of our [paper](https://arxiv.org/abs/2502.18549), which has been published in **IEEE Robotics and Automation Letters (RA-L)**.

## 🔍 Overview
We focus on the target defense problem (TDP) for multiple unmanned surface vehicles (USVs), which concerns intercepting an adversarial USV before it breaches a designated target region, using one or more defending USVs. A particularly challenging scenario arises when the attacker exhibits superior maneuverability compared to the defenders, significantly complicating effective interception. To tackle this challenge, we introduce ARBoids, a novel adaptive residual reinforcement learning framework that integrates deep reinforcement learning (DRL) with the biologically inspired, force-based Boids model. Within this framework, the Boids model serves as a computationally efficient baseline policy for multi-agent coordination, while DRL learns a residual policy to adaptively refine and optimize the defenders' actions. A novel adapter module is designed to dynamically balance the weights between base policy and RL policy.

## ⚙️ Installation

### 1. Clone this Repository:
   ```bash
   git clone https://github.com/taojy687/ARBoids.git
   cd ARBoids
   ```

### 2. Build Training Environment:
Create a new virtual environment:
   ```bash
   conda create -n arboids python=3.10
   conda activate arboids
   ```
Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Independent Python 3.12 training environments

The training entry points support Python 3.12, PyTorch 2.12.1 with CUDA 13.0,
and NumPy 1.26.4. The tested core versions are in `requirements-py312.txt`.
Use a project-specific environment; install the CUDA build before the other dependencies:

```bash
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-py312.txt
python -m pip check
```

The configured local environment is `D:\ARBoids\.venv` (Windows). The separate
server environment is `/root/autodl-tmp/ARBoids/.venv` (Linux, activate with
`conda activate /root/autodl-tmp/ARBoids/.venv`). Neither environment uses the
other machine's installed packages.

Run the main integration check from the repository root. It keeps the published
512-unit networks, batch size 4096 and optimizer settings, with 512 warm-up steps,
2,000 total steps and two evaluations of 10 episodes each. This checks execution
and checkpoint generation; it is not a convergence or paper-result experiment.

Windows PowerShell:

```powershell
cd D:\ARBoids
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:MPLBACKEND = 'Agg'
.\.venv\Scripts\python.exe -X utf8 -u train/train.py --config train/configs/smoke.yaml --device cuda:0 --seed 42
```

Linux:

```bash
cd /root/autodl-tmp/ARBoids
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg .venv/bin/python -X utf8 -u train/train.py --config train/configs/smoke.yaml --device cuda:0 --seed 42
```

For a short alternating defender/attacker check, use the same environment:

```bash
python -X utf8 -u train/run_adversarial_loop.py --config train/configs/adversarial-smoke.yaml --device cuda:0 --rounds 2 --seed 42
```

Main and loop runs create timestamped directories under `train/experiments/`.
Use `--run-id` to select a name and `--output-dir` to select a parent directory.
The alternating loop shares one run directory across its phases. Reusing a run
name appends metrics and replaces checkpoints, so use a new name for a fresh run.
The final training step is evaluated and saved even when it is not a multiple
of `eval_interval`. Checkpoint-save failures stop the process with an error.

For the full training schedule, replace the smoke config with `train/configs/train.yaml`.
The original full-training parameters are unchanged. VRX still requires the
separate ROS 2/Gazebo setup below; loading weights into its policy network does
not by itself test a VRX simulation.

### 3. Build VRX Environment

Use Ubuntu 22.04 with ROS 2 Humble and Gazebo Garden. ROS uses the system
Python 3.10 ABI; keep its environment separate from Python 3.12 training.
The scripts install the simulator, fetch the missing official VRX/Fuel models,
and build five ROS packages. Run from the repository root:

```bash
sudo bash scripts/setup_vrx_ubuntu.sh
python3 -X utf8 scripts/fetch_vrx_assets.py
python3 -X utf8 scripts/fetch_fuel_assets.py
bash scripts/build_vrx.sh
source vrx_ws/activate.bash
```

`build_vrx.sh` creates `.vrx-venv` with system ROS packages, CPU PyTorch 2.12.1,
and NumPy 1.26.4. CPU policy inference leaves the GPU available for training and
Gazebo rendering. Assets are cached under `.vrx-assets`; the asset manifests
record upstream versions and licenses. Keep these directories for offline use.

The local Windows installation uses the dedicated `ARBoids-22.04` WSL 2 distro
stored in `D:\ARBoids-WSL`. It shares `D:\ARBoids` as `/mnt/d/ARBoids`.
Its Linux build and Python environment live in `/opt/arboids-runtime`:

```bash
ARBOIDS_VRX_WS=/opt/arboids-runtime/vrx_ws \
ARBOIDS_VRX_VENV=/opt/arboids-runtime/venv bash scripts/build_vrx.sh
source /opt/arboids-runtime/vrx_ws/activate.bash
```

WSLg supplies the local GUI and D3D12 rendering on the NVIDIA adapter. On the
server, use `--headless` for EGL rendering. The launcher loads a project-scoped
Ogre shim that limits rendering workers to eight; this avoids the Ogre 2.3
thread-count assertion on hosts exposing more than 127 CPU cores.

## 🚀 Training ARBoids Model

Navigate to the `train` directory before running any training scripts:
```bash
cd train
```

To train the ARBoids model, use the following command. You can specify the custom configuration file, device, and random seed.

```bash
python train.py --config configs/train.yaml --device cuda:0 --seed 42
```

The adversarial training involves an iterative process where the Defender and Attacker are trained alternately. To run the adversarial training loop, use the following command. You can specify the number of rounds and the device to use.

```bash
# Run 3 rounds of adversarial training (Def -> Att -> Def -> Att -> ...)
python run_adversarial_loop.py --rounds 3 --device cuda:0
```

Alternatively, you can run each step manually by specifying the round and learning side:

```bash
# Example: Round 1 Defender Training
python adversarial-learning.py --round 1 --side Def

# Example: Round 1 Attacker Training
python adversarial-learning.py --round 1 --side Att
```

## 🌊 VRX Simulation

Run from the repository root after sourcing `vrx_ws/activate.bash` on Linux,
or `/opt/arboids-runtime/vrx_ws/activate.bash` inside the local WSL distro:

```bash
python -X utf8 -u vrx/run_experiment.py \
  --checkpoint train/experiments/<run>/adares1.pth \
  --controller AdaRes --setting 0 --agility 2.25 --seed 42 \
  --headless --capture-frames
```

Setting `0` is the open-water scenario; `1` is the dock scenario. Omit
`--headless` for the local interactive Gazebo window. Windows PowerShell:

```powershell
cd D:\ARBoids
.\scripts\run_vrx.ps1 -Checkpoint 'train\experiments\<run>\adares1.pth' -Setting 1 -CaptureFrames
```

Each run writes `result.json`, `trajectory.npz`, and `gazebo.log` to a unique
directory under `vrx/results`. `--capture-frames` adds a fixed overview camera
and saves actual Gazebo PNG frames. A valid task failure still has `passed=true`
and `success=false`; missing feedback, failed bridges or simulator crashes have
`passed=false` and a nonzero exit code. Timeouts use simulation time (60 seconds).

The deployment adapter maps policy action 0 to the starboard thruster and action
1 to port. This matches the yaw sign of the original 2D training dynamics with
Gazebo's ENU coordinates. Trajectories retain both policy actions and the actual
port/starboard commands. The training dynamics and algorithm are unchanged.
The legacy `tad_vrx_experiment.py` command delegates to this same runner.
Deployment observations contain 14 local features plus two values per actual
teammate, matching the training input (18 values with three defenders).

The final evaluation defaults are 60 simulation seconds, a 5 m capture radius,
a 5 m defender collision threshold, and a 15 m target radius. These replace the
legacy VRX script's 100 s / 5.5 m settings. Independent 2D evaluation also uses
60 s by default; `--duration 80` reproduces the source environment's horizon.

The published paper and released training code differ in two relevant settings:
the source uses an 80 s training horizon and curriculum agility noise of ±0.25,
whereas the paper describes 60 s and ±0.5. The current main training preserves
the released code. Its in-training validation also uses 80 s. Independent 2D
evaluation retains the source's early attacker-win rule when the attacker is
closer to the target than every defender. VRX checks an actual target breach.
Report these differences when comparing results with the paper; a single seed
and the initial VRX batches do not reproduce the five-seed baseline/ablation suite.
See [the paper's experimental setup](https://arxiv.org/html/2502.18549v3#S4.SS1).


Run repeated trials with distinct evaluation seeds:

```bash
python -X utf8 -u vrx/run_batch.py \
  --checkpoint train/experiments/<run>/adares1.pth \
  --setting 0 --episodes 10 --seed 20000 --agility 2.25 \
  --output-dir vrx/results/<new-batch-name>
```

The batch retains every trial and stops on an infrastructure failure. Use a
new output name for a new batch. Independent 2D evaluation uses the training
environment instead:

```bash
.venv/bin/python -X utf8 -u train/evaluate_policy.py \
  --checkpoint train/experiments/<run>/adares1.pth \
  --episodes 100 --seed 10000 --agility 2.25 \
  --output-dir train/experiments/<run>/eval-agility2.25
```

## 📁 Project Structure

```bash
ARBoids/
├── train/                 # 2D Training Environment
│   ├── configs/           # Configuration files (train.yaml, adversarial.yaml)
│   ├── envs/              # Environment definitions (TADgame.py)
│   ├── policy/            # Policy networks (SAC, etc.)
│   ├── utils/             # Utility functions
│   ├── adversarial-learning.py # Adversarial training step
│   ├── run_adversarial_loop.py # Adversarial training loop
│   └── train.py           # Main training script
├── vrx/                   # High-fidelity Gazebo/ROS 2 Simulation
│   ├── vrx_gz/            # Gazebo resources (worlds, models, launch files)
│   ├── vrx_ros/           # ROS nodes
│   ├── vrx_urdf/          # Robot descriptions (URDF/Xacro)
│   ├── models.py          # Policy networks for VRX
│   ├── run_experiment.py  # Bounded ROS/Gazebo experiment
│   ├── run_batch.py       # Repeated VRX evaluation
│   ├── tad_vrx_experiment.py # Original controller helpers / legacy entry point
│   └── utils.py           # Utility functions for VRX
├── LICENSE
├── README.md
└── requirements.txt
```

## 📝 Citation

If you find this work useful, please cite our paper:

```bibtex
@article{tao2026arboids,
  title={ARBoids: Adaptive Residual Reinforcement Learning With Boids Model for Cooperative Multi-USV Target Defense},
  author={Tao, Jiyue and Shen, Tongsheng and Zhao, Dexin and Zhang, Feitian},
  journal={IEEE Robotics and Automation Letters},
  volume={11},
  number={3},
  pages={3637-3644},
  year={2026},
}
```
