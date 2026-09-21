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

### 3. Build VRX Environment:
The Gazebo based simulator [VRX repository](https://github.com/osrf/vrx) is recommended for running the evaluation. We followed the installation guide from [Distributional_RL_Decision_and_Control](https://github.com/RobustFieldAutonomyLab/Distributional_RL_Decision_and_Control) with some modifications to set up the VRX environment.

Install [ROS 2 Humble](https://docs.ros.org/en/humble/Installation.html) and [Gazebo Garden](https://gazebosim.org/docs/garden/installation) following the official guide. Then install additional dependencies by running:

```bash
sudo apt install python3-sdformat13 ros-humble-ros-gzgarden ros-humble-xacro
```

Navigate to the root directory and run the following commands:

```bash
mkdir -p vrx_ws/src
cp -r vrx/* vrx_ws/src/
cd vrx_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
. install/setup.bash
```


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

Ensure you have successfully built the VRX environment as described in the installation guide.

Source the ROS 2 environment and the VRX workspace:
```bash
source /opt/ros/humble/setup.bash
source vrx_ws/install/setup.bash
```

Navigate to the `vrx` directory and run the experiment (requires a pretrained model checkpoint). You can specify additional arguments such as model path, controller type, and device. The model checkpoint should be placed at the path specified by `--modelname`:
```bash
cd vrx
python tad_vrx_experiment.py --modelname checkpoints/adares1.pth --controller AdaRes --device cuda:0 --setting 1
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
│   ├── tad_vrx_experiment.py # Main VRX experiment script
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
