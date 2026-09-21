import argparse
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys


def main():
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run Adversarial Learning Loop")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--config", default=str(script_dir / "configs/adversarial.yaml"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-id", default=None, help="Shared experiment name; default: a new timestamped run")
    parser.add_argument("--output-dir", default=str(script_dir / "experiments"))
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error("--rounds must be positive")
    run_id = args.run_id or datetime.now().strftime("adversarial_%Y-%m-%d_%H-%M-%S")
    for round_num in range(1, args.rounds + 1):
        for side in ("Def", "Att"):
            command = [
                sys.executable, "-X", "utf8", "-u", str(script_dir / "adversarial-learning.py"),
                "--config", args.config, "--device", args.device,
                "--seed", str(args.seed), "--round", str(round_num), "--side", side,
                "--run-id", run_id, "--output-dir", args.output_dir,
            ]
            print(f"[Loop] Round {round_num} {side}: {shlex.join(command)}", flush=True)
            subprocess.run(command, check=True)
    print("[Success] All rounds completed!", flush=True)


if __name__ == "__main__":
    main()
