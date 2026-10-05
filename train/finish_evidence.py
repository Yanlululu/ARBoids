"""Wait for the frozen training matrix, evaluate once, and report all contrasts."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from evidence_eval import sha256, suites
from evidence_matrix import SEEDS, ARMS
from evidence_stats import seed_comparison, holm, paired_fixed


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def training_barrier(root, wait):
    while True:
        status_path = root / "results/matrix/status.json"
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status["failures"]:
                raise RuntimeError(f"Training failures must be resolved before final testing: {status['failures']}")
        complete = []
        for arm in ARMS:
            for seed in SEEDS:
                path = root / "results/matrix" / f"{arm}-seed{seed}" / "complete.json"
                if not path.exists():
                    continue
                data = json.loads(path.read_text(encoding="utf-8"))
                checkpoint = path.parent / "channel-mappo-best1.pth"
                if not data.get("passed") or data["steps"] != 1000000 or sha256(checkpoint) != data["sha256"]:
                    raise RuntimeError(f"Incomplete or changed training output: {path}")
                complete.append(data)
        if len(complete) == len(ARMS) * len(SEEDS):
            dataset = {d["dataset_sha256"] for d in complete}
            if len(dataset) != 1:
                raise RuntimeError("Shared demonstration data contract failed")
            for arm in ARMS:
                networks = set()
                for seed in SEEDS:
                    init = root / "results/matrix" / f"{arm}-seed{seed}" / "initialization.pth"
                    record = next(d for d in complete if d["arm"] == arm and d["seed"] == seed)
                    if sha256(init) != record["initialization_sha256"]:
                        raise RuntimeError("Initialization checkpoint changed")
                    state = torch.load(init, map_location="cpu", weights_only=True)
                    networks.add(state["trainer_state"]["pretraining"]["initialization_sha256"])
                if len(networks) != len(SEEDS):
                    raise RuntimeError(f"Training seeds reused initial actor weights: {arm}")
            return complete
        if not wait:
            raise RuntimeError(f"Training is not complete: {len(complete)}/{len(ARMS) * len(SEEDS)}")
        marker = root / "jobs/matrix.json"
        if marker.exists() and os.name == "posix":
            job = json.loads(marker.read_text())
            process = Path("/proc") / str(job["pid"])
            if not process.exists() or b"evidence_matrix.py" not in (process / "cmdline").read_bytes():
                raise RuntimeError("Training scheduler stopped before its matrix completed")
        time.sleep(30)


def evaluate_matrix(root, jobs):
    queue = [(arm, seed) for arm in ARMS for seed in SEEDS]
    active, finished, failures = [], [], []
    evaluation_root = root / "results/matrix-test"
    evaluation_root.mkdir(parents=True, exist_ok=True)
    while queue or active:
        while queue and len(active) < jobs:
            arm, seed = queue.pop(0)
            checkpoint = root / "results/matrix" / f"{arm}-seed{seed}" / "channel-mappo-best1.pth"
            destination = evaluation_root / f"{arm}-seed{seed}"
            destination.mkdir(parents=True, exist_ok=True)
            log = (destination / "evaluation.log").open("a", encoding="utf-8")
            suite = "all" if arm in ("full", "scalar", "thrusters") else "nominal"
            command = [sys.executable, "-X", "utf8", "-u", str(Path(__file__).with_name("evidence_eval.py")),
                       "--kind", "channel", "--checkpoint", str(checkpoint), "--cohort", "matrix",
                       "--suite", suite, "--output", str(destination), "--device", "cpu", "--parallel", "8"]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MPLBACKEND="Agg")
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
            active.append((arm, seed, process, log))
            print(f"[FINAL TEST START] {arm} seed={seed}", flush=True)
        remaining = []
        for arm, seed, process, log in active:
            code = process.poll()
            if code is None:
                remaining.append((arm, seed, process, log))
            else:
                log.close()
                (failures if code else finished).append(dict(arm=arm, seed=seed, returncode=code))
                if code:
                    queue.clear()
                print(f"[FINAL TEST END] {arm} seed={seed} returncode={code}", flush=True)
        active = remaining
        atomic_json(evaluation_root / "status.json", dict(queued=len(queue), finished=finished,
                    failures=failures, active=[dict(arm=a, seed=s, pid=p.pid) for a, s, p, _ in active]))
        if active:
            time.sleep(10)
    if failures:
        raise RuntimeError(f"Final evaluation jobs failed: {failures}")


def load_rows(root, arm, seed, suite="nominal"):
    path = root / "results/matrix-test" / f"{arm}-seed{seed}" / f"{suite}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    completion = json.loads((root / "results/matrix" / f"{arm}-seed{seed}" / "complete.json").read_text())
    if not data["passed"] or data["policy"]["checkpoint_sha256"] != completion["sha256"]:
        raise RuntimeError(f"Test result/checkpoint mismatch: {path}")
    return sorted(data["rows"], key=lambda row: row["seed"])


def compare_arms(root, reference, suite="nominal"):
    events = {"collision": lambda r: r["outcome_code"] == 2,
              "success": lambda r: r["success"], "breach": lambda r: r["outcome_code"] == 1}
    left, right = [], []
    for seed in SEEDS:
        a, b = load_rows(root, "full", seed, suite), load_rows(root, reference, seed, suite)
        paired_fixed(a, b, 2)  # also verifies unique seeds and identical initial conditions
        left.append(a)
        right.append(b)
    return {name: seed_comparison([[event(r) for r in row] for row in left],
                                  [[event(r) for r in row] for row in right])
            for name, event in events.items()}


def costs(root, arm):
    result = []
    for seed in SEEDS:
        directory = root / "results/matrix" / f"{arm}-seed{seed}"
        with (directory / "metrics1.csv").open(newline="", encoding="utf-8") as source:
            rows = list(csv.DictReader(source))
        def total(key):
            return float(sum(float(r.get(key) or 0.) for r in rows))
        result.append(dict(seed=seed, on_policy_steps=int(rows[-1]["step"]),
                           branch_steps=total("branch_steps"),
                           training_and_development_wall_seconds=float(rows[-1]["elapsed_seconds"]),
                           guidance_weight_sum=total("guidance_weight"),
                           rollout_updates=len(rows), mean_projection_fraction=float(np.mean(
                               [float(r["projection_fraction"]) for r in rows]))))
    return result


def analyze(root):
    report = dict(protocol="channel-evidence-v1", passed=True, training_seeds=list(SEEDS),
                  nominal={}, contrasts={}, generalization={}, costs={})
    for arm in ARMS:
        rates = []
        for seed in SEEDS:
            rows = load_rows(root, arm, seed)
            rates.append(dict(seed=seed, success=float(np.mean([r["success"] for r in rows])),
                              collision=float(np.mean([r["outcome_code"] == 2 for r in rows])),
                              breach=float(np.mean([r["outcome_code"] == 1 for r in rows]))))
        report["nominal"][arm] = dict(per_seed=rates, **{
            key: dict(mean=float(np.mean([r[key] for r in rates])),
                      between_seed_sd=float(np.std([r[key] for r in rates], ddof=1)))
            for key in ("success", "collision", "breach")})
        report["costs"][arm] = costs(root, arm)
        if arm != "full":
            report["contrasts"][arm] = compare_arms(root, arm)
    primary = ("scalar", "thrusters")
    adjusted = holm([report["contrasts"][arm]["collision"]["exact_paired_sign_flip_two_sided_p"] for arm in primary])
    for arm, value in zip(primary, adjusted):
        report["contrasts"][arm]["collision"]["holm_primary_p"] = value
    for spec in suites("matrix"):
        if spec["name"] == "nominal":
            continue
        report["generalization"][spec["name"]] = {
            arm: compare_arms(root, arm, spec["name"]) for arm in primary}
    report["claim_assessment"] = dict(
        architecture_collision_effect_supported=all(
            report["contrasts"][arm]["collision"]["paired_difference"] < 0 and
            report["contrasts"][arm]["collision"]["holm_primary_p"] < .05 for arm in primary),
        all_nominal_outcomes_favorable=all(
            report["contrasts"][arm]["success"]["paired_difference"] >= 0 and
            report["contrasts"][arm]["breach"]["paired_difference"] <= 0 for arm in primary),
        publication_ready="Not determined by benchmark completion; consult novelty, adversarial and VRX evidence")
    report["analysis_code_sha256"] = sha256(__file__)
    report["statistics_code_sha256"] = sha256(Path(__file__).with_name("evidence_stats.py"))
    output = root / "artifacts"
    output.mkdir(exist_ok=True)
    atomic_json(output / "matrix-analysis.json", report)
    lines = ["# Independent matched-training results", "",
             "Rates are means over eight independently trained policies. SD is between training seeds.", "",
             "| Configuration | Success, mean ± SD | Collision, mean ± SD | Breach, mean ± SD |",
             "|---|---:|---:|---:|"]
    for arm, values in report["nominal"].items():
        rates = [f'{100 * values[k]["mean"]:.2f}% ± {100 * values[k]["between_seed_sd"]:.2f}'
                 for k in ("success", "collision", "breach")]
        lines.append(f'| {arm} | ' + ' | '.join(rates) + ' |')
    lines += ["", "Primary contrasts are full minus control on collision probability.", "",
              "| Control | Difference, percentage points | Crossed bootstrap 95% CI | Holm adjusted p |",
              "|---|---:|---:|---:|"]
    for arm in primary:
        r = report["contrasts"][arm]["collision"]
        lo, hi = r["crossed_bootstrap_95_interval"]
        lines.append(f'| {arm} | {100*r["paired_difference"]:.3f} | [{100*lo:.3f}, {100*hi:.3f}] | {r["holm_primary_p"]:.6g} |')
    lines += ["", "All secondary contrasts, generalization conditions, seed-level outcomes and costs are in matrix-analysis.json.",
              "Lower collision alone does not establish comprehensive superiority when success or target breaches worsen.",
              "The conditional training setup includes a shared pretrained ARBoids teacher and simulator corrections;",
              "teacher training and demonstration collection must be added to any end-to-end sample-efficiency comparison.", ""]
    (output / "matrix-results.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    lock = root / "artifacts/final-analysis-protocol.json"
    protocol = dict(analysis_sha256=sha256(__file__), statistics_sha256=sha256(Path(__file__).with_name("evidence_stats.py")),
                    evaluator_sha256=sha256(Path(__file__).with_name("evidence_eval.py")),
                    training_seeds=list(SEEDS), arms=list(ARMS), primary_contrasts=["scalar", "thrusters"],
                    tests_open_only_after_training_complete=True)
    if lock.exists() and json.loads(lock.read_text()) != protocol:
        raise RuntimeError("Locked final analysis code/protocol changed")
    lock.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(lock, protocol)
    completion = training_barrier(root, args.wait)
    atomic_json(root / "artifacts/final-checkpoints.json", completion)
    if not args.analyze_only:
        evaluate_matrix(root, args.jobs)
    report = analyze(root)
    print(json.dumps(report["claim_assessment"]), flush=True)
    atomic_json(root / "artifacts/evidence-complete.json", dict(passed=True, completed_unix=time.time(),
                scope="Independent matrix and its held-out tests; separate evidence has separate completion records"))


if __name__ == "__main__":
    main()
