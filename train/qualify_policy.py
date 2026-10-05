"""Development acceptance gate; never starts formal experiments."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import torch
from RL.deployment import DeployedPolicy
from train_mappo import evaluate_episodes, summarize


def compare_paired(rows, baseline):
    """Success may tie; collisions must strictly decrease on identical seeds."""
    seeds = [row["seed"] for row in rows]
    expected = [row["seed"] for row in baseline]
    if not seeds or len(set(seeds)) != len(seeds) or set(seeds) != set(expected):
        raise ValueError("Acceptance requires the same unique evaluation seeds")
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate baseline seeds")
    for row in [*rows, *baseline]:
        if row["outcome_code"] not in (1, 2, 3, 4) or row["success"] != int(row["outcome_code"] in (3, 4)):
            raise ValueError("Inconsistent terminal outcome")
    success = sum(row["success"] for row in rows)
    original_success = sum(row["success"] for row in baseline)
    collision = sum(row["outcome_code"] == 2 for row in rows)
    original_collision = sum(row["outcome_code"] == 2 for row in baseline)
    return dict(episodes=len(rows), successes=success, baseline_successes=original_success,
                collisions=collision, baseline_collisions=original_collision,
                success_preserved=success >= original_success,
                collision_reduced=collision < original_collision,
                passed=success >= original_success and collision < original_collision)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stability-checks", type=int, default=3)
    parser.add_argument("--max-collision-rate", type=float,
                        help="Optional observed-rate ceiling on consecutive checks and both evaluation sets")
    parser.add_argument("--reference-checkpoint", type=Path, help="Also require fewer collisions than a previous deployed policy")
    args = parser.parse_args()
    if args.stability_checks < 1:
        parser.error("stability-checks must be positive")
    if args.max_collision_rate is not None and not 0. <= args.max_collision_rate <= 1.:
        parser.error("max-collision-rate must be between zero and one")
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if (config["environment"] != baseline["environment"]
            or config["agent"]["defender_num"] != baseline["defenders"]
            or config["curriculum"]["eva_agility"] != baseline["agility"]):
        raise ValueError("Policy/baseline evaluation protocols differ")
    reference = Path(baseline["checkpoint"])
    if hashlib.sha256(reference.read_bytes()).hexdigest() != baseline["checkpoint_sha256"]:
        raise ValueError("Baseline checkpoint hash mismatch")
    start = config["training"]["eval_seed"]
    count = config["training"]["eval_episodes"]
    reference_rows = [r for r in baseline["development"]["episodes"] if start <= r["seed"] < start + count]
    if len(reference_rows) != count:
        raise ValueError("Training development seeds are missing from baseline")
    target_success = sum(row["success"] for row in reference_rows) / count
    target_collisions = sum(row["outcome_code"] == 2 for row in reference_rows)
    with args.metrics.open(newline="", encoding="utf-8") as source:
        checks = [row for row in csv.DictReader(source) if row.get("def_sr", "") not in ("", "nan")]
    checks = checks[-args.stability_checks:]
    stable = (len(checks) == args.stability_checks
              and all(float(row["def_sr"]) >= target_success
                      and float(row["eval_collisions"]) < target_collisions for row in checks)
              and all(int(a["step"]) < int(b["step"]) for a, b in zip(checks, checks[1:]))
              and saved["training_steps"] in [int(row["step"]) for row in checks])
    if not stable:
        raise ValueError("Consecutive MAPPO validation checks have not met the acceptance gate")
    if args.max_collision_rate is not None and any(
            float(row["eval_collisions"]) / count > args.max_collision_rate for row in checks):
        raise ValueError("Consecutive validation checks exceed the requested collision-rate ceiling")
    policy = DeployedPolicy.load(args.checkpoint, args.device)
    previous = DeployedPolicy.load(args.reference_checkpoint, args.device) if args.reference_checkpoint else None
    if previous and (previous.config["environment"] != config["environment"]
                     or previous.config["agent"]["defender_num"] != config["agent"]["defender_num"]):
        raise ValueError("Previous policy evaluation protocol differs")
    result = dict(checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  baseline_checkpoint_sha256=baseline["checkpoint_sha256"],
                  evaluation="decentralized_actor", stability_passed=True,
                  stability_steps=[int(row["step"]) for row in checks],
                  training_steps=saved["training_steps"],
                  selection_training_steps=int(checks[-1]["step"]),
                  selection_elapsed_seconds=float(checks[-1]["elapsed_seconds"]),
                  pretraining=saved.get("trainer_state", {}).get("pretraining", {}),
                  formal_experiments_started=False)
    # Actor/model initialization can form several stages. Keep the optional
    # demonstration costs visible instead of reporting only the last PPO run.
    lineage = saved.get("trainer_state", {})
    result["initialization_history"] = []
    while isinstance(lineage, dict):
        stage = {key: lineage[key] for key in ("initialization", "pretraining", "gate_refinement")
                 if key in lineage}
        if stage:
            result["initialization_history"].append(stage)
        lineage = lineage.get("source_trainer_state")
    result["max_collision_rate"] = args.max_collision_rate
    result["success_reference"] = "original_arboids"
    started = time.perf_counter()
    if previous:
        result["reference_checkpoint"] = str(args.reference_checkpoint.resolve())
        result["reference_checkpoint_sha256"] = hashlib.sha256(args.reference_checkpoint.read_bytes()).hexdigest()
    for name in ("development", "qualification"):
        original = baseline[name]["episodes"]
        first = baseline[name]["first_seed"]
        if sorted(row["seed"] for row in original) != list(range(first, first + len(original))):
            raise ValueError("Expected contiguous baseline seeds")
        rows = evaluate_episodes(policy, config, len(original), first, agility=baseline["agility"])
        comparison = compare_paired(rows, original)
        result[name] = dict(comparison=comparison, summary=summarize(rows), episodes=rows)
        print(json.dumps({name: comparison}), flush=True)
        if previous:
            cached = baseline.get("reference_policy", {})
            compatible = (cached.get("checkpoint_sha256") == result["reference_checkpoint_sha256"]
                          and cached.get("environment") == config["environment"]
                          and cached.get("defenders") == config["agent"]["defender_num"]
                          and cached.get("agility") == baseline["agility"])
            prior_rows = cached[name] if compatible else evaluate_episodes(
                previous, config, len(original), first, agility=baseline["agility"])
            prior_comparison = compare_paired(rows, prior_rows)
            result[name]["previous_policy"] = dict(comparison=prior_comparison, summary=summarize(prior_rows), episodes=prior_rows)
            print(json.dumps({f"{name}_vs_previous_policy": prior_comparison}), flush=True)
            if name == "development":
                subset = [r for r in prior_rows if start <= r["seed"] < start + count]
                prior_collisions = sum(r["outcome_code"] == 2 for r in subset)
                result["refinement_stability_passed"] = all(
                    float(row["def_sr"]) >= target_success and float(row["eval_collisions"]) < prior_collisions
                    for row in checks)
    result["performance_qualified"] = all(result[name]["comparison"]["passed"] for name in ("development", "qualification"))
    if previous:
        result["performance_qualified"] &= (result["refinement_stability_passed"]
            and all(result[name]["previous_policy"]["comparison"]["collision_reduced"]
                    for name in ("development", "qualification")))
    if args.max_collision_rate is not None:
        result["collision_ceiling_passed"] = all(
            result[name]["summary"]["collision_rate"] <= args.max_collision_rate
            for name in ("development", "qualification"))
        result["performance_qualified"] &= result["collision_ceiling_passed"]
    result["wall_seconds"] = time.perf_counter() - started
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(performance_qualified=result["performance_qualified"], output=str(args.output.resolve()))), flush=True)


if __name__ == "__main__":
    main()
