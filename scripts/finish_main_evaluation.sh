#!/usr/bin/env bash
# Evaluate the completed policy in two independent, single-threaded CPU jobs.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run=$(realpath "${1:?Provide the existing main-training run directory}")
episodes=${2:-100}
cd "$repo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUTF8=1
trap 'status=$?; printf "%s\n" "$status" > "$run/evaluation.exit"' EXIT
echo '[WAIT] Waiting for main training to finish.'
while [ ! -f "$run/exit_code" ]; do sleep 30; done
if [ "$(cat "$run/exit_code")" != 0 ]; then
  echo 'Main training failed; evaluation cancelled.' >&2
  exit 2
fi
checkpoint="$run/adares1.pth"
before=$(sha256sum "$checkpoint" | cut -d ' ' -f 1)
agilities=(2.0 2.25)
pids=()
for agility in "${agilities[@]}"; do
  .venv/bin/python -X utf8 -u train/evaluate_policy.py \
    --checkpoint "$checkpoint" --episodes "$episodes" --seed 10000 \
    --agility "$agility" --duration 60 --device cpu \
    --output-dir "$run/eval-agility$agility" > "$run/eval-agility$agility.log" 2>&1 &
  pids+=("$!")
  echo "[EVAL] Started agility=$agility, episodes=$episodes, pid=$!."
done
failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[EVAL] Finished agility=${agilities[$index]}."
  else
    status=$?
    echo "[ERROR] agility=${agilities[$index]} exited with status=$status." >&2
    failed=1
  fi
done
if [ "$failed" != 0 ]; then exit 3; fi
after=$(sha256sum "$checkpoint" | cut -d ' ' -f 1)
if [ "$before" != "$after" ]; then
  echo 'Checkpoint changed during evaluation.' >&2
  exit 4
fi
echo "[DONE] Both independent evaluations completed; checkpoint SHA-256: $after"
