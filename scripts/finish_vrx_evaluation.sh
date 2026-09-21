#!/usr/bin/env bash
# Complete VRX evaluation after the main training and independent 2D evaluations.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run=$(realpath "${1:?Provide the existing main-training run directory}")
episodes=${2:-10}
cd "$repo"
trap 'status=$?; printf "%s\n" "$status" > "$run/vrx.exit"' EXIT
echo '[WAIT] Waiting for main training to finish.'
while [ ! -f "$run/exit_code" ]; do sleep 30; done
if [ "$(cat "$run/exit_code")" != 0 ]; then
  echo 'Main training failed; VRX evaluation cancelled.' >&2
  exit 2
fi
echo '[WAIT] Waiting for the two independent 100-episode evaluations.'
while [ ! -f "$run/evaluation.exit" ]; do sleep 30; done
if [ "$(cat "$run/evaluation.exit")" != 0 ]; then
  echo 'Independent 2D evaluation failed; VRX evaluation cancelled.' >&2
  exit 3
fi
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUTF8=1
source "$repo/vrx_ws/activate.bash"
python -X utf8 - "$run" <<'PY'
import csv
import hashlib
import json
from pathlib import Path
import sys
run = Path(sys.argv[1])
checkpoint = hashlib.sha256((run/'adares1.pth').read_bytes()).hexdigest()
for agility in ('2.0', '2.25'):
    result = json.loads((run/f'eval-agility{agility}/summary.json').read_text())
    if not result['passed'] or result['episodes'] != 100 or result['checkpoint_sha256'] != checkpoint:
        raise RuntimeError('Incomplete or inconsistent final-checkpoint evaluation')
print('[CHECK] Both independent 2D evaluations use the same final checkpoint.', flush=True)
PY
for setting in 0 1; do
  if [ "$setting" = 0 ]; then name=ocean; else name=dock; fi
  echo "[VRX] Starting $episodes $name trials."
  python -X utf8 -u vrx/run_batch.py --checkpoint "$run/adares1.pth" \
    --episodes "$episodes" --setting "$setting" --seed 20000 --agility 2.25 \
    --capture-first --output-dir "$run/vrx-$name"
done
echo '[DONE] Both VRX evaluation batches completed.'
