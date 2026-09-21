#!/usr/bin/env bash
# One new seed: published parameters -> 2D evaluation -> VRX batches.
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
run=$(realpath "${1:?Provide a new paper-parameters-* experiment directory}")
seed=${2:-42}
cd "$repo"
if [[ "$run" != "$repo/train/experiments/"paper-parameters-* ]] || [ ! -d "$run" ]; then
  echo 'Expected a newly created paper-parameters-* directory under train/experiments.' >&2
  exit 64
fi
for name in config.yaml metrics1.csv adares1.pth exit_code evaluation.exit vrx.exit pipeline.exit; do
  if [ -e "$run/$name" ]; then echo "Refusing to overwrite existing experiment: $run/$name" >&2; exit 65; fi
done
trap 'status=$?; printf "%s\n" "$status" > "$run/pipeline.exit"' EXIT
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg PYTHONUTF8=1
echo '[TRAIN] Starting a fresh 1,000,000-step paper-parameter run.'
if .venv/bin/python -X utf8 -u train/train.py --config train/configs/paper-parameters.yaml \
    --device cuda:0 --seed "$seed" --run-id "$(basename "$run")" \
    > "$run/train.log" 2>&1; then status=0; else status=$?; fi
printf '%s\n' "$status" > "$run/exit_code"
if [ "$status" != 0 ]; then exit "$status"; fi
echo '[EVAL] Starting two independent 100-episode CPU evaluations.'
bash scripts/finish_main_evaluation.sh "$run" 100 > "$run/evaluation.log" 2>&1
.venv/bin/python -X utf8 - "$run" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
run=Path(sys.argv[1])
checkpoint=hashlib.sha256((run/'adares1.pth').read_bytes()).hexdigest()
for agility in ('2.0','2.25'):
    result=json.loads((run/f'eval-agility{agility}/summary.json').read_text())
    assert result['passed'] and result['episodes']==100
    assert result['protocol']=='paper-parameters-v1' and result['early_attacker_win'] is False
    assert result['duration_limit']==60. and result['agility_noise_half_width']==.5
    assert result['checkpoint_sha256']==checkpoint
print('[CHECK] Both evaluations used the final checkpoint and the paper parameter protocol.', flush=True)
PY
echo '[VRX] Starting 100 dock trials followed by 10 open-water comparison trials.'
(
  trap 'status=$?; printf "%s\n" "$status" > "$run/vrx.exit"' EXIT
  export PATH=/usr/sbin:/usr/bin:/sbin:/bin
  source "$repo/vrx_ws/activate.bash"
  python -X utf8 -u vrx/run_batch.py --checkpoint "$run/adares1.pth" \
    --episodes 100 --setting 1 --seed 20000 --agility 2.25 --duration 60 \
    --termination-rule paper --capture-first --output-dir "$run/vrx-dock"
  python -X utf8 -u vrx/run_batch.py --checkpoint "$run/adares1.pth" \
    --episodes 10 --setting 0 --seed 20000 --agility 2.25 --duration 60 \
    --termination-rule paper --capture-first --output-dir "$run/vrx-ocean"
) > "$run/vrx_pipeline.log" 2>&1
echo '[DONE] Training, 200 independent 2D episodes and 110 VRX trials completed.'
