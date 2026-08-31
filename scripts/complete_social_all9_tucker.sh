#!/usr/bin/env bash
# Train and evaluate the nine-social-graph GraphSAGE mixture under the original
# matrix_s0 fixed-compute protocol.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-${ROOT}/configs/social_sources.yaml}"
STATE_ROOT="${STATE_ROOT:-${ROOT}/state/social_all9_matrix_s0}"
RESULT_ROOT="${RESULT_ROOT:-${ROOT}/results/social_all9_matrix_s0/raw}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/log/social_all9_matrix_s0}"
GPU="${GPU:-2}"
RUN_ID="matrix_all9_social_w256_existing_s0"
SOURCES_CSV="covid,cp_hk,midterm,ukr_rus,covid_political,election2020,facebook_page_reference,twibot20,ukr_rus_suspended"
TARGETS=(covid_political election2020 facebook_page_reference twibot20 ukr_rus_suspended)

[[ "$GPU" =~ ^[23]$ ]] || {
  echo "refusing GPU $GPU: only Tucker GPUs 2 and 3 are authorized" >&2
  exit 2
}

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
PYTHON="${CONDA_PREFIX}/bin/python"
mkdir -p "$STATE_ROOT" "$RESULT_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/eval" "$LOG_ROOT/launch"
cd "$ROOT"

wait_for_gpu() {
  local used util
  while true; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")"
    util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU")"
    if (( used < 2000 && util < 10 )); then return; fi
    echo "[gpu $GPU] waiting utc=$(date -u +%FT%TZ) used_mib=$used util_pct=$util"
    sleep 60
  done
}

checkpoint="$STATE_ROOT/$RUN_ID/checkpoints/step_2500.pt"
summary="$STATE_ROOT/$RUN_ID/summary.json"

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "protocol=matrix_s0-link-prediction-fixed-compute"
  echo "config=$CONFIG"
  echo "run_id=$RUN_ID"
  echo "checkpoint_step=2500"
  echo "training_seed=0"
  echo "labels_per_class=10"
  echo "sources=$SOURCES_CSV"
  echo "targets=${TARGETS[*]}"
  echo "gpu=$GPU"
  echo "wandb_mode=$WANDB_MODE"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance.txt"

wait_for_gpu
if [[ ! -f "$summary" || ! -f "$checkpoint" ]]; then
  if [[ -e "$STATE_ROOT/$RUN_ID" ]]; then
    echo "refusing ambiguous partial run: $STATE_ROOT/$RUN_ID" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT/src" "$PYTHON" -u -m mixture_scaling.train \
    --config "$CONFIG" --sources "$SOURCES_CSV" --run-id "$RUN_ID" \
    --device 0 --seed 0 --max-steps 2500 --output-root "$STATE_ROOT" \
    > "$LOG_ROOT/train/${RUN_ID}.log" 2>&1
fi
[[ -f "$summary" && -f "$checkpoint" ]] || {
  echo "training did not produce the required summary and step-2500 checkpoint" >&2
  exit 1
}

for target in "${TARGETS[@]}"; do
  output="$RESULT_ROOT/all9_to_${target}_step2500.json"
  if [[ ! -f "$output" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$ROOT/src" "$PYTHON" -u -m mixture_scaling.evaluate \
      --config "$CONFIG" --checkpoint "$checkpoint" --target "$target" \
      --device 0 --output "$output" \
      > "$LOG_ROOT/eval/all9_to_${target}_step2500.log" 2>&1
  fi
done

"$PYTHON" - "$summary" "$RESULT_ROOT" "$SOURCES_CSV" <<'PY'
import json
import sys
from pathlib import Path

summary_path, result_root, sources_text = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
sources = sources_text.split(",")
targets = {
    "covid_political",
    "election2020",
    "facebook_page_reference",
    "twibot20",
    "ukr_rus_suspended",
}
summary = json.loads(summary_path.read_text())
if summary.get("status") != "complete" or int(summary.get("final_step", -1)) != 2500:
    raise SystemExit(f"invalid training summary: {summary_path}")
if summary.get("sources") != sources or int(summary.get("seed", -1)) != 0:
    raise SystemExit("training provenance does not match the requested all-nine seed-0 run")

files = sorted(result_root.glob("all9_to_*_step2500.json"))
if len(files) != 5:
    raise SystemExit(f"expected five result files, found {len(files)}")
rows = [json.loads(path.read_text()) for path in files]
if {row.get("target") for row in rows} != targets:
    raise SystemExit("downstream target set is incomplete")
for row in rows:
    if row.get("sources") != sources:
        raise SystemExit(f"source provenance mismatch for {row.get('target')}")
    if int(row.get("seed", -1)) != 0 or int(row.get("checkpoint_step", -1)) != 2500:
        raise SystemExit(f"checkpoint provenance mismatch for {row.get('target')}")
    for metric in ("roc_auc_ovr_macro", "f1_macro", "accuracy"):
        value = float(row[metric])
        if not 0.0 <= value <= 1.0:
            raise SystemExit(f"invalid {metric} for {row.get('target')}: {value}")
print("SOCIAL_ALL9_MATRIX_OK physical_cells=5 checkpoint_step=2500 training_seed=0")
PY

{
  echo "completed_utc=$(date -u +%FT%TZ)"
  echo "physical_cells=5"
  echo "checkpoint=$checkpoint"
} > "$LOG_ROOT/COMPLETE"
cat "$LOG_ROOT/COMPLETE"
