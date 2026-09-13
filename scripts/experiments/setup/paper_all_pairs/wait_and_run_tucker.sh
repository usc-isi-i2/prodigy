#!/usr/bin/env bash
# Overlap pair training with the long fixed-exposure tail, then keep pair
# evaluation behind the primary and mechanism audits.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLAGSHIP_ROOT="${FLAGSHIP_ROOT:-/dataMeR1/phil/gfm/prodigy-paper-remainder-opt/log/paper_flagship_ladders/20260908}"
CORE_ROOT="${CORE_ROOT:-/dataMeR1/phil/gfm/prodigy-paper-remainder-opt/log/paper_three_seed_remainder/20260908/core_nm_evaluation}"
GPUS_TEXT="${GPUS:-}"
RUN_STAMP="${RUN_STAMP:-20260908}"
RUN_ROOT="${RUN_ROOT:-${SCRIPT_DIR}/../../../../log/paper_all_pairs/${RUN_STAMP}}"
TRAIN_COMPLETE="${RUN_ROOT}/training_complete_utc.txt"
MECHANISM_TRAIN_COMPLETE="${MECHANISM_TRAIN_COMPLETE:-/dataMeR1/phil/gfm/prodigy-paper-mechanism/log/paper_mechanism_sweeps/${RUN_STAMP}/training_complete_utc.txt}"
MECHANISM_COMPLETE="${MECHANISM_COMPLETE:-/dataMeR1/phil/gfm/prodigy-paper-mechanism/log/paper_mechanism_sweeps/${RUN_STAMP}/complete_utc.txt}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy

wait_for_primary() {
  while true; do
  if "${CONDA_PREFIX}/bin/python" - "$FLAGSHIP_ROOT" "$CORE_ROOT" <<'PY'
import csv
import json
import math
from pathlib import Path
import sys

flagship, core = map(Path, sys.argv[1:])
training = json.load(open(flagship / "seeds_1-2/status.json", encoding="utf-8"))
if training.get("status") != "complete" or len(training.get("finished", [])) != 96:
    raise SystemExit(1)
if any(row.get("exitcode") != 0 for row in training["finished"]):
    raise SystemExit(1)
nm = json.load(open(flagship / "nm_evaluation/status.json", encoding="utf-8"))
if nm.get("status") != "complete":
    raise SystemExit(1)
nm_rows = []
for path in sorted((flagship / "nm_evaluation/cells").glob("*/*.json")):
    nm_rows.append(json.load(open(path, encoding="utf-8")))
nm_keys = [(row.get("model_id"), row.get("target")) for row in nm_rows]
if len(nm_rows) != 864 or len(set(nm_keys)) != 864:
    raise SystemExit(1)
if any(row.get("protocol") != "nmi_fixed_nm_v1" or int(row.get("episodes", 0)) != 512
       or not all(math.isfinite(float(row[key])) for key in ("roc_auc", "accuracy", "loss"))
       for row in nm_rows):
    raise SystemExit(1)
for target in {row["target"] for row in nm_rows}:
    if len({row["fingerprint"] for row in nm_rows if row["target"] == target}) != 1:
        raise SystemExit(1)
cls = flagship / "classification_evaluation/classification_long.tsv"
with open(cls, encoding="utf-8") as handle:
    if sum(1 for _ in csv.DictReader(handle, delimiter="\t")) != 600:
        raise SystemExit(1)
core_status = json.load(open(core / "status.json", encoding="utf-8"))
if core_status.get("status") != "complete" or core_status.get("audit", {}).get("cells") != 738:
    raise SystemExit(1)
PY
    then
      break
    fi
    sleep 60
  done
}

wait_for_stable_gpus() {
  local stable=0 gpu_csv
  while (( stable < 4 )); do
    gpu_csv="$(tr ' ' ',' <<< "$1")"
    if nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -i "$gpu_csv" |
        awk -F, '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($1>1000 || $2>10) bad=1} END{exit bad}'; then
      stable=$((stable + 1))
    else
      stable=0
    fi
    sleep 30
  done
}

select_eval_gpus() {
  if [[ -n "$GPUS_TEXT" ]]; then
    read -r -a supplied <<< "$GPUS_TEXT"
    (( ${#supplied[@]} >= 2 )) || { echo "pair evaluation requires at least two GPUs" >&2; return 1; }
    for gpu in "${supplied[@]}"; do
      [[ "$gpu" =~ ^[23]$ ]] || { echo "refusing GPU $gpu: this campaign is restricted to 2-3" >&2; return 1; }
    done
    [[ "${supplied[*]}" == "2 3" || "${supplied[*]}" == "3 2" ]] \
      || { echo "pair evaluation requires exactly GPUs 2 and 3" >&2; return 1; }
    echo "$GPUS_TEXT"
    return
  fi
  while true; do
    available=()
    for gpu in 2 3; do
      values="$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' ')"
      IFS=, read -r used util <<< "$values"
      if (( used < 1000 && util < 10 )); then available+=("$gpu"); fi
    done
    if (( ${#available[@]} >= 2 )); then
      echo "${available[*]}"
      return
    fi
    sleep 30
  done
}

while [[ ! -f "$MECHANISM_TRAIN_COMPLETE" ]]; do sleep 30; done
if [[ ! -f "$TRAIN_COMPLETE" ]]; then
  if tmux has-session -t paper-optimized-queue 2>/dev/null; then
    PHASE=train GPUS="2 3" MODELS_PER_GPU=10 WORKER_BUDGET=80 \
      bash "$SCRIPT_DIR/run_tucker.sh"
  else
    while [[ ! -f "$MECHANISM_COMPLETE" ]]; do sleep 30; done
    wait_for_primary
    selected_gpus="$(select_eval_gpus)"
    read -r -a selected_gpu_ids <<< "$selected_gpus"
    wait_for_stable_gpus "$selected_gpus"
    PHASE=train GPUS="$selected_gpus" MODELS_PER_GPU=14 \
      WORKER_BUDGET="$((${#selected_gpu_ids[@]} * 56))" \
      bash "$SCRIPT_DIR/run_tucker.sh"
  fi
fi

while [[ ! -f "$MECHANISM_COMPLETE" ]]; do sleep 30; done
wait_for_primary
selected_gpus="$(select_eval_gpus)"
wait_for_stable_gpus "$selected_gpus"
PHASE=eval GPUS="$selected_gpus" exec bash "$SCRIPT_DIR/run_tucker.sh"
