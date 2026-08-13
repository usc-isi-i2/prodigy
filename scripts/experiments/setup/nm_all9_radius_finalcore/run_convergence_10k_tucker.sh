#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/nm_all9_radius_finalcore_10k}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/nm_all9_radius_finalcore_10k}"
FEASIBILITY_REPORT="${FEASIBILITY_REPORT:-${REPO_ROOT}/log/nm_all9_radius_finalcore/preflight/feasibility.json}"
GPUS_TEXT="${GPUS:-0 1}"
RUN_STAMP="${RUN_STAMP:-20260812}"
DRY_RUN="${DRY_RUN:-0}"
ARM_IDS_TEXT="${ARM_IDS:-}"
read -r -a GPU_IDS <<< "$GPUS_TEXT"

[[ "$DRY_RUN" =~ ^(0|1)$ ]] || { echo "DRY_RUN must be 0 or 1" >&2; exit 2; }
(( ${#GPU_IDS[@]} > 0 )) || { echo "at least one GPU is required" >&2; exit 2; }
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^(0|1)$ ]] || {
    echo "refusing GPU $gpu: current Tucker ownership is GPUs 0 and 1 only" >&2
    exit 2
  }
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

mkdir -p "$STATE_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/launch"
PLAN="$LOG_ROOT/launch/plan.tsv"
cd "$REPO_ROOT"
if [[ -z "$ARM_IDS_TEXT" ]]; then
  "$PYTHON" "$SCRIPT_DIR/radius_plan.py" > "$PLAN"
  [[ "$(($(wc -l < "$PLAN") - 1))" == 3 ]] || {
    echo "default convergence plan must contain exactly three arms" >&2
    exit 2
  }
else
  "$PYTHON" - "$SCRIPT_DIR" "$ARM_IDS_TEXT" > "$PLAN" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1])))
from radius_plan import get_arm
print("arm_id\tconfig\tradii")
seen = set()
for arm_id in sys.argv[2].split():
    if arm_id in seen:
        raise ValueError(f"duplicate arm {arm_id}")
    seen.add(arm_id)
    arm = get_arm(arm_id)
    print(f"{arm.arm_id}\t{arm.config}\t{','.join(arm.radii)}")
PY
fi

if [[ "$DRY_RUN" != 1 ]]; then
  [[ -f "$FEASIBILITY_REPORT" ]] || {
    echo "missing feasibility gate $FEASIBILITY_REPORT" >&2
    exit 1
  }
  "$PYTHON" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); assert p.get("ready") is True, "feasibility report is not ready"' \
    "$FEASIBILITY_REPORT"
  for gpu in "${GPU_IDS[@]}"; do
    used="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    (( used <= 1000 )) || {
      echo "GPU $gpu is busy (${used} MiB); refusing launch" >&2
      exit 1
    }
  done
  available_kib="$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)"
  required_gib=$(( ${#GPU_IDS[@]} * 125 + 100 ))
  required_kib=$((required_gib * 1024 * 1024))
  (( available_kib >= required_kib )) || {
    echo "insufficient host RAM: need ${required_gib} GiB available" >&2
    exit 1
  }
fi

jobs=()
while IFS=$'\t' read -r arm_id config _radii; do
  [[ "$arm_id" == arm_id ]] && continue
  jobs+=("${arm_id}:${config}")
done < "$PLAN"

validate_checkpoint() {
  "$PYTHON" - "$1" <<'PY'
import math
import sys
import torch

path = sys.argv[1]
try:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
except TypeError:
    checkpoint = torch.load(path, map_location="cpu")
training = checkpoint.get("_training_checkpoint")
assert isinstance(training, dict), "missing full training state"
assert training.get("format_version") == 1
assert training.get("completed_steps") == 10000
assert training.get("exact_resume_supported") is True
assert "optimizer" in training and "rng" in training
assert training.get("train_batch_sampler") is not None
for tensor in checkpoint["model"].values():
    if torch.is_tensor(tensor) and not torch.isfinite(tensor).all():
        raise AssertionError("non-finite model tensor")
print(f"VALID full-state terminal checkpoint {path}")
PY
}

worker() {
  local worker_index="$1" gpu="$2" item arm_id config index=0
  for item in "${jobs[@]}"; do
    if (( index % ${#GPU_IDS[@]} == worker_index )); then
      IFS=: read -r arm_id config <<< "$item"
      prefix="radiusfc10k_${arm_id}_s0"
      run_name="${prefix}_${RUN_STAMP}"
      run_dir="$STATE_ROOT/$run_name"
      checkpoint="$run_dir/checkpoint/state_dict_10000.ckpt"
      training_checkpoint="$run_dir/checkpoint/training_state_10000.ckpt"
      if [[ -f "$checkpoint" && -f "$training_checkpoint" ]]; then
        validate_checkpoint "$training_checkpoint"
        echo "[gpu $gpu] SKIP complete $run_name"
        ((index+=1)); continue
      fi
      if [[ -e "$run_dir" ]]; then
        echo "[gpu $gpu] REFUSE incomplete existing run $run_dir" >&2
        return 1
      fi
      cmd=("$PYTHON" -u experiments/run_single_experiment.py
           --config "$config" --device "$gpu" --seed 0
           --prefix "$prefix" --timestamp "$RUN_STAMP"
           --state_dir "$STATE_ROOT" --log_dir "$LOG_ROOT"
           --dataset_len_cap 10000 --epochs 1
           --checkpoint_steps 2500,5000,7500,10000
           --checkpoint_step 100000 --eval_step 100000 --workers 0)
      if [[ "$DRY_RUN" == 1 ]]; then
        printf 'DRY gpu=%s arm=%s' "$gpu" "$arm_id"
        printf ' %q' "${cmd[@]}"
        printf '\n'
      else
        echo "[gpu $gpu] START arm=$arm_id seed=0 utc=$(date -u +%FT%TZ)"
        "${cmd[@]}" > "$LOG_ROOT/train/${run_name}.log" 2>&1
        for step in 2500 5000 7500 10000; do
          [[ -f "$run_dir/checkpoint/state_dict_${step}.ckpt" ]] || {
            echo "missing weights checkpoint at step $step" >&2
            return 1
          }
          full="$run_dir/checkpoint/training_state_${step}.ckpt"
          [[ -f "$full" ]] || {
            echo "missing full-state checkpoint at step $step" >&2
            return 1
          }
          if [[ "$step" == 10000 ]]; then validate_checkpoint "$full"; fi
        done
        echo "[gpu $gpu] DONE $run_name utc=$(date -u +%FT%TZ)"
      fi
    fi
    ((index+=1))
  done
}

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "seed=0"
  echo "arm_ids=${ARM_IDS_TEXT:-global radius_mix close_only}"
  echo "gpus=$GPUS_TEXT"
  echo "steps=10000"
  echo "checkpoint_steps=2500,5000,7500,10000"
  echo "workers=0"
  echo "feasibility_report=$FEASIBILITY_REPORT"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance.txt"

pids=()
for gpu_index in "${!GPU_IDS[@]}"; do
  worker "$gpu_index" "${GPU_IDS[$gpu_index]}" & pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
