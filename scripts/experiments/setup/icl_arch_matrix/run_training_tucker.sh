#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/prodigy_training.yaml}"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/icl_arch_matrix}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/icl_arch_matrix}"
VISION_ROOT="${VISION_ROOT:-/dataMeR1/phil/gfm/upstream/VISION}"
GILT_ROOT="${GILT_ROOT:-/dataMeR1/phil/gfm/upstream/inductnode}"
GPUS_TEXT="${GPUS:-0 1}"
ARCHITECTURES_TEXT="${ARCHITECTURES:-prodigy vision gilt}"
MODEL_IDS_TEXT="${MODEL_IDS:-}"
RUN_STAMP="${RUN_STAMP:-20260810}"
DRY_RUN="${DRY_RUN:-0}"
read -r -a GPU_IDS <<< "$GPUS_TEXT"
read -r -a ARCHITECTURES <<< "$ARCHITECTURES_TEXT"

for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[01]$ ]] || { echo "refusing non-owned Tucker GPU $gpu" >&2; exit 2; }
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE="${WANDB_MODE:-disabled}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
mkdir -p "$STATE_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/launch"
cd "$REPO_ROOT"

PLAN="$LOG_ROOT/launch/plan.tsv"
"$PYTHON" scripts/experiments/setup/final_core/core_plan.py > "$PLAN"
jobs=()
while IFS=$'\t' read -r model_id _n_sources sources _aliases; do
  [[ "$model_id" == model_id ]] && continue
  if [[ -n "$MODEL_IDS_TEXT" && " $MODEL_IDS_TEXT " != *" $model_id "* ]]; then continue; fi
  for architecture in "${ARCHITECTURES[@]}"; do
    jobs+=("${architecture}:${model_id}:${sources}")
  done
done < "$PLAN"

worker() {
  local worker_index="$1" gpu="$2" item architecture model_id sources index=0
  for item in "${jobs[@]}"; do
    if (( index % ${#GPU_IDS[@]} == worker_index )); then
      IFS=: read -r architecture model_id sources <<< "$item"
      if [[ "$architecture" == prodigy ]]; then
        run_name="archmatrix_prodigy_${model_id}_s0_${RUN_STAMP}"
        run_dir="$STATE_ROOT/prodigy/$run_name"
        checkpoint="$STATE_ROOT/prodigy/$run_name/checkpoint/state_dict_500.ckpt"
        cmd=("$PYTHON" -u experiments/run_single_experiment.py --config "$CONFIG"
             --device "$gpu" --seed 0 --prefix "archmatrix_prodigy_${model_id}_s0"
             --timestamp "$RUN_STAMP" --state_dir "$STATE_ROOT/prodigy"
             --log_dir "$LOG_ROOT/prodigy" --neighbor_sampling_source_subset "$sources")
      else
        run_dir="$STATE_ROOT/$architecture/$model_id"
        checkpoint="$STATE_ROOT/$architecture/$model_id/checkpoint/state_dict_500.pt"
        upstream="$VISION_ROOT"
        [[ "$architecture" == gilt ]] && upstream="$GILT_ROOT"
        cmd=("$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.train_model
             --architecture "$architecture" --upstream-root "$upstream"
             --config "$CONFIG" --model-id "$model_id" --sources "$sources"
             --state-root "$STATE_ROOT" --device "$gpu" --workers 0)
      fi
      if [[ -f "$checkpoint" ]]; then
        echo "[gpu $gpu] SKIP complete $architecture/$model_id"
      elif [[ -e "$run_dir" ]]; then
        echo "[gpu $gpu] REFUSE incomplete existing run $run_dir" >&2
        return 1
      elif [[ "$DRY_RUN" == 1 ]]; then
        printf 'DRY gpu=%s' "$gpu"; printf ' %q' "${cmd[@]}"; printf '\n'
      else
        echo "[gpu $gpu] START $architecture/$model_id utc=$(date -u +%FT%TZ)"
        "${cmd[@]}" > "$LOG_ROOT/train/${architecture}_${model_id}.log" 2>&1
        [[ -f "$checkpoint" ]] || { echo "missing $checkpoint" >&2; return 1; }
        echo "[gpu $gpu] DONE $architecture/$model_id utc=$(date -u +%FT%TZ)"
      fi
    fi
    ((index+=1))
  done
}

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "seed=0"
  echo "steps=500"
  echo "gpus=$GPUS_TEXT"
  echo "architectures=$ARCHITECTURES_TEXT"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance.txt"

pids=()
for gpu_index in "${!GPU_IDS[@]}"; do worker "$gpu_index" "${GPU_IDS[$gpu_index]}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
