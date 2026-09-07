#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/prodigy_training_2000.yaml}"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/icl_arch_saturation_2000}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/icl_arch_saturation_2000}"
VISION_ROOT="${VISION_ROOT:-/dataMeR1/phil/gfm/upstream/VISION}"
GILT_ROOT="${GILT_ROOT:-/dataMeR1/phil/gfm/upstream/inductnode}"
RUN_STAMP="${RUN_STAMP:-20260815}"
STEPS="${STEPS:-2000}"
CHECKPOINTS="${CHECKPOINTS:-20,60,100,300,900,2000}"
SEEDS_TEXT="${SEEDS_TEXT:-0 1 2}"
GPUS_TEXT="${GPUS_TEXT:-0 1}"
MODELS=(
  "ss_covid_political:covid_political"
  "ss_election2020:election2020"
  "ss_ukr_rus_suspended:ukr_rus_suspended"
  "ss_twibot20:twibot20"
  "ss_facebook_page_reference:facebook_page_reference"
)
read -r -a SEEDS <<< "$SEEDS_TEXT"
ARCHITECTURES=(vision prodigy gilt)
read -r -a GPUS <<< "$GPUS_TEXT"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
PYTHON="${CONDA_PREFIX}/bin/python"
mkdir -p "$STATE_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/launch"
cd "$REPO_ROOT"

jobs=()
for architecture in "${ARCHITECTURES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for item in "${MODELS[@]}"; do jobs+=("${architecture}:${seed}:${item}"); done
  done
done

worker() {
  local worker_index="$1" gpu="$2" index=0 item architecture seed model_id sources
  for item in "${jobs[@]}"; do
    if (( index % ${#GPUS[@]} == worker_index )); then
      IFS=: read -r architecture seed model_id sources <<< "$item"
      echo "[gpu $gpu] START $architecture/$model_id seed=$seed utc=$(date -u +%FT%TZ)"
      if [[ "$architecture" == prodigy ]]; then
        run_name="archsat_prodigy_${model_id}_s${seed}_${RUN_STAMP}"
        checkpoint="$STATE_ROOT/prodigy/$run_name/checkpoint/state_dict_${STEPS}.ckpt"
        if [[ -f "$checkpoint" ]]; then
          echo "[gpu $gpu] SKIP complete $architecture/$model_id seed=$seed"
        elif [[ -e "$STATE_ROOT/prodigy/$run_name" ]]; then
          echo "REFUSE incomplete $STATE_ROOT/prodigy/$run_name" >&2; return 1
        else
          CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.train_prodigy \
            --config "$CONFIG" --device 0 --seed "$seed" \
            --prefix "archsat_prodigy_${model_id}_s${seed}" --timestamp "$RUN_STAMP" \
            --state_dir "$STATE_ROOT/prodigy" --log_dir "$LOG_ROOT/prodigy" \
            --neighbor_sampling_source_subset "$sources" \
            > "$LOG_ROOT/train/${architecture}_${model_id}_s${seed}.log" 2>&1
        fi
      else
        run_name="${model_id}_s${seed}"
        checkpoint="$STATE_ROOT/$architecture/$run_name/checkpoint/state_dict_${STEPS}.pt"
        upstream="$VISION_ROOT"; [[ "$architecture" == gilt ]] && upstream="$GILT_ROOT"
        if [[ -f "$checkpoint" ]]; then
          echo "[gpu $gpu] SKIP complete $architecture/$model_id seed=$seed"
        elif [[ -e "$STATE_ROOT/$architecture/$run_name" ]]; then
          echo "REFUSE incomplete $STATE_ROOT/$architecture/$run_name" >&2; return 1
        else
          CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.train_model \
            --architecture "$architecture" --upstream-root "$upstream" --config "$CONFIG" \
            --model-id "$model_id" --sources "$sources" --state-root "$STATE_ROOT" \
            --run-name "$run_name" --seed "$seed" --steps "$STEPS" \
            --checkpoint-steps "$CHECKPOINTS" --device 0 --workers 0 \
            > "$LOG_ROOT/train/${architecture}_${model_id}_s${seed}.log" 2>&1
        fi
      fi
      [[ -f "$checkpoint" ]] || { echo "missing $checkpoint" >&2; return 1; }
      echo "[gpu $gpu] DONE $architecture/$model_id seed=$seed utc=$(date -u +%FT%TZ)"
    fi
    ((index+=1))
  done
}

{
  echo "commit=$(git rev-parse HEAD)"
  echo "steps=$STEPS"
  echo "checkpoints=$CHECKPOINTS"
  echo "architectures=${ARCHITECTURES[*]}"
  echo "seeds=${SEEDS[*]}"
  echo "models=${MODELS[*]}"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance.txt"

pids=()
for i in "${!GPUS[@]}"; do worker "$i" "${GPUS[$i]}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
