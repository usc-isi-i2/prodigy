#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/icl_arch_native_source_900_seed0}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/icl_arch_native_source_900_seed0}"
VISION_ROOT="${VISION_ROOT:-/dataMeR1/phil/gfm/upstream/VISION}"
GILT_ROOT="${GILT_ROOT:-/dataMeR1/phil/gfm/upstream/inductnode}"
STEPS="${STEPS:-900}"
CHECKPOINTS="${CHECKPOINTS:-20,60,100,300,900}"
GPUS_TEXT="${GPUS_TEXT:-0 1}"
ARCHITECTURES_TEXT="${ARCHITECTURES_TEXT:-vision gilt}"
read -r -a GPUS <<< "$GPUS_TEXT"
read -r -a ARCHITECTURES <<< "$ARCHITECTURES_TEXT"
MODELS=(
  "ss_covid_political:covid_political"
  "ss_election2020:election2020"
  "ss_ukr_rus_suspended:ukr_rus_suspended"
  "ss_twibot20:twibot20"
  "ss_facebook_page_reference:facebook_page_reference"
)

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
  for item in "${MODELS[@]}"; do jobs+=("${architecture}:${item}"); done
done

worker() {
  local worker_index="$1" gpu="$2" index=0 job architecture model_id source upstream checkpoint
  for job in "${jobs[@]}"; do
    if (( index % ${#GPUS[@]} == worker_index )); then
      IFS=: read -r architecture model_id source <<< "$job"
      checkpoint="$STATE_ROOT/$architecture/${model_id}_s0/checkpoint/state_dict_${STEPS}.pt"
      if [[ -f "$checkpoint" ]]; then
        echo "[gpu $gpu] SKIP complete $architecture/$model_id"
      elif [[ -e "$STATE_ROOT/$architecture/${model_id}_s0" ]]; then
        echo "REFUSE incomplete $STATE_ROOT/$architecture/${model_id}_s0" >&2
        return 1
      else
        upstream="$VISION_ROOT"; [[ "$architecture" == gilt ]] && upstream="$GILT_ROOT"
        echo "[gpu $gpu] START $architecture/$model_id utc=$(date -u +%FT%TZ)"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m \
          scripts.experiments.setup.icl_arch_matrix.train_native_source_model \
          --architecture "$architecture" --upstream-root "$upstream" \
          --source "$source" --model-id "$model_id" --state-root "$STATE_ROOT" \
          --steps "$STEPS" --checkpoint-steps "$CHECKPOINTS" --seed 0 --device 0 \
          > "$LOG_ROOT/train/${architecture}_${model_id}_s0.log" 2>&1
        echo "[gpu $gpu] DONE $architecture/$model_id utc=$(date -u +%FT%TZ)"
      fi
      [[ -f "$checkpoint" ]] || { echo "missing $checkpoint" >&2; return 1; }
    fi
    ((index+=1))
  done
}

{
  echo "commit=$(git rev-parse HEAD)"
  echo "protocol=source_confined_native_tasks"
  echo "vision_task=feature_similarity_pseudo_episodes"
  echo "gilt_task=episodic_node_classification_train_split_only"
  echo "steps=$STEPS"
  echo "checkpoints=$CHECKPOINTS"
  echo "gpus=${GPUS[*]}"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance.txt"

pids=()
for i in "${!GPUS[@]}"; do worker "$i" "${GPUS[$i]}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
