#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/icl_arch_native_source_900_seed0}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/log/icl_arch_native_source_900_seed0_eval}"
VISION_ROOT="${VISION_ROOT:-/dataMeR1/phil/gfm/upstream/VISION}"
GILT_ROOT="${GILT_ROOT:-/dataMeR1/phil/gfm/upstream/inductnode}"
STEPS_TEXT="${STEPS_TEXT:-20 60 100 300 900}"
GPUS_TEXT="${GPUS_TEXT:-0 1}"
read -r -a STEPS <<< "$STEPS_TEXT"
read -r -a GPUS <<< "$GPUS_TEXT"
ARCHITECTURES=(vision gilt)
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
mkdir -p "$OUT_ROOT/results" "$OUT_ROOT/logs" "$OUT_ROOT/launch"
cd "$REPO_ROOT"

jobs=()
for architecture in "${ARCHITECTURES[@]}"; do
  jobs+=("random:${architecture}:-:-")
  for item in "${MODELS[@]}"; do
    IFS=: read -r model_id _source <<< "$item"
    for step in "${STEPS[@]}"; do jobs+=("cls:${architecture}:${model_id}:${step}"); done
  done
done

worker() {
  local worker_index="$1" gpu="$2" index=0 job kind architecture model_id step result upstream
  for job in "${jobs[@]}"; do
    if (( index % ${#GPUS[@]} == worker_index )); then
      IFS=: read -r kind architecture model_id step <<< "$job"
      upstream="$VISION_ROOT"; [[ "$architecture" == gilt ]] && upstream="$GILT_ROOT"
      if [[ "$kind" == random ]]; then
        result="$OUT_ROOT/results/random_${architecture}_s0_o0.jsonl"
        if [[ ! -f "$result" ]]; then
          CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m \
            scripts.experiments.setup.icl_arch_matrix.evaluate_adapters \
            --architecture "$architecture" --upstream-root "$upstream" \
            --results "$result" --training-seed 0 --eval-episode-seed-offset 0 \
            --include-facebook --random-init --device 0 \
            > "$OUT_ROOT/logs/random_${architecture}.log" 2>&1
        fi
      else
        result="$OUT_ROOT/results/cls_${architecture}_${model_id}_s0_o0_step${step}.jsonl"
        if [[ ! -f "$result" ]]; then
          CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m \
            scripts.experiments.setup.icl_arch_matrix.evaluate_adapters \
            --architecture "$architecture" --upstream-root "$upstream" \
            --state-root "$STATE_ROOT" --run-name "${model_id}_s0" \
            --model-ids "$model_id" --checkpoint-step "$step" \
            --training-seed 0 --eval-episode-seed-offset 0 --include-facebook \
            --results "$result" --device 0 \
            > "$OUT_ROOT/logs/cls_${architecture}_${model_id}_step${step}.log" 2>&1
        fi
      fi
    fi
    ((index+=1))
  done
}

pids=()
for i in "${!GPUS[@]}"; do worker "$i" "${GPUS[$i]}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
