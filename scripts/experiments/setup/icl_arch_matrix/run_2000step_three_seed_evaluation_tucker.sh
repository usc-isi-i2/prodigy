#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG="${CONFIG:-${SCRIPT_DIR}/prodigy_training_2000.yaml}"
TRAIN_STATE_ROOT="${TRAIN_STATE_ROOT:-/dataMeR1/phil/gfm/prodigy-archsat/state/icl_arch_saturation_2000}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/log/icl_arch_saturation_2000_eval}"
VISION_ROOT="${VISION_ROOT:-/dataMeR1/phil/gfm/upstream/VISION}"
GILT_ROOT="${GILT_ROOT:-/dataMeR1/phil/gfm/upstream/inductnode}"
RUN_STAMP="${RUN_STAMP:-20260815}"
FINAL_STEP="${FINAL_STEP:-2000}"
CHECKPOINTS_TEXT="${CHECKPOINTS_TEXT:-20 60 100 300 900 2000}"
CHECKPOINTS_WITH_ZERO="${CHECKPOINTS_WITH_ZERO:-0,20,60,100,300,900,2000}"
SEEDS_TEXT="${SEEDS_TEXT:-0 1 2}"
OFFSETS_TEXT="${OFFSETS_TEXT:-0 1 2}"
read -r -a CHECKPOINTS <<< "$CHECKPOINTS_TEXT"
MODELS=(
  "ss_covid_political:covid_political"
  "ss_election2020:election2020"
  "ss_ukr_rus_suspended:ukr_rus_suspended"
  "ss_twibot20:twibot20"
  "ss_facebook_page_reference:facebook_page_reference"
)
read -r -a SEEDS <<< "$SEEDS_TEXT"
read -r -a OFFSETS <<< "$OFFSETS_TEXT"
ARCHITECTURES=(prodigy vision gilt)
GPUS=(0 1)

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
PYTHON="${CONDA_PREFIX}/bin/python"
mkdir -p "$OUT_ROOT/results" "$OUT_ROOT/logs" "$OUT_ROOT/state" "$OUT_ROOT/launch"
cd "$REPO_ROOT"

for architecture in "${ARCHITECTURES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for item in "${MODELS[@]}"; do
      IFS=: read -r model_id _source <<< "$item"
      if [[ "$architecture" == prodigy ]]; then
        final="$TRAIN_STATE_ROOT/prodigy/archsat_prodigy_${model_id}_s${seed}_${RUN_STAMP}/checkpoint/state_dict_${FINAL_STEP}.ckpt"
      else
        final="$TRAIN_STATE_ROOT/$architecture/${model_id}_s${seed}/checkpoint/state_dict_${FINAL_STEP}.pt"
      fi
      [[ -f "$final" ]] || { echo "training incomplete: $final" >&2; exit 3; }
    done
  done
done

jobs=()
for architecture in "${ARCHITECTURES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    for offset in "${OFFSETS[@]}"; do
      jobs+=("random:${architecture}:${seed}:${offset}:-:-:-")
      for item in "${MODELS[@]}"; do
        IFS=: read -r model_id source <<< "$item"
        jobs+=("nm:${architecture}:${seed}:${offset}:${model_id}:${source}:-")
        for step in "${CHECKPOINTS[@]}"; do
          jobs+=("cls:${architecture}:${seed}:${offset}:${model_id}:${source}:${step}")
        done
      done
    done
  done
done

worker() {
  local wi="$1" gpu="$2" index=0 job kind architecture seed offset model_id source step result upstream
  for job in "${jobs[@]}"; do
    if (( index % ${#GPUS[@]} == wi )); then
      IFS=: read -r kind architecture seed offset model_id source step <<< "$job"
      result="$OUT_ROOT/results/${kind}_${architecture}_${model_id}_s${seed}_o${offset}_step${step}.jsonl"
      if [[ -f "$result" ]]; then ((index+=1)); continue; fi
      echo "[gpu $gpu] START $kind $architecture $model_id seed=$seed offset=$offset step=$step utc=$(date -u +%FT%TZ)"
      if [[ "$kind" == random ]]; then
        result="$OUT_ROOT/results/random_${architecture}_s${seed}_o${offset}.jsonl"
        [[ -f "$result" ]] && { ((index+=1)); continue; }
        if [[ "$architecture" == prodigy ]]; then
          CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy \
            --config "$CONFIG" --state-root "$TRAIN_STATE_ROOT" --eval-state-root "$OUT_ROOT/state" \
            --log-root "$OUT_ROOT/logs" --results "$result" --run-stamp "$RUN_STAMP" \
            --training-seed "$seed" --eval-episode-seed-offset "$offset" --include-facebook \
            --random-init --device 0 > "$OUT_ROOT/logs/random_${architecture}_s${seed}_o${offset}.log" 2>&1
        else
          upstream="$VISION_ROOT"; [[ "$architecture" == gilt ]] && upstream="$GILT_ROOT"
          CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_adapters \
            --architecture "$architecture" --upstream-root "$upstream" --results "$result" \
            --training-seed "$seed" --eval-episode-seed-offset "$offset" --include-facebook \
            --random-init --device 0 > "$OUT_ROOT/logs/random_${architecture}_s${seed}_o${offset}.log" 2>&1
        fi
      elif [[ "$kind" == nm ]]; then
        if [[ "$architecture" == prodigy ]]; then
          CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy_nm_trajectory \
            --config "$CONFIG" --state-root "$TRAIN_STATE_ROOT" --eval-state-root "$OUT_ROOT/state" \
            --log-root "$OUT_ROOT/logs" --results "$result" --run-stamp "$RUN_STAMP" \
            --model-id "$model_id" --training-seed "$seed" --checkpoint-layout saturation \
            --checkpoint-steps "$CHECKPOINTS_WITH_ZERO" --eval-episode-seed-offset "$offset" \
            --device 0 > "$OUT_ROOT/logs/nm_${architecture}_${model_id}_s${seed}_o${offset}.log" 2>&1
        else
          upstream="$VISION_ROOT"; [[ "$architecture" == gilt ]] && upstream="$GILT_ROOT"
          CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_adapter_nm_trajectory \
            --architecture "$architecture" --upstream-root "$upstream" --config "$CONFIG" \
            --state-root "$TRAIN_STATE_ROOT" --model-id "$model_id" --source "$source" \
            --training-seed "$seed" --checkpoint-steps "$CHECKPOINTS_WITH_ZERO" \
            --eval-episode-seed-offset "$offset" --results "$result" --device 0 \
            > "$OUT_ROOT/logs/nm_${architecture}_${model_id}_s${seed}_o${offset}.log" 2>&1
        fi
      else
        if [[ "$architecture" == prodigy ]]; then
          CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy \
            --config "$CONFIG" --state-root "$TRAIN_STATE_ROOT" --eval-state-root "$OUT_ROOT/state" \
            --log-root "$OUT_ROOT/logs" --results "$result" --run-stamp "$RUN_STAMP" \
            --training-seed "$seed" --checkpoint-layout saturation --checkpoint-step "$step" \
            --model-ids "$model_id" --eval-episode-seed-offset "$offset" --include-facebook \
            --device 0 > "$OUT_ROOT/logs/cls_${architecture}_${model_id}_s${seed}_o${offset}_step${step}.log" 2>&1
        else
          upstream="$VISION_ROOT"; [[ "$architecture" == gilt ]] && upstream="$GILT_ROOT"
          CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u -m scripts.experiments.setup.icl_arch_matrix.evaluate_adapters \
            --architecture "$architecture" --upstream-root "$upstream" --state-root "$TRAIN_STATE_ROOT" \
            --run-name "${model_id}_s${seed}" --model-ids "$model_id" --checkpoint-step "$step" \
            --training-seed "$seed" --eval-episode-seed-offset "$offset" --include-facebook \
            --results "$result" --device 0 \
            > "$OUT_ROOT/logs/cls_${architecture}_${model_id}_s${seed}_o${offset}_step${step}.log" 2>&1
        fi
      fi
      echo "[gpu $gpu] DONE $kind $architecture $model_id seed=$seed offset=$offset step=$step utc=$(date -u +%FT%TZ)"
    fi
    ((index+=1))
  done
}

{
  echo "commit=$(git rev-parse HEAD)"
  echo "training_state_root=$TRAIN_STATE_ROOT"
  echo "checkpoints=$CHECKPOINTS_WITH_ZERO"
  echo "training_seeds=${SEEDS[*]}"
  echo "evaluation_offsets=${OFFSETS[*]}"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$OUT_ROOT/launch/provenance.txt"

pids=()
for i in "${!GPUS[@]}"; do worker "$i" "${GPUS[$i]}" & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
