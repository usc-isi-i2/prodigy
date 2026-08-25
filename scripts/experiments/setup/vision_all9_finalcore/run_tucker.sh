#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG="${CONFIG:-${REPO_ROOT}/scripts/experiments/setup/final_core/training.yaml}"
STATE_ROOT="${STATE_ROOT:-${REPO_ROOT}/state/vision_all9_finalcore}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/vision_all9_finalcore}"
VISION_ROOT="${VISION_ROOT:-/dataMeR1/phil/gfm/upstream/VISION}"
GPU="${GPU:-0}"
SEED="${SEED:-0}"
RUN_NAME="all9_s${SEED}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export WANDB_MODE=disabled
PYTHON="${CONDA_PREFIX}/bin/python"
mkdir -p "$STATE_ROOT" "$LOG_ROOT/train" "$LOG_ROOT/eval" "$LOG_ROOT/results" "$LOG_ROOT/launch"
cd "$REPO_ROOT"

used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU")"
util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU")"
if (( used >= 2000 || util >= 10 )); then
  echo "GPU $GPU is occupied: used_mib=$used util_pct=$util" >&2
  exit 2
fi

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "protocol=vision_native_feature_similarity_all9_balanced"
  echo "sources=ukr_rus,covid,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk,facebook_page_reference"
  echo "optimizer_updates=2500"
  echo "batch_size=4"
  echo "total_episodes=10000"
  echo "checkpoints=100,300,900,2500"
  echo "seed=$SEED"
  echo "gpu=$GPU"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/launch/provenance.txt"

checkpoint="$STATE_ROOT/vision/$RUN_NAME/checkpoint/state_dict_2500.pt"
if [[ ! -f "$checkpoint" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -m \
    scripts.experiments.setup.vision_all9_finalcore.train_vision_all9 \
    --config "$CONFIG" --upstream-root "$VISION_ROOT" --state-root "$STATE_ROOT" \
    --run-name "$RUN_NAME" --model-id all9 --sources \
    ukr_rus,covid,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk,facebook_page_reference \
    --steps 2500 --checkpoint-steps 100,300,900,2500 \
    --seed "$SEED" --device 0 > "$LOG_ROOT/train/${RUN_NAME}.log" 2>&1
fi
[[ -f "$checkpoint" ]] || { echo "missing terminal checkpoint $checkpoint" >&2; exit 1; }

result="$LOG_ROOT/results/vision_all9_s${SEED}_step2500_cls.jsonl"
if [[ ! -f "$result" ]]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" -u -m \
    scripts.experiments.setup.icl_arch_matrix.evaluate_adapters \
    --architecture vision --upstream-root "$VISION_ROOT" --state-root "$STATE_ROOT" \
    --run-name "$RUN_NAME" --model-ids all9 --checkpoint-step 2500 \
    --training-seed "$SEED" --eval-episode-seed-offset 0 --include-facebook \
    --results "$result" --device 0 > "$LOG_ROOT/eval/${RUN_NAME}_step2500_cls.log" 2>&1
fi
[[ "$(wc -l < "$result")" == 5 ]] || { echo "expected five CLS rows in $result" >&2; exit 1; }

{
  echo "completed_utc=$(date -u +%FT%TZ)"
  echo "checkpoint=$checkpoint"
  echo "classification_results=$result"
} > "$LOG_ROOT/COMPLETE"
echo "VISION all-nine final-core train+CLS evaluation complete"
