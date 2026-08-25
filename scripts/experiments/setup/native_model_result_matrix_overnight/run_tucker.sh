#!/usr/bin/env bash
# Resumable priority-ordered orchestration for the native-model result matrix.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/native_model_result_matrix_overnight}"
mkdir -p "$LOG_ROOT"
cd "$REPO_ROOT"

for gpu in 2 3; do
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu")"
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu")"
  if (( used >= 2000 || util >= 10 )); then
    echo "GPU $gpu is occupied at launch: used_mib=$used util_pct=$util" >&2
    exit 2
  fi
done

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "priority_1=matched_adaptation_efficiency"
  echo "priority_2=vision_checkpoint_only_native_cross_ssl"
  echo "priority_3=vision_genuinely_missing_native_mixture_training"
  echo "physical_gpus=2,3"
  echo "started_utc=$(date -u +%FT%TZ)"
} > "$LOG_ROOT/provenance.txt"

if [[ ! -f "$LOG_ROOT/ADAPTATION_COMPLETE" ]]; then
  GPUS_TEXT="2 3" bash \
    scripts/experiments/setup/adaptation_efficiency/run_tucker.sh \
    > "$LOG_ROOT/adaptation.log" 2>&1
  date -u +%FT%TZ > "$LOG_ROOT/ADAPTATION_COMPLETE"
fi

if [[ ! -f "$LOG_ROOT/VISION_CROSS_SSL_COMPLETE" ]]; then
  GPU_A=2 GPU_B=3 bash \
    scripts/experiments/setup/vision_native_cross_ssl/run_tucker.sh \
    > "$LOG_ROOT/vision_cross_ssl.log" 2>&1
  date -u +%FT%TZ > "$LOG_ROOT/VISION_CROSS_SSL_COMPLETE"
fi

if [[ ! -f "$LOG_ROOT/VISION_MIXTURE_COMPLETE" ]]; then
  GPU_A=2 GPU_B=3 bash \
    scripts/experiments/setup/vision_native_mixture_finalcore/run_tucker.sh \
    > "$LOG_ROOT/vision_mixture.log" 2>&1
  date -u +%FT%TZ > "$LOG_ROOT/VISION_MIXTURE_COMPLETE"
fi

{
  echo "completed_utc=$(date -u +%FT%TZ)"
  echo "adaptation_complete=$(cat "$LOG_ROOT/ADAPTATION_COMPLETE")"
  echo "vision_cross_ssl_complete=$(cat "$LOG_ROOT/VISION_CROSS_SSL_COMPLETE")"
  echo "vision_mixture_complete=$(cat "$LOG_ROOT/VISION_MIXTURE_COMPLETE")"
} > "$LOG_ROOT/COMPLETE"
cat "$LOG_ROOT/COMPLETE"
