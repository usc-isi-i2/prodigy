#!/usr/bin/env bash
# Preserve, validate, and plot the completed three-seed VISION mixture campaign.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ANALYSIS_ROOT="${REPO_ROOT}/scripts/experiments/analysis/synthesis/cross_experiment/native_model_result_matrix"
SEED0_ROOT="${SEED0_ROOT:-${ANALYSIS_ROOT}/data/vision_native_mixture_raw}"
SEED1_ROOT="${SEED1_ROOT:-/dataMeR1/phil/gfm/prodigy-vision-mixture-seeds/log/vision_native_mixture_finalcore_s1}"
SEED2_ROOT="${SEED2_ROOT:-/dataMeR1/phil/gfm/prodigy-vision-mixture-seeds/log/vision_native_mixture_finalcore_s2}"
STATUS_ROOT="${STATUS_ROOT:-${REPO_ROOT}/log/paper_vision_three_seed_postprocess}"
mkdir -p "$STATUS_ROOT"

for root in "$SEED1_ROOT" "$SEED2_ROOT"; do
  while [[ ! -f "$root/COMPLETE" ]]; do
    sleep 60
  done
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export MPLBACKEND=Agg
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${CONDA_PREFIX}/bin/python"

for seed in 1 2; do
  source_root_var="SEED${seed}_ROOT"
  source_root="${!source_root_var}"
  raw_root="${ANALYSIS_ROOT}/data/vision_native_mixture_seed${seed}_raw"
  mkdir -p "$raw_root"
  mapfile -t files < <(find "$source_root/results" -maxdepth 1 -type f -name '*.jsonl' | sort)
  [[ "${#files[@]}" -eq 48 ]] || {
    echo "seed $seed: expected 48 result files, found ${#files[@]}" >&2
    exit 1
  }
  for result in "${files[@]}"; do
    [[ "$(wc -l < "$result")" -eq 5 ]] || {
      echo "seed $seed: expected five rows in $result" >&2
      exit 1
    }
    cp -p "$result" "$raw_root/$(basename "$result")"
  done
done

cd "$REPO_ROOT"
"$PYTHON" \
  scripts/experiments/analysis/synthesis/cross_experiment/native_model_result_matrix/analyze_vision_mixture_three_seed.py \
  --mixture-root "$SEED0_ROOT" \
  --mixture-root "$ANALYSIS_ROOT/data/vision_native_mixture_seed1_raw" \
  --mixture-root "$ANALYSIS_ROOT/data/vision_native_mixture_seed2_raw" \
  --all9-root "$ANALYSIS_ROOT/data/vision_all9_saturation_raw" \
  --output "$ANALYSIS_ROOT" \
  > "$STATUS_ROOT/vision_analysis.log" 2>&1

"$PYTHON" \
  scripts/experiments/analysis/synthesis/cross_experiment/native_model_result_matrix/plot_native_mixture_ladder_three_seed.py \
  --vision "$ANALYSIS_ROOT/data/vision_native_mixture_three_seed_per_target.csv" \
  --output-root "$ANALYSIS_ROOT" \
  > "$STATUS_ROOT/cross_family_analysis.log" 2>&1

{
  echo "status=complete"
  echo "commit=$(git rev-parse HEAD)"
  echo "completed_utc=$(date -u +%FT%TZ)"
  echo "vision_physical_cells=780"
  echo "vision_order_expanded_cells=900"
  echo "cross_family_cells=1035"
} > "$STATUS_ROOT/COMPLETE"
cat "$STATUS_ROOT/COMPLETE"
