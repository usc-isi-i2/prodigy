#!/usr/bin/env bash
# Invoke from the isolated research worktree, inside a dedicated tmux session.
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=""
export WANDB_MODE=offline
unset PYTHONPATH
REPRO_CHECKOUT="$(pwd)"
REPRO_ROOT="${1:?Supply a new explicit reproduction directory outside the checkout}"
if [[ -e "$REPRO_ROOT" ]]; then
  echo "Output already exists; inspect before running." >&2
  exit 1
fi
mkdir -p "$REPRO_ROOT"
python scripts/experiments/setup/target_performance_mechanisms/portable_inference/build.py \
  --output "$REPRO_ROOT/build/code" > "$REPRO_ROOT/build_receipt.json"
mkdir "$REPRO_ROOT/extracted"
python -m zipfile -e "$REPRO_ROOT/build/code.zip" "$REPRO_ROOT/extracted"
(
  cd "$REPRO_ROOT/extracted/code"
  python -m unittest test_contracts > "$REPRO_ROOT/synthetic_tests.log" 2>&1
)
python -m scripts.experiments.setup.target_performance_mechanisms.portable_inference.export_private_inputs \
  --package "$REPRO_ROOT/extracted/code" \
  --topology "$REPRO_CHECKOUT/log/role_topology_full_20260907" \
  --messages "$REPRO_CHECKOUT/log/message_content_full_20260907" \
  --dose "$REPRO_CHECKOUT/log/support_dose_full_20260907" \
  --output "$REPRO_ROOT/private_inputs" > "$REPRO_ROOT/export.log" 2>&1
cd "$REPRO_ROOT/extracted/code"
python -m graph_role.run --inputs "$REPRO_ROOT/private_inputs" \
  --output "$REPRO_ROOT/smoke" --max-batches 1 \
  --model-ids memberctl_cp_hk_lowest_sorted_s0 \
  --dataset-ids covid_political/original facebook_page_reference/original twibot20/original \
  > "$REPRO_ROOT/smoke.log" 2>&1
python -m graph_role.run --inputs "$REPRO_ROOT/private_inputs" \
  --output "$REPRO_ROOT/full" > "$REPRO_ROOT/full.log" 2>&1
