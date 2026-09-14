#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/src"
PYTHON="${CONDA_PREFIX}/bin/python"
ROOT=${1:?new output root required}
PARENT=${2:?completed parent root required}
if [[ -f "$ROOT/results/COMPLETE.json" ]]; then
 echo "Completed run exists: $ROOT"; exit 0
fi
if [[ -e "$ROOT" ]]; then
 echo "Refusing existing partial output: $ROOT" >&2; exit 1
fi
"$PYTHON" -m mixture_scaling.async_extension plan --root "$ROOT" --parent-root "$PARENT"
mkdir -p "$ROOT/log"
pids=()
for item in 'kd_extended 0' 'ukraine_only 1'; do
 read -r arm gpu <<< "$item"
 "$PYTHON" -u -m mixture_scaling.async_extension train --arm "$arm" --device "$gpu" --root "$ROOT" --parent-root "$PARENT" > "$ROOT/log/$arm.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == 1 ]]; then exit 1; fi
# Freeze the source-only selection manifest before any downstream evaluation.
"$PYTHON" -m mixture_scaling.async_extension prepare_eval --root "$ROOT" --parent-root "$PARENT" > "$ROOT/log/selection.log" 2>&1
pids=()
for gpu in 0 1 2 3; do
 "$PYTHON" -u -m mixture_scaling.async_extension eval --device "$gpu" --root "$ROOT" --parent-root "$PARENT" > "$ROOT/log/eval_gpu$gpu.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == 1 ]]; then exit 1; fi
"$PYTHON" -m mixture_scaling.async_extension aggregate --root "$ROOT" --parent-root "$PARENT"
