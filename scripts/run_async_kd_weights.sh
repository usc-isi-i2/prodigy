#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/src"
PYTHON="${CONDA_PREFIX}/bin/python"
ROOT=${1:?new output root required}
PARENT=${2:?original KD parent root required}
REFERENCE=${3:?completed extension root required}
if [[ -f "$ROOT/results/COMPLETE.json" ]]; then echo "Completed run exists: $ROOT"; exit 0; fi
if [[ -e "$ROOT" ]]; then echo "Refusing existing partial output: $ROOT" >&2; exit 1; fi
COMMON=(--root "$ROOT" --parent-root "$PARENT" --reference-root "$REFERENCE")
"$PYTHON" -m mixture_scaling.async_kd_weights plan "${COMMON[@]}"
mkdir -p "$ROOT/log"
pids=()
for item in 'kd_w010 0.1 0' 'kd_w030 0.3 1' 'kd_w050 0.5 2'; do
 read -r arm weight gpu <<< "$item"
 "$PYTHON" -u -m mixture_scaling.async_extension train --arm kd_extended --run-id "$arm" --kd-weight "$weight" --device "$gpu" --root "$ROOT" --parent-root "$PARENT" > "$ROOT/log/$arm.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == 1 ]]; then exit 1; fi
"$PYTHON" -m mixture_scaling.async_kd_weights prepare_eval "${COMMON[@]}" > "$ROOT/log/selection.log" 2>&1
pids=()
for gpu in 0 1 2 3; do
 "$PYTHON" -u -m mixture_scaling.async_kd_weights eval "${COMMON[@]}" --device "$gpu" > "$ROOT/log/eval_gpu$gpu.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == 1 ]]; then exit 1; fi
"$PYTHON" -m mixture_scaling.async_kd_weights aggregate "${COMMON[@]}"
