#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/src"
PYTHON="${CONDA_PREFIX}/bin/python"
ROOT=${1:?new control output root required}
REFERENCE=${2:-/dataMeR1/phil/gfm/mixture-scaling/state/async_seed_replication_s12/seed2}
if [[ -f "$ROOT/results/COMPLETE.json" ]]; then echo "Completed run exists: $ROOT"; exit 0; fi
if [[ -e "$ROOT" ]]; then echo "Refusing existing partial output: $ROOT" >&2; exit 1; fi
"$PYTHON" -m mixture_scaling.async_seed2_control plan --root "$ROOT" --reference-root "$REFERENCE"
mkdir -p "$ROOT/log"
"$PYTHON" -u -m mixture_scaling.async_seed2_control train --root "$ROOT" --reference-root "$REFERENCE" --device 0 > "$ROOT/log/ukraine_only.log" 2>&1
"$PYTHON" -m mixture_scaling.async_seed2_control prepare_eval --root "$ROOT" --reference-root "$REFERENCE" > "$ROOT/log/selection.log" 2>&1
pids=()
for gpu in 0 1 2 3; do
 "$PYTHON" -u -m mixture_scaling.async_seed2_control eval --root "$ROOT" --reference-root "$REFERENCE" --device "$gpu" > "$ROOT/log/eval_gpu$gpu.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == 1 ]]; then exit 1; fi
"$PYTHON" -m mixture_scaling.async_seed2_control aggregate --root "$ROOT" --reference-root "$REFERENCE"
