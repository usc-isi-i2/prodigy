#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/src"
PYTHON="${CONDA_PREFIX}/bin/python"
ROOT=${1:?new merge output root required}
if [[ -f "$ROOT/results/COMPLETE.json" ]]; then echo "Completed run exists: $ROOT"; exit 0; fi
if [[ -e "$ROOT" ]]; then echo "Refusing existing partial output: $ROOT" >&2; exit 1; fi
"$PYTHON" -m mixture_scaling.async_weight_merge plan --root "$ROOT"
mkdir -p "$ROOT/log"
"$PYTHON" -u -m mixture_scaling.async_weight_merge validate --root "$ROOT" --device 0 > "$ROOT/log/validation.log" 2>&1
if [[ -f "$ROOT/node_neighbors/lp/merge_selected/best.pt" ]]; then
 pids=()
 for gpu in 0 1 2 3; do
  "$PYTHON" -u -m mixture_scaling.async_weight_merge eval --root "$ROOT" --device "$gpu" > "$ROOT/log/eval_gpu$gpu.log" 2>&1 &
  pids+=("$!")
 done
 failed=0
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 if [[ "$failed" == 1 ]]; then exit 1; fi
fi
"$PYTHON" -m mixture_scaling.async_weight_merge aggregate --root "$ROOT"
