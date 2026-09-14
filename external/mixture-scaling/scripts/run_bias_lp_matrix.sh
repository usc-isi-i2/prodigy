#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${CONDA_PREFIX}/bin/python"
ROOT=${1:?output root required}
mkdir -p "$ROOT/log"
for phase in train eval; do
 pids=()
 for gpu in 0 1 2 3; do
  "$PYTHON" -u -m mixture_scaling.bias_lp_matrix "$phase" --root "$ROOT" --device "$gpu" > "$ROOT/log/${phase}_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
 done
 failed=0
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 if [[ "$failed" == 1 ]]; then exit 1; fi
done
"$PYTHON" -m mixture_scaling.bias_lp_matrix aggregate --root "$ROOT"
