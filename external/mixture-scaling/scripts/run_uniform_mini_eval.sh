#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
EVAL_PYTHON="${CONDA_PREFIX}/bin/python"
"$EVAL_PYTHON" -c 'import sys,torch,torch_geometric; print(sys.executable,torch.__version__,torch_geometric.__version__)'
ROOT=${1:?fresh output root required}
mkdir -p "$ROOT/log"
pids=()
for gpu in 0 1 2 3; do
 "$EVAL_PYTHON" -u -m mixture_scaling.uniform_mini_eval eval --output-root "$ROOT" --device "$gpu" --worker-index "$gpu" > "$ROOT/log/gpu${gpu}.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == 1 ]]; then exit 1; fi
"$EVAL_PYTHON" -m mixture_scaling.uniform_mini_eval aggregate --output-root "$ROOT"
