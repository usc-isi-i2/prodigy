#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=4
ROOT=${1:?fresh LP root required}
mkdir -p "$ROOT/log"
for view in node node_neighbors; do
 for phase in train eval; do
  pids=()
  for gpu in 0 1 2 3; do
   python -u -m mixture_scaling.mini_lp "$phase" --view "$view" --root "$ROOT" --device "$gpu" --worker-index "$gpu" > "$ROOT/log/${view}_${phase}_gpu${gpu}.log" 2>&1 &
   pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if [[ "$failed" == 1 ]]; then exit 1; fi
 done
 python -m mixture_scaling.mini_lp aggregate --view "$view" --root "$ROOT"
done
