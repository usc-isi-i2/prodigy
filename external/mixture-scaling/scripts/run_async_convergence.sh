#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/src"
PYTHON="${CONDA_PREFIX}/bin/python"
ROOT=${1:?output root required}
mkdir -p "$ROOT/log"
"$PYTHON" -u -m mixture_scaling.async_singleton_reference --device 3 --root "$ROOT" > "$ROOT/log/singleton_references.log" 2>&1 &
reference_pid=$!
"$PYTHON" -u -m mixture_scaling.async_convergence train --arm extended_bce --device 0 --root "$ROOT" > "$ROOT/log/extended_bce.log" 2>&1 &
extended_pid=$!
"$PYTHON" -u -m mixture_scaling.async_convergence train --arm async_kd --device 1 --root "$ROOT" > "$ROOT/log/async_kd.log" 2>&1 &
kd_pid=$!
# The sibling uses the first exact rewind state. Launch as soon as that state is ready.
while [[ ! -f "$ROOT/first_rewind.pt" ]]; do
 if ! kill -0 "$kd_pid" 2>/dev/null; then wait "$kd_pid"; echo "KD exited without a rewind checkpoint" >&2; exit 1; fi
 sleep 2
done
"$PYTHON" -u -m mixture_scaling.async_convergence train --arm rewind_bce --device 2 --root "$ROOT" > "$ROOT/log/rewind_bce.log" 2>&1 &
rewind_pid=$!
failed=0
for pid in "$extended_pid" "$kd_pid" "$rewind_pid" "$reference_pid"; do wait "$pid" || failed=1; done
if [[ "$failed" == 1 ]]; then exit 1; fi
"$PYTHON" -m mixture_scaling.async_convergence probes --device 0 --root "$ROOT" > "$ROOT/log/singleton_probes.log" 2>&1
"$PYTHON" -m mixture_scaling.async_convergence prepare_eval --root "$ROOT"
pids=()
for gpu in 0 1 2 3; do
 "$PYTHON" -u -m mixture_scaling.async_convergence eval --device "$gpu" --root "$ROOT" > "$ROOT/log/eval_gpu$gpu.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == 1 ]]; then exit 1; fi
"$PYTHON" -m mixture_scaling.async_convergence aggregate --root "$ROOT"
