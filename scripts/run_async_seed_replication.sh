#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/mhchu/miniconda3/bin:$PATH"
source /home/mhchu/miniconda3/etc/profile.d/conda.sh
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PWD/src"
PYTHON="${CONDA_PREFIX}/bin/python"
ROOT=${1:?new replication output root required}
if [[ -f "$ROOT/results/COMPLETE.json" ]]; then echo "Completed run exists: $ROOT"; exit 0; fi
if [[ -e "$ROOT" ]]; then echo "Refusing existing partial output: $ROOT" >&2; exit 1; fi
"$PYTHON" -m mixture_scaling.async_seed_replication plan --root "$ROOT"
mkdir -p "$ROOT/log"
pids=()
for item in '1 0 2' '2 1 3'; do
 read -r seed parent_gpu singleton_gpu <<< "$item"
 "$PYTHON" -u -m mixture_scaling.async_convergence train --root "$ROOT/seed$seed/parent" --arm async_kd --seed "$seed" --data-seed 0 --probe-seed 0 --device "$parent_gpu" > "$ROOT/log/parent_seed$seed.log" 2>&1 &
 pids+=("$!")
 "$PYTHON" -u -m mixture_scaling.async_seed_singletons --root "$ROOT/seed$seed" --seed "$seed" --device "$singleton_gpu" > "$ROOT/log/singletons_seed$seed.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == 1 ]]; then exit 1; fi
pids=()
for item in '1 kd_w010 0' '1 kd_w100 1' '2 kd_w010 2' '2 kd_w100 3'; do
 read -r seed arm gpu <<< "$item"
 "$PYTHON" -u -m mixture_scaling.async_seed_replication continue --root "$ROOT" --training-seed "$seed" --run-id "$arm" --device "$gpu" > "$ROOT/log/${arm}_seed$seed.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == 1 ]]; then exit 1; fi
"$PYTHON" -m mixture_scaling.async_seed_replication prepare_eval --root "$ROOT" > "$ROOT/log/selection.log" 2>&1
pids=()
for gpu in 0 1 2 3; do
 (
  for seed in 1 2; do
   "$PYTHON" -u -m mixture_scaling.async_seed_replication eval --root "$ROOT" --training-seed "$seed" --device "$gpu"
  done
 ) > "$ROOT/log/eval_gpu$gpu.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [[ "$failed" == 1 ]]; then exit 1; fi
"$PYTHON" -m mixture_scaling.async_seed_replication aggregate --root "$ROOT"
