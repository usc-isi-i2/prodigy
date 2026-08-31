#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS_TEXT="${GPUS_TEXT:-2 3}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
read -r -a GPUS <<< "$GPUS_TEXT"
[[ "${#GPUS[@]}" == 2 && "${GPUS[0]}" == 2 && "${GPUS[1]}" == 3 ]]

jobs=()
for target in covid_political election2020 ukr_rus_suspended twibot20; do
  for budget in 10 100 1000; do
    for arm in pretrained scratch; do
      jobs+=("$target $budget $arm")
    done
  done
done

worker_count=$((2 * SLOTS_PER_GPU))
run_worker() {
  local worker="$1" gpu="$2" i target budget arm
  for i in "${!jobs[@]}"; do
    (( i % worker_count == worker )) || continue
    read -r target budget arm <<< "${jobs[$i]}"
    echo "START target=$target budget=$budget arm=$arm gpu=$gpu time=$(date -Is)"
    TARGET="$target" BUDGET="$budget" ARM="$arm" GPU="$gpu" \
      bash "$SCRIPT_DIR/run_cell_tucker.sh"
    echo "DONE target=$target budget=$budget arm=$arm gpu=$gpu time=$(date -Is)"
  done
}

pids=()
for gpu_index in "${!GPUS[@]}"; do
  for ((slot=0; slot<SLOTS_PER_GPU; slot++)); do
    worker=$((gpu_index * SLOTS_PER_GPU + slot))
    run_worker "$worker" "${GPUS[$gpu_index]}" & pids+=("$!")
  done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"

root="${STATE_ROOT:-$(cd "$SCRIPT_DIR/../../../.." && pwd)/state/rq1_native_cls_pilot}/results"
count="$(find "$root" -name result.json -type f | wc -l | tr -d ' ')"
[[ "$count" == 24 ]] || { echo "expected 24 results, found $count" >&2; exit 20; }
echo "RQ1_NATIVE_CLS_PILOT_COMPLETE results=24 time=$(date -Is)"
