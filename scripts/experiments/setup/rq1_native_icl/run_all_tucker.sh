#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS_TEXT="${GPUS_TEXT:-2 3}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-2}"
read -r -a GPUS <<< "$GPUS_TEXT"
[[ "${GPUS[*]}" == "2 3" ]]
jobs=()
for target in covid_political election2020 ukr_rus_suspended twibot20; do
  for shot in 1 3 5 10; do
    for label_seed in 0 1 2 3 4; do
      for arm in pretrained no_pretrain; do jobs+=("$target $shot $label_seed $arm"); done
    done
  done
done
worker_count=$((2 * SLOTS_PER_GPU))
run_worker() {
  local worker="$1" gpu="$2" i target shot label_seed arm
  for i in "${!jobs[@]}"; do
    (( i % worker_count == worker )) || continue
    read -r target shot label_seed arm <<< "${jobs[$i]}"
    echo "START target=$target shot=$shot label_seed=$label_seed arm=$arm gpu=$gpu time=$(date -Is)"
    TARGET="$target" SHOT="$shot" LABEL_SEED="$label_seed" ARM="$arm" GPU="$gpu" \
      bash "$SCRIPT_DIR/run_cell_tucker.sh"
    echo "DONE target=$target shot=$shot label_seed=$label_seed arm=$arm gpu=$gpu time=$(date -Is)"
  done
}
pids=()
for gi in "${!GPUS[@]}"; do
  for ((slot=0; slot<SLOTS_PER_GPU; slot++)); do
    worker=$((gi*SLOTS_PER_GPU+slot)); run_worker "$worker" "${GPUS[$gi]}" & pids+=("$!")
  done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
(( status == 0 )) || exit "$status"
root="${STATE_ROOT:?set STATE_ROOT}/results"
count="$(find "$root" -name result.json -type f | wc -l | tr -d ' ')"
[[ "$count" == 160 ]] || { echo "expected 160 results, found $count" >&2; exit 20; }
echo "RQ1_NATIVE_ICL_COMPLETE results=160 time=$(date -Is)"
