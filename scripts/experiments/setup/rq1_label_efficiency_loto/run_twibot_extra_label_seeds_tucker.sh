#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
BASE_STATE="${BASE_STATE:-/dataMeR1/phil/gfm/prodigy-rq1/state/rq1_label_efficiency_loto}"
BASE_LOG="${BASE_LOG:-/dataMeR1/phil/gfm/prodigy-rq1/log/rq1_label_efficiency_loto}"
CONTROLLER_PID="${CONTROLLER_PID:?PID of the intentionally stopped protocol controller}"
CHECKPOINT="${BASE_STATE}/pretrain/rq1_loto_twibot20_pretrain_s1_20260828/state_dict"
CACHE="${BASE_STATE}/subgraph_cache_v2/twibot20_seed1.pt"
OUTPUT_ROOT="${BASE_STATE}/twibot_extra_label_seeds_seed1"
LOG_ROOT="${BASE_LOG}/twibot_extra_label_seeds_seed1"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"
cd "$REPO_ROOT"

while true; do
  completed="$(find "${BASE_STATE}/adapt_cached_v2/seed_1" -name result.json -type f | wc -l)"
  active="$(pgrep -fc 'rq1_label_efficiency_loto.adapt .*--seed 1 .*adapt_cached_v2/seed_1' || true)"
  echo "canonical seed1 completed=${completed}/32 active=${active}"
  [[ "$completed" == 32 && "$active" == 0 ]] && break
  sleep 300
done

jobs=()
for label_seed in 3 4; do
  for budget in 1 10 100 1000; do
    jobs+=("${label_seed}:scratch:${budget}:")
    jobs+=("${label_seed}:pretrained:${budget}:${CHECKPOINT}")
  done
done

worker() {
  local worker_index="$1" gpu="$2" index=0 item label_seed arm budget checkpoint out log
  for item in "${jobs[@]}"; do
    if (( index % 8 == worker_index )); then
      IFS=: read -r label_seed arm budget checkpoint <<< "$item"
      out="${OUTPUT_ROOT}/label_seed_${label_seed}/${budget}/${arm}"
      log="${LOG_ROOT}/label_seed_${label_seed}_${budget}_${arm}.log"
      cmd=("${CONDA_PREFIX}/bin/python" -u -m scripts.experiments.setup.rq1_label_efficiency_loto.adapt
        --target twibot20 --arm "$arm" --budget "$budget" --seed 1 --label-seed "$label_seed"
        --output "$out" --device cuda:0 --patience 4 --subgraph-cache "$CACHE"
        --protocol-version cached-neighborhoods-patience4-v2-extra-label-seed)
      [[ "$arm" == pretrained ]] && cmd+=(--pretrained-checkpoint "$checkpoint")
      echo "[gpu $gpu] label_seed=$label_seed budget=$budget arm=$arm"
      [[ -f "$out/result.json" ]] || CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log" 2>&1
    fi
    ((index+=1))
  done
}

pids=()
for worker_index in 0 1 2 3; do worker "$worker_index" 2 & pids+=("$!"); done
for worker_index in 4 5 6 7; do worker "$worker_index" 3 & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
[[ "$status" == 0 ]] || exit "$status"

completed="$(find "$OUTPUT_ROOT" -name result.json -type f | wc -l)"
[[ "$completed" == 16 ]] || { echo "expected 16 extra-label-seed results, found $completed" >&2; exit 4; }
kill -CONT "$CONTROLLER_PID"
echo "extra label-seed sweep complete; resumed controller pid=$CONTROLLER_PID"
