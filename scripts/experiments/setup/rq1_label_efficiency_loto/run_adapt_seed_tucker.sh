#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PRETRAIN_STATE_ROOT="${PRETRAIN_STATE_ROOT:-${REPO_ROOT}/state/rq1_label_efficiency_loto/pretrain}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/state/rq1_label_efficiency_loto/adapt_cached_v2}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/log/rq1_label_efficiency_loto/adapt_cached_v2}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/state/rq1_label_efficiency_loto/subgraph_cache_v2}"
SEED="${SEED:?set SEED to 0, 1, or 2}"
RUN_STAMP="${RUN_STAMP:-20260828}"
GPUS_TEXT="${GPUS_TEXT:-2 3}"
SLOTS_PER_GPU="${SLOTS_PER_GPU:-4}"
DRY_RUN="${DRY_RUN:-0}"
ADAPT_PROTOCOL="${ADAPT_PROTOCOL:-canonical}"
TARGETS_TEXT="${TARGETS_TEXT:-}"
LABEL_SEED="${LABEL_SEED:-$SEED}"
GRID_LAYOUT="${GRID_LAYOUT:-0}"
USE_SHARED_CACHE="${USE_SHARED_CACHE:-0}"
PROTOCOL_VERSION="${PROTOCOL_VERSION:-revised-eval100-then200-patience3-delta001-v1}"
read -r -a GPUS <<< "$GPUS_TEXT"
[[ "$SEED" =~ ^[012]$ ]] || { echo "SEED must be 0, 1, or 2" >&2; exit 2; }
for gpu in "${GPUS[@]}"; do
  [[ "$gpu" =~ ^[23]$ ]] || { echo "only Tucker GPUs 2 and 3 are authorized" >&2; exit 2; }
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${CONDA_PREFIX}/bin/python"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$CACHE_ROOT"
cd "$REPO_ROOT"

mapfile -t target_rows < <("$PYTHON" "$SCRIPT_DIR/plan.py" | tail -n +2)
if [[ -n "$TARGETS_TEXT" ]]; then
  read -r -a requested_targets <<< "$TARGETS_TEXT"
  filtered_rows=()
  for row in "${target_rows[@]}"; do
    IFS=$'\t' read -r target _excluded _sources <<< "$row"
    for requested in "${requested_targets[@]}"; do
      if [[ "$target" == "$requested" ]]; then
        filtered_rows+=("$row")
        break
      fi
    done
  done
  [[ "${#filtered_rows[@]}" -eq "${#requested_targets[@]}" ]] || {
    echo "TARGETS_TEXT contains an unknown or duplicate target: $TARGETS_TEXT" >&2
    exit 2
  }
  target_rows=("${filtered_rows[@]}")
fi
if [[ "$SEED" != 0 || "$USE_SHARED_CACHE" == 1 ]]; then
  cache_pids=()
  for row in "${target_rows[@]}"; do
    IFS=$'\t' read -r target _excluded _sources <<< "$row"
    if [[ -f "$CACHE_ROOT/${target}_seed${SEED}.pt" ]]; then
      echo "SKIP existing cache $CACHE_ROOT/${target}_seed${SEED}.pt"
      continue
    fi
    "$PYTHON" -u -m scripts.experiments.setup.rq1_label_efficiency_loto.precompute_adapt_cache \
      --target "$target" --seed "$SEED" --output "$CACHE_ROOT/${target}_seed${SEED}.pt" \
      > "$LOG_ROOT/cache_${target}_seed${SEED}.log" 2>&1 & cache_pids+=("$!")
  done
  for pid in "${cache_pids[@]}"; do wait "$pid"; done
fi
jobs=()
for row in "${target_rows[@]}"; do
  IFS=$'\t' read -r target _excluded _sources <<< "$row"
  checkpoint="$PRETRAIN_STATE_ROOT/rq1_loto_${target}_pretrain_s${SEED}_${RUN_STAMP}/state_dict"
  [[ -f "$checkpoint" ]] || { echo "missing pretraining checkpoint $checkpoint" >&2; exit 3; }
  for budget in 1 10 100 1000; do
    jobs+=("$target:scratch:$budget:")
    jobs+=("$target:pretrained:$budget:$checkpoint")
  done
done

worker_count=$(( ${#GPUS[@]} * SLOTS_PER_GPU ))

worker() {
  local worker_index="$1" gpu="$2" item target arm budget checkpoint out log cmd index=0
  for item in "${jobs[@]}"; do
    if (( index % worker_count == worker_index )); then
      IFS=: read -r target arm budget checkpoint <<< "$item"
      if [[ "$GRID_LAYOUT" == 1 ]]; then
        out="$OUTPUT_ROOT/model_seed_${SEED}/label_seed_${LABEL_SEED}/${target}/${budget}/${arm}"
        log="$LOG_ROOT/model_seed_${SEED}_label_seed_${LABEL_SEED}_${target}_${budget}_${arm}.log"
      else
        out="$OUTPUT_ROOT/seed_${SEED}/${target}/${budget}/${arm}"
        log="$LOG_ROOT/seed_${SEED}_${target}_${budget}_${arm}.log"
      fi
      if [[ -f "$out/result.json" ]]; then
        echo "[gpu $gpu] SKIP $target $budget $arm"
        ((index+=1)); continue
      fi
      cmd=("$PYTHON" -u -m scripts.experiments.setup.rq1_label_efficiency_loto.adapt
        --target "$target" --arm "$arm" --budget "$budget" --seed "$SEED"
        --label-seed "$LABEL_SEED"
        --output "$out" --device cuda:0 --patience 4)
      if [[ "$ADAPT_PROTOCOL" == revised ]]; then
        cmd+=(--first-eval-update 100 --eval-every 200 --patience 3
          --min-updates 500 --min-delta 0.001 --separate-selection-and-stopping
          --protocol-version "$PROTOCOL_VERSION")
      elif [[ "$ADAPT_PROTOCOL" != canonical ]]; then
        echo "unknown ADAPT_PROTOCOL=$ADAPT_PROTOCOL" >&2; return 2
      fi
      [[ "$SEED" != 0 || "$USE_SHARED_CACHE" == 1 ]] && \
        cmd+=(--subgraph-cache "$CACHE_ROOT/${target}_seed${SEED}.pt")
      [[ "$arm" == pretrained ]] && cmd+=(--pretrained-checkpoint "$checkpoint")
      printf '[gpu %s] cmd=' "$gpu"; printf '%q ' "${cmd[@]}"; printf '\n'
      [[ "$DRY_RUN" == 1 ]] || CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log" 2>&1
    fi
    ((index+=1))
  done
}

pids=()
for gpu_index in "${!GPUS[@]}"; do
  for ((slot=0; slot<SLOTS_PER_GPU; slot++)); do
    worker_index=$((gpu_index * SLOTS_PER_GPU + slot))
    worker "$worker_index" "${GPUS[$gpu_index]}" & pids+=("$!")
  done
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
