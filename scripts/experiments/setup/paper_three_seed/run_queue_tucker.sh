#!/usr/bin/env bash
# Train seeds 1 and 2 for all 57 registered PRODIGY paper conditions, then run
# the matching frozen NM evaluations. Intended for a dedicated Tucker worktree.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PLAN_FILE="${REPO_ROOT}/log/paper_three_seed/plan.tsv"
TRAIN_LOG_ROOT="${REPO_ROOT}/log/paper_three_seed/train"
RUN_STAMP="${RUN_STAMP:-20260807}"
SEEDS_TEXT="${SEEDS:-1 2}"
GPUS_TEXT="${GPUS:-0 1 2 3}"
RUN_EVAL="${RUN_EVAL:-1}"
read -r -a SEED_IDS <<< "$SEEDS_TEXT"
read -r -a GPU_IDS <<< "$GPUS_TEXT"

[[ ${#SEED_IDS[@]} -gt 0 ]] || { echo "SEEDS must not be empty" >&2; exit 2; }
[[ ${#GPU_IDS[@]} -gt 0 ]] || { echo "GPUS must not be empty" >&2; exit 2; }
for seed in "${SEED_IDS[@]}"; do
  [[ "$seed" =~ ^(1|2)$ ]] || { echo "registered missing seeds are 1 and 2, got $seed" >&2; exit 2; }
done
for gpu in "${GPU_IDS[@]}"; do
  [[ "$gpu" =~ ^[0-3]$ ]] || { echo "refusing non-owned Tucker GPU $gpu" >&2; exit 2; }
done

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

mkdir -p "$(dirname "$PLAN_FILE")" "$TRAIN_LOG_ROOT"
cd "$REPO_ROOT"
"$PYTHON" "$SCRIPT_DIR/make_plan.py" > "$PLAN_FILE"
[[ "$(($(wc -l < "$PLAN_FILE") - 1))" == 57 ]] || { echo "plan must contain 57 jobs" >&2; exit 2; }

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  for gpu in "${GPU_IDS[@]}"; do
    metrics="$(nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    IFS=',' read -r used utilization <<< "$metrics"
    if (( used > 1000 || utilization > 10 )); then
      echo "GPU $gpu is busy (memory=${used}MiB utilization=${utilization}%); refusing launch" >&2
      exit 1
    fi
  done
fi

declare -a JOB_LINES BUCKETS
while IFS=$'\t' read -r family arm config eval_group target_step; do
  [[ "$family" == family ]] && continue
  for seed in "${SEED_IDS[@]}"; do
    JOB_LINES+=("$seed"$'\t'"$family"$'\t'"$arm"$'\t'"$config"$'\t'"$eval_group"$'\t'"$target_step")
  done
done < "$PLAN_FILE"

for index in "${!JOB_LINES[@]}"; do
  bucket=$((index % ${#GPU_IDS[@]}))
  BUCKETS[$bucket]+="${JOB_LINES[$index]}"$'\n'
done

run_one() {
  local gpu="$1" seed="$2" family="$3" arm="$4" config="$5" target_step="$6"
  local prefix="paper3seed_${family}_${arm}_s${seed}"
  local run_name="${prefix}_${RUN_STAMP}"
  local checkpoint="${REPO_ROOT}/state/${run_name}/checkpoint/state_dict_${target_step}.ckpt"
  local run_dir="${REPO_ROOT}/state/${run_name}"
  local log_path="${TRAIN_LOG_ROOT}/${run_name}_gpu${gpu}.log"
  local args=(
    "$PYTHON" experiments/run_single_experiment.py
    --config "$config" --device "$gpu" --seed "$seed"
    --prefix "$prefix" --timestamp "$RUN_STAMP"
  )
  case "$family" in
    specialist)
      args+=(--epochs 4)
      ;;
    ladder_1hop)
      args+=(
        --n_hop 1 --neighbor_sampling_hop_sizes ""
        --neighbor_sampling_node_limit 2000 --neighbor_matching_walk_hops 0
        --workers 16
      )
      ;;
    ladder_2hop|ladder_gatv2|fixed_exposure_2hop)
      ;;
    *)
      echo "unknown family $family" >&2
      return 2
      ;;
  esac
  if [[ -f "$checkpoint" ]]; then
    echo "[gpu $gpu] SKIP complete $run_name" >&2
    return 0
  fi
  if [[ -e "$run_dir" ]]; then
    echo "[gpu $gpu] REFUSE incomplete existing run $run_dir" >&2
    return 1
  fi
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf 'DRY GPU=%q' "$gpu"
    printf ' %q' "${args[@]}"
    printf '\n'
    return 0
  fi
  echo "[gpu $gpu] START seed=$seed family=$family arm=$arm utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
  "${args[@]}" > "$log_path" 2>&1
  [[ -f "$checkpoint" ]] || { echo "missing comparison checkpoint after $run_name" >&2; return 1; }
  echo "[gpu $gpu] DONE $run_name utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
}

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  for bucket in "${!GPU_IDS[@]}"; do
    gpu="${GPU_IDS[$bucket]}"
    while IFS=$'\t' read -r seed family arm config eval_group target_step; do
      [[ -n "$seed" ]] || continue
      run_one "$gpu" "$seed" "$family" "$arm" "$config" "$target_step"
    done <<< "${BUCKETS[$bucket]:-}"
  done
  exit 0
fi

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "seeds=$SEEDS_TEXT"
  echo "gpus=$GPUS_TEXT"
  echo "run_stamp=$RUN_STAMP"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${REPO_ROOT}/log/paper_three_seed/provenance.txt"

declare -a PIDS
for bucket in "${!GPU_IDS[@]}"; do
  gpu="${GPU_IDS[$bucket]}"
  lines="${BUCKETS[$bucket]:-}"
  [[ -n "${lines//$'\n'/}" ]] || continue
  (
    while IFS=$'\t' read -r seed family arm config eval_group target_step; do
      [[ -n "$seed" ]] || continue
      run_one "$gpu" "$seed" "$family" "$arm" "$config" "$target_step"
    done <<< "$lines"
  ) &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" != 0 ]]; then
  echo "at least one training worker failed; evaluation not started" >&2
  exit "$status"
fi
date -u +%Y-%m-%dT%H:%M:%SZ > "${REPO_ROOT}/log/paper_three_seed/training_complete_utc.txt"

if [[ "$RUN_EVAL" == 1 ]]; then
  SEEDS="$SEEDS_TEXT" GPUS="$(IFS=,; echo "${GPU_IDS[*]}")" RUN_STAMP="$RUN_STAMP" \
    bash "$SCRIPT_DIR/run_eval_tucker.sh"
fi
