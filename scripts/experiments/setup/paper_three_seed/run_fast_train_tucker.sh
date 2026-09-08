#!/usr/bin/env bash
# Train the two missing paper seeds with shared-graph concurrency where possible.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SEEDS_TEXT="${SEEDS:-1 2}"
GPUS_TEXT="${GPUS:-0 2 3}"
MODELS_PER_GPU="${MODELS_PER_GPU:-8}"
WORKER_BUDGET="${WORKER_BUDGET:-72}"
RUN_STAMP="${RUN_STAMP:-20260908}"
RUN_ROOT="${REPO_ROOT}/log/paper_three_seed_fast/${RUN_STAMP}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
unset CUDA_VISIBLE_DEVICES || true
cd "$REPO_ROOT"
mkdir -p "$RUN_ROOT"

mapfile -t PLAN < <("${CONDA_PREFIX}/bin/python" "$SCRIPT_DIR/make_plan.py" | tail -n +2)

configs_for_family() {
  local wanted="$1" line family arm config eval_group target_step
  for line in "${PLAN[@]}"; do
    IFS=$'\t' read -r family arm config eval_group target_step <<< "$line"
    [[ "$family" == "$wanted" ]] && printf '%s\n' "$config"
  done
}

run_shared_family() {
  local seed="$1" family="$2"; shift 2
  local run_dir="$RUN_ROOT/seed${seed}_${family}"
  mapfile -t configs < <(configs_for_family "$family")
  [[ ${#configs[@]} -gt 0 ]] || return 0
  if [[ -f "$run_dir/status.json" ]] && grep -q '"status": "complete"' "$run_dir/status.json"; then
    echo "SKIP complete seed=$seed family=$family"
    return 0
  fi
  [[ ! -e "$run_dir" ]] || { echo "REFUSE existing incomplete $run_dir" >&2; return 1; }
  local mode_args=()
  [[ "${DRY_RUN:-0}" == 1 ]] && mode_args+=(--dry-run)
  "${CONDA_PREFIX}/bin/python" experiments/run_shared_graph.py \
    --configs "${configs[@]}" --gpus $GPUS_TEXT \
    --models-per-gpu "$MODELS_PER_GPU" --worker-budget "$WORKER_BUDGET" \
    --threads-per-model 4 --run-dir "$run_dir" "${mode_args[@]}" -- --seed "$seed" "$@"
}

for seed in $SEEDS_TEXT; do
  run_shared_family "$seed" ladder_1hop \
    --n_hop 1 --neighbor_sampling_hop_sizes '' \
    --neighbor_sampling_node_limit 2000 --neighbor_matching_walk_hops 0
  run_shared_family "$seed" ladder_2hop
  run_shared_family "$seed" ladder_gatv2
  run_shared_family "$seed" fixed_exposure_2hop
done

[[ "${DRY_RUN:-0}" == 1 ]] || date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/training_complete_utc.txt"
