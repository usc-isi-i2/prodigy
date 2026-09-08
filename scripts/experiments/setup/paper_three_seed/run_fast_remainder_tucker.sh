#!/usr/bin/env bash
# Finish the paper replication after the seed-1 one-hop batch, minimizing graph reloads.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PLAN_FILE="${REPO_ROOT}/log/paper_three_seed_remainder/plan.tsv"
RUN_ROOT="${REPO_ROOT}/log/paper_three_seed_remainder/${RUN_STAMP:-20260908}"
GPUS_TEXT="${GPUS:-0 1 2 3}"
MODELS_PER_GPU="${MODELS_PER_GPU:-8}"
WORKER_BUDGET="${WORKER_BUDGET:-128}"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
unset CUDA_VISIBLE_DEVICES || true
cd "$REPO_ROOT"
mkdir -p "$(dirname "$PLAN_FILE")" "$RUN_ROOT"
"${CONDA_PREFIX}/bin/python" "$SCRIPT_DIR/make_plan.py" > "$PLAN_FILE"

configs_for() {
  local pattern="$1"
  awk -F'\t' -v pattern="$pattern" 'NR>1 && $1 ~ pattern {print $3}' "$PLAN_FILE"
}

# All two-hop GraphSAGE conditions and both missing seeds share one 111 GB graph load.
mapfile -t twohop < <(configs_for '^(ladder_2hop|fixed_exposure_2hop)$')
twohop_dir="$RUN_ROOT/twohop_seeds_1-2"
if [[ ! -f "$twohop_dir/status.json" ]]; then
  [[ ! -e "$twohop_dir" ]] || { echo "REFUSE incomplete $twohop_dir" >&2; exit 1; }
  "${CONDA_PREFIX}/bin/python" experiments/run_shared_graph.py \
    --configs "${twohop[@]}" --seeds 1 2 --gpus $GPUS_TEXT \
    --models-per-gpu "$MODELS_PER_GPU" --worker-budget "$WORKER_BUDGET" \
    --threads-per-model 4 --run-dir "$twohop_dir"
fi

# Seed 1's one-hop mixtures were completed by the initial batch; run seed 2 once.
mapfile -t onehop < <(configs_for '^ladder_1hop$')
onehop_dir="$RUN_ROOT/onehop_seed_2"
if [[ ! -f "$onehop_dir/status.json" ]]; then
  [[ ! -e "$onehop_dir" ]] || { echo "REFUSE incomplete $onehop_dir" >&2; exit 1; }
  "${CONDA_PREFIX}/bin/python" experiments/run_shared_graph.py \
    --configs "${onehop[@]}" --seeds 2 --gpus $GPUS_TEXT \
    --models-per-gpu "$MODELS_PER_GPU" --worker-budget "$WORKER_BUDGET" \
    --threads-per-model 4 --run-dir "$onehop_dir" -- \
    --n_hop 1 --neighbor_sampling_hop_sizes '' \
    --neighbor_sampling_node_limit 2000 --neighbor_matching_walk_hops 0
fi

date -u +%Y-%m-%dT%H:%M:%SZ > "$RUN_ROOT/training_complete_utc.txt"
