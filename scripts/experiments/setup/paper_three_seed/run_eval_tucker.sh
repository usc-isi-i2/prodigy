#!/usr/bin/env bash
# Evaluate completed paper-replication checkpoints with their matching model and sampler.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PLAN_FILE="${REPO_ROOT}/log/paper_three_seed/plan.tsv"
MODEL_LIST_ROOT="${REPO_ROOT}/log/paper_three_seed/model_lists"
SEEDS_TEXT="${SEEDS:-1 2}"
RUN_STAMP="${RUN_STAMP:-20260807}"
GPUS_CSV="${GPUS:-0,1,2,3}"
DATASETS="ukr_rus_twitter,covid19_twitter,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk_twitter"

export PATH="/home/mhchu/miniconda3/bin:$PATH"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"
mkdir -p "$MODEL_LIST_ROOT"
cd "$REPO_ROOT"
[[ -s "$PLAN_FILE" ]] || "$PYTHON" "$SCRIPT_DIR/make_plan.py" > "$PLAN_FILE"

for group in sage_1hop gat_1hop sage_2hop; do
  : > "$MODEL_LIST_ROOT/${group}.txt"
done
while IFS=$'\t' read -r family arm config eval_group target_step; do
  [[ "$family" == family ]] && continue
  for seed in $SEEDS_TEXT; do
    prefix="paper3seed_${family}_${arm}_s${seed}"
    run_name="${prefix}_${RUN_STAMP}"
    checkpoint="${REPO_ROOT}/state/${run_name}/checkpoint/state_dict_${target_step}.ckpt"
    [[ -f "$checkpoint" ]] || { echo "missing $checkpoint" >&2; exit 1; }
    printf '%s %s\n' "$run_name" "$checkpoint" >> "$MODEL_LIST_ROOT/${eval_group}.txt"
  done
done < "$PLAN_FILE"

run_eval() {
  local group="$1"; shift
  "$PYTHON" scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
    --model-list "$MODEL_LIST_ROOT/${group}.txt" \
    --data-root /dataMeR1/phil/data \
    --datasets "$DATASETS" \
    --tasks nm --shots 3 --nm-n-way 30 --gpus "$GPUS_CSV" \
    "$@"
}

run_eval sage_1hop
run_eval gat_1hop --gnn-type gat
run_eval sage_2hop -- \
  --n_hop 2 --neighbor_sampling_hop_sizes 9,9 \
  --neighbor_sampling_node_limit 101 --neighbor_matching_walk_hops 1
date -u +%Y-%m-%dT%H:%M:%SZ > "${REPO_ROOT}/log/paper_three_seed/evaluation_complete_utc.txt"
