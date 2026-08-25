#!/usr/bin/env bash
# Evaluate each model only on the labeled graph excluded from its donor mixture.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
GPUS="${GPUS:-2}"
IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
for gpu in "${GPU_ARR[@]}"; do
  [[ "${gpu}" =~ ^[23]$ ]] || {
    echo "refusing GPU ${gpu}: this project currently owns only Tucker GPUs 2 and 3" >&2
    exit 2
  }
done

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

model_args=()
[[ "${ALLOW_PARTIAL:-0}" != "1" ]] || model_args+=(--allow-partial)
python3 "${SCRIPT_DIR}/make_model_lists.py" "${model_args[@]}"

for target in covid_political election2020 ukr_rus_suspended twibot20; do
  extra=()
  [[ "${DRY_RUN:-0}" != "1" ]] || extra+=(--dry-run)
  python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
    --model-list "${SCRIPT_DIR}/model_lists/${target}.txt" \
    --data-root /dataMeR1/phil/data \
    --datasets "${target}" \
    --tasks pl \
    --shots 10 \
    --gpus "${GPUS}" \
    --continue-on-error \
    "${extra[@]}" \
    -- --n_hop 2 --neighbor_sampling_hop_sizes 9,9 \
       --neighbor_sampling_node_limit 101 --neighbor_matching_walk_hops 1
done

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  python3 scripts/harness/benchmark_tasks/parse_benchmark_eval_logs.py \
    --log-root "${REPO_ROOT}/log" \
    --out-dir scripts/experiments/analysis/evaluation/task_tables
fi
