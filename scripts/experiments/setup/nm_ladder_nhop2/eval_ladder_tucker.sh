#!/usr/bin/env bash
# Evaluate 2-hop checkpoints with 2-hop episode extraction.
#
#   PHASE=A ./eval_ladder_tucker.sh
#   PHASE=robustness GPUS="0,1" ./eval_ladder_tucker.sh
#   PHASE=smoke ./eval_ladder_tucker.sh --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PHASE="${PHASE:-A}"
MODEL_LIST="${MODEL_LIST:-${SCRIPT_DIR}/model_list_${PHASE}.txt}"
SAMPLER_ARGS=(
  --n_hop 2
  --neighbor_sampling_hop_sizes 9,9
  --neighbor_sampling_node_limit 101
  --neighbor_matching_walk_hops 1
)

case "${PHASE}" in
  smoke)
    DATASETS="election2020"
    EXTRA_EXPERIMENT_ARGS=("${SAMPLER_ARGS[@]}" --val_len_cap 20 --test_len_cap 20)
    ;;
  A|robustness|all)
    DATASETS="ukr_rus_twitter,covid19_twitter,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk_twitter"
    EXTRA_EXPERIMENT_ARGS=("${SAMPLER_ARGS[@]}")
    ;;
  *)
    echo "unknown PHASE=${PHASE}; expected smoke, A, robustness, or all" >&2
    exit 2
    ;;
esac

[[ -f "${MODEL_LIST}" ]] || {
  echo "missing ${MODEL_LIST}; run make_model_list.py --phase ${PHASE} first" >&2
  exit 2
}

IFS=',' read -r -a GPU_ARR <<< "${GPUS:-0}"
for gpu in "${GPU_ARR[@]}"; do
  if [[ ! "${gpu}" =~ ^[0-3]$ ]]; then
    echo "refusing GPU ${gpu}: this project owns only Tucker GPUs 0-3" >&2
    exit 2
  fi
done

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"
python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${MODEL_LIST}" \
  --data-root "${DATA_ROOT:-/dataMeR1/phil/data}" \
  --datasets "${DATASETS}" \
  --tasks nm \
  --shots 3 \
  --nm-n-way 30 \
  --gpus "${GPUS:-0}" \
  "$@" \
  -- "${EXTRA_EXPERIMENT_ARGS[@]}"
