#!/usr/bin/env bash
# Evaluate 9 specialists + 36 pairs + 9 leave-one-out models on five CLS targets.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/log/nm_cls_lattice_20260905}"
MODEL_LIST="${OUTPUT_ROOT}/model_list.tsv"
SPECIALIST_ROOT="${SPECIALIST_ROOT:-/dataMeR1/phil/gfm/worktree-runtime-archive-20260812/prodigy-final-core/files/state/final_core}"
PAIR_ROOT="${PAIR_ROOT:-/dataMeR1/phil/gfm/prodigy-nm-pairs/log/nm_pairwise_finalcore/shared_seed0_20260904/state}"
LOO_ROOT="${LOO_ROOT:-/dataMeR1/phil/gfm/prodigy-nm-loo/log/nm_leave_one_out_finalcore/shared_seed0_20260905_retry1/state}"
GPUS_TEXT="${GPUS:-0 1 2 3}"
read -r -a GPU_IDS <<< "${GPUS_TEXT}"

for gpu in "${GPU_IDS[@]}"; do
  [[ "${gpu}" =~ ^[0-3]$ ]] || { echo "refusing GPU ${gpu}; only 0-3 are ours" >&2; exit 2; }
done
[[ "${#GPU_IDS[@]}" == 4 ]] || { echo "expected all four owned GPUs" >&2; exit 2; }

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-${CONDA_PREFIX}/bin/python}"

mkdir -p "${OUTPUT_ROOT}/results" "${OUTPUT_ROOT}/runs" "${OUTPUT_ROOT}/queue" "${OUTPUT_ROOT}/eval_state"
cd "${REPO_ROOT}"
"${PYTHON}" -m scripts.experiments.setup.nm_pairwise_finalcore.make_cls_lattice_model_list \
  --output "${MODEL_LIST}" --specialist-root "$SPECIALIST_ROOT" \
  --pair-root "$PAIR_ROOT" --loo-root "$LOO_ROOT"
[[ "$(($(wc -l < "${MODEL_LIST}") - 1))" == 54 ]] || { echo "model list must have 54 rows" >&2; exit 2; }

for index in 0 1 2 3; do
  shard="${OUTPUT_ROOT}/model_list_gpu${index}.tsv"
  { head -n 1 "${MODEL_LIST}"; tail -n +2 "${MODEL_LIST}" | awk -v i="${index}" '((NR-1)%4)==i'; } > "${shard}"
  [[ "$(($(wc -l < "${shard}") - 1))" -ge 13 ]] || { echo "short shard ${shard}" >&2; exit 2; }
done

{
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git rev-parse --abbrev-ref HEAD)"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "gpus=${GPUS_TEXT}"
  echo "wandb_mode=${WANDB_MODE}"
  echo "protocol=128_fixed_2way_10shot_cls_seed0_v1"
  echo "checkpoint_step=2500"
} > "${OUTPUT_ROOT}/provenance.txt"

pids=()
for index in 0 1 2 3; do
  gpu="${GPU_IDS[$index]}"
  result="${OUTPUT_ROOT}/results/gpu${index}.jsonl"
  [[ ! -e "${result}" ]] || { echo "refusing to overwrite ${result}" >&2; exit 2; }
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u \
    -m scripts.experiments.setup.icl_arch_matrix.evaluate_prodigy \
    --config scripts/experiments/setup/final_core/training.yaml \
    --state-root /unused \
    --eval-state-root "${OUTPUT_ROOT}/eval_state/gpu${index}" \
    --log-root "${OUTPUT_ROOT}/runs/gpu${index}" \
    --results "${result}" \
    --model-list "${OUTPUT_ROOT}/model_list_gpu${index}.tsv" \
    --include-facebook \
    --checkpoint-step 2500 \
    --training-seed 0 \
    --eval-episode-seed-offset 0 \
    --device 0 > "${OUTPUT_ROOT}/queue/gpu${index}.log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do wait "${pid}" || status=1; done
(( status == 0 )) || { echo "one or more CLS shards failed" >&2; exit 1; }

"${PYTHON}" -m scripts.experiments.setup.nm_pairwise_finalcore.aggregate_cls_lattice \
  --input-root "${OUTPUT_ROOT}/results" \
  --output "${OUTPUT_ROOT}/classification_long.tsv"
date -u +%FT%TZ > "${OUTPUT_ROOT}/complete_utc.txt"
