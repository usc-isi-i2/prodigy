#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

[[ "$(git -C "${REPO_ROOT}" branch --show-current)" == "codex/nm-ladder-unconfined" ]] || {
  echo "wrong branch/worktree" >&2
  exit 2
}

python3 "${SCRIPT_DIR}/make_configs.py"

train_bucket() {
  local device="$1"; shift
  local rung config
  for rung in "$@"; do
    config="${SCRIPT_DIR}/configs/train_r${rung}.yaml"
    echo "[$(date -Is)] training rung ${rung} on GPU ${device}"
    bash "${SCRIPT_DIR}/train_one_tucker.sh" "${config}" "${device}" \
      > "${LOG_DIR}/train_r${rung}.log" 2>&1
    echo "[$(date -Is)] completed rung ${rung}"
  done
}

train_bucket 2 1 3 5 7 &
worker2=$!
train_bucket 3 2 4 6 8 &
worker3=$!
wait "${worker2}"
wait "${worker3}"

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

python3 "${SCRIPT_DIR}/make_model_list.py" \
  --state-dir "${REPO_ROOT}/state" --output "${SCRIPT_DIR}/model_list.txt"

python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
  --model-list "${SCRIPT_DIR}/model_list.txt" \
  --data-root /dataMeR1/phil/data \
  --datasets ukr_rus_twitter,covid19_twitter,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk_twitter \
  --tasks nm --shots 3 --nm-n-way 30 --workers 2 --continue-on-error --gpus 2,3 \
  -- --n_hop 2 --neighbor_sampling_hop_sizes 9,9 \
  --neighbor_sampling_node_limit 101 --neighbor_matching_walk_hops 1

python3 scripts/experiments/analysis/transfer/ablations/prodigy_nm/source_schedule/nm_ladder_sequential_nhop2/assemble_unconfined.py \
  --log-root "${REPO_ROOT}/log"
python3 scripts/experiments/analysis/transfer/ablations/prodigy_nm/source_schedule/nm_ladder_sequential_nhop2/plot_schedule_means.py
echo "[$(date -Is)] pipeline complete"
