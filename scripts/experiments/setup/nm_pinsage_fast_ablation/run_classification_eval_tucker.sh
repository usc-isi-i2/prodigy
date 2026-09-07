#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
TRAIN_RUN_DIR="${TRAIN_RUN_DIR:?set TRAIN_RUN_DIR to the completed shared-training run}"
GPUS="${GPUS:-0,1,2,3}"
PY="${PY:-/home/mhchu/miniconda3/envs/prodigy/bin/python}"

"${PY}" "${SCRIPT_DIR}/make_model_lists.py" "${TRAIN_RUN_DIR}"
cd "${REPO_ROOT}"

common=(
  scripts/eval/eval_ckpts_all_graph_tasks_tucker.py
  --data-root /dataMeR1/phil/data
  --datasets covid_political,election2020,ukr_rus_suspended,twibot20
  --tasks classification --shots 10 --workers 2 --continue-on-error --gpus "${GPUS}"
)

"${PY}" "${common[@]}" --model-list "${SCRIPT_DIR}/model_list_graphsage.txt" -- \
  --gnn_type sage --n_hop 2 --neighbor_sampling_hop_sizes 9,9 \
  --neighbor_sampling_node_limit 101 --neighbor_matching_walk_hops 1 \
  --neighbor_sampling_method uniform

"${PY}" "${common[@]}" --model-list "${SCRIPT_DIR}/model_list_pinsage.txt" -- \
  --gnn_type pinsage --n_hop 2 --neighbor_sampling_hop_sizes 9,9 \
  --neighbor_sampling_node_limit 101 --neighbor_matching_walk_hops 1 \
  --neighbor_sampling_method pinsage --pinsage_num_walks 64 \
  --pinsage_walk_length 2 --pinsage_restart_prob 0 --pinsage_topk 100
