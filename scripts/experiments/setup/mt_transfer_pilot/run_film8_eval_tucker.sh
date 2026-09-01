#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
cd "${ROOT}"

TARGETS=(covid_political election2020 facebook_page_reference twibot20 ukr_rus_suspended)
GPUS=(2 3 2 3 2)
pids=()
for i in "${!TARGETS[@]}"; do
  heldout="${TARGETS[$i]}"; gpu="${GPUS[$i]}"
  run="$(ls -dt state/mtpilot_NM_MT_task8_film_heldout_${heldout}_[0-9]*/ | head -n1)"
  ckpt="${run}checkpoint/state_dict_900.ckpt"
  [[ -f "${ckpt}" ]] || { echo "missing ${ckpt}" >&2; exit 1; }
  list="/tmp/mtpilot_film8_${heldout}.txt"
  echo "FILM8_EXCL_${heldout} ${ROOT}/${ckpt}" > "${list}"
  seen=(neighbor_matching political_leaning)
  [[ "${heldout}" != facebook_page_reference ]] && seen+=(page_category)
  [[ "${heldout}" != twibot20 ]] && seen+=(bot_detection)
  [[ "${heldout}" != ukr_rus_suspended ]] && seen+=(account_suspension)
  seen_csv="$(IFS=,; echo "${seen[*]}")"
  python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
    --model-list "${list}" --python python3 --data-root /dataMeR1/phil/data \
    --datasets "$(IFS=,; echo "${TARGETS[*]}")" \
    --graph-filenames facebook_page_reference=page_reference_graph.pt \
    --tasks pl --shots 3 --pl-dataset-len-cap 25 --batch-size 4 --workers 2 \
    --gpus "${gpu},${gpu}" --continue-on-error -- \
    --task_embedding_dim 8 --task_embedding_dropout 0.25 \
    --task_embedding_fusion film --task_embedding_seen_families "${seen_csv}" \
    > "/tmp/mtpilot_film8_eval_${heldout}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done
