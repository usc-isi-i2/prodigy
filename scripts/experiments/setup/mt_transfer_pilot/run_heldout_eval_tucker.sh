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
  target="${TARGETS[$i]}"; gpu="${GPUS[$i]}"; list="/tmp/mtpilot_heldout_${target}.txt"
  : > "${list}"
  for arm in MT NM NM_MT; do
    run="$(ls -dt state/mtpilot_${arm}_heldout_${target}_[0-9]*/ | head -n1)"
    ckpt="${run}checkpoint/state_dict_900.ckpt"
    [[ -f "${ckpt}" ]] || { echo "missing ${ckpt}" >&2; exit 1; }
    echo "HELDOUT_${arm} ${ROOT}/${ckpt}" >> "${list}"
  done
  filename_args=()
  [[ "${target}" == facebook_page_reference ]] && filename_args=(--graph-filenames facebook_page_reference=page_reference_graph.pt)
  python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
    --model-list "${list}" --python python3 --data-root /dataMeR1/phil/data \
    --datasets "${target}" "${filename_args[@]}" --tasks pl --shots 3 \
    --pl-dataset-len-cap 25 --batch-size 4 --workers 2 \
    --gpus "${gpu},${gpu},${gpu}" --continue-on-error \
    > "/tmp/mtpilot_heldout_eval_${target}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done
