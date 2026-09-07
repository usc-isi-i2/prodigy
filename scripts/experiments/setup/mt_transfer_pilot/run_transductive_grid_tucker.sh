#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
SOURCE_ROOT="${SOURCE_ROOT:-/dataMeR1/phil/gfm/prodigy-mtvarway}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export WANDB_MODE=offline
cd "${ROOT}"

TARGETS=(covid_political election2020 facebook_page_reference twibot20 ukr_rus_suspended)
GPUS=(2 3 2 3 2)
THRESHOLDS=(0.7 0.8 0.9)
ALPHAS=(0.25 0.5)

for threshold in "${THRESHOLDS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    ttag="${threshold/./}"; atag="${alpha/./}"
    pids=()
    for i in "${!TARGETS[@]}"; do
      heldout="${TARGETS[$i]}"; gpu="${GPUS[$i]}"
      run="$(ls -dt "${SOURCE_ROOT}"/state/mtpilot_NM_MT_varway_heldout_${heldout}_[0-9]*/ | head -n1)"
      ckpt="${run}checkpoint/state_dict_900.ckpt"
      [[ -f "${ckpt}" ]] || { echo "missing ${ckpt}" >&2; exit 1; }
      list="/tmp/mtpilot_trans_t${ttag}_a${atag}_${heldout}.txt"
      echo "TRANS_T${ttag}_A${atag}_I1_EXCL_${heldout} ${ckpt}" > "${list}"
      python3 scripts/eval/eval_ckpts_all_graph_tasks_tucker.py \
        --model-list "${list}" --python python3 --data-root /dataMeR1/phil/data \
        --datasets "$(IFS=,; echo "${TARGETS[*]}")" \
        --graph-filenames facebook_page_reference=page_reference_graph.pt \
        --tasks pl --shots 3 --pl-dataset-len-cap 25 --batch-size 4 --workers 2 \
        --gpus "${gpu},${gpu}" --continue-on-error -- \
        --transductive_refinement True --transductive_threshold "${threshold}" \
        --transductive_alpha "${alpha}" --transductive_iterations 1 \
        > "/tmp/mtpilot_trans_t${ttag}_a${atag}_${heldout}.log" 2>&1 &
      pids+=("$!")
    done
    for pid in "${pids[@]}"; do wait "${pid}"; done
    echo "DONE threshold=${threshold} alpha=${alpha}" >&2
  done
done
