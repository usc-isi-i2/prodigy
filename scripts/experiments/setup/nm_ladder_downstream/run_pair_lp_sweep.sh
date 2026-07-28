#!/usr/bin/env bash
# Static link prediction for the NM ladder, through the REPAIRED pair-conditioned
# evaluator (scripts/eval/pair_link_sweep.py). Do not use the runner's `--tasks slp`
# path: it is void (AGENTS.md, analysis/multitask_ssl/FINDINGS_rescore.md).
#
# pair_link_sweep inverts the loop -- the graph is loaded once and every checkpoint is
# scored against ONE shared pair set, so all 21 rungs see identical positives and
# identical negatives, and the CN / Adamic-Adar / preferential-attachment / Jaccard /
# raw-feature-cosine floors are computed on that same set. 5 graph passes, not 105 jobs.
#
#   bash run_pair_lp_sweep.sh                       # 5 graphs, serial, GPU 0
#   GPU=1 bash run_pair_lp_sweep.sh                 # pin a different GPU
#   DATASETS="midterm,twibot20" bash run_pair_lp_sweep.sh
#   DRY_RUN=1 bash run_pair_lp_sweep.sh             # print the invocations only
#
# Serial by default on purpose: the adjacency build on ukr_rus/covid19 is the memory
# peak of this repo's eval path, and two of them at once is how the box starts swapping.
#
# Per-row validity reads, from the emitted CSV: leakage_edges=0, endpoint_sensitivity
# ~1.000, endpoint_permutation_auc ~0.5. Any deviation voids that row.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ML="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
[[ -f "${ML}" ]] || { echo "model list not found: ${ML} (run make_model_list.py)" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/scripts/experiments/analysis/nm_ladder_downstream/data/pair_lp}"
GPU="${GPU:-0}"

# The 5 graphs carrying a static_holdout view, with their catalog relative_paths
# (docs/graph_catalog.json is the source of truth for these). A `case` rather than an
# associative array: macOS ships bash 3.2, and this script should stay dry-runnable on
# the laptop before it is handed to Tucker.
graph_path_of() {
  case "$1" in
    ukr_rus_twitter) echo "ukr_rus_twitter/graphs/retweet_graph_parquet.pt" ;;
    covid19_twitter) echo "covid19_twitter/graphs/retweet_graph_parquet.pt" ;;
    midterm)         echo "midterm/graphs/retweet_graph_parquet.pt" ;;
    twibot20)        echo "twibot20/graphs/retweet_graph.pt" ;;
    cp_hk_twitter)   echo "cp_hk_twitter/graphs/retweet_graph.pt" ;;
    *)               echo "" ;;
  esac
}
# Smallest first, so a configuration mistake surfaces in minutes rather than after the
# 23M-node covid pass.
DATASETS="${DATASETS:-cp_hk_twitter,twibot20,midterm,ukr_rus_twitter,covid19_twitter}"

mkdir -p "${OUT_DIR}"

IFS=',' read -r -a ds_arr <<< "${DATASETS}"
for ds in "${ds_arr[@]}"; do
  rel="$(graph_path_of "${ds}")"
  [[ -n "${rel}" ]] || { echo "unknown dataset (no static_holdout): ${ds}" >&2; exit 2; }
  echo "=== pair-LP sweep: ${ds} (21 models, shared pair set) ==="
  cmd=(python3 scripts/eval/pair_link_sweep.py
       --graph "${DATA_ROOT}/${rel}"
       --dataset "${ds}"
       --model-list "${ML}"
       --out-dir "${OUT_DIR}"
       --negative-kinds "${NEGATIVE_KINDS:-degree_matched,random,hard_2hop}"
       --device "cuda:${GPU}")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN: '; printf '%q ' "${cmd[@]}"; printf '\n'
  else
    "${cmd[@]}"
  fi
done

echo "NM_LADDER_DOWNSTREAM_PAIR_LP_DONE"
