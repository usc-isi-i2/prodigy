#!/usr/bin/env bash
# Node regression for the NM ladder, through the REPAIRED frozen-encoder probe
# (scripts/eval/regression_probe_sweep.py). Do NOT use the runner's `--tasks reg`
# path: it is void (setup/regression_probe_repair/README.md). That path builds a
# `regression_head` that is in no checkpoint, loads with strict=False so it stays at
# random init, and `--eval_only` never takes an optimizer step -- so the number it
# reports is a fixed random projection of the frozen embedding.
#
# regression_probe_sweep inverts the loop the same way pair_link_sweep does: the graph
# is loaded once and every checkpoint is scored against ONE shared episode set, so all
# 21 rungs see identical support and query nodes, and the raw-feature floor is computed
# on those same nodes. 4 graph passes, not 252 jobs.
#
#   bash run_reg_probe_sweep.sh                      # 4 graphs, serial, GPU 0
#   GPU=1 bash run_reg_probe_sweep.sh                # pin a different GPU
#   DATASETS="midterm,twibot20" bash run_reg_probe_sweep.sh
#   DRY_RUN=1 bash run_reg_probe_sweep.sh            # print the invocations only
#
# Shots / targets / transform are copied from run_eval_sweep.sh verbatim, so a probe
# row is the same episode design the void runner rows claimed to measure.
#
# Every run emits a `__features_only__` row per (dataset, target): the raw-feature floor
# on the same episodes. An encoder that does not clear it carries no signal the raw
# features did not already have. run_gate.sh checks that floor against the published
# features_only_floor.csv before any of this is trustworthy -- run it first.
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
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/scripts/experiments/analysis/transfer/ablations/downstream/one_hop/nm_ladder_downstream/data/reg_probe}"
GPU="${GPU:-0}"

# The 4 graphs carrying the profile panel, with their catalog relative_paths. Note
# twibot20 is retweet_graph.pt, not the _parquet artifact -- same mapping as
# run_pair_lp_sweep.sh, which is the source of truth for these.
graph_path_of() {
  case "$1" in
    ukr_rus_twitter) echo "ukr_rus_twitter/graphs/retweet_graph_parquet.pt" ;;
    covid19_twitter) echo "covid19_twitter/graphs/retweet_graph_parquet.pt" ;;
    midterm)         echo "midterm/graphs/retweet_graph_parquet.pt" ;;
    twibot20)        echo "twibot20/graphs/retweet_graph.pt" ;;
    *)               echo "" ;;
  esac
}
# Smallest first, so a configuration mistake surfaces in minutes rather than after the
# 23M-node covid pass.
DATASETS="${DATASETS:-midterm,twibot20,ukr_rus_twitter,covid19_twitter}"

mkdir -p "${OUT_DIR}"

IFS=',' read -r -a ds_arr <<< "${DATASETS}"
for ds in "${ds_arr[@]}"; do
  rel="$(graph_path_of "${ds}")"
  [[ -n "${rel}" ]] || { echo "unknown dataset (no profile panel): ${ds}" >&2; exit 2; }
  echo "=== reg-probe sweep: ${ds} (21 models x 3 targets, shared episode set) ==="
  cmd=(python3 scripts/eval/regression_probe_sweep.py
       --graph "${DATA_ROOT}/${rel}"
       --dataset "${ds}"
       --model-list "${ML}"
       --out-dir "${OUT_DIR}"
       --targets "${TARGETS:-followers_count,statuses_count,account_age_days}"
       --transform log1p --shots 10 --n-query 12 --episodes 500 --alpha 1.0
       --device "cuda:${GPU}")
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'DRY_RUN: '; printf '%q ' "${cmd[@]}"; printf '\n'
  else
    "${cmd[@]}"
  fi
done

echo "NM_LADDER_DOWNSTREAM_REG_PROBE_DONE"
