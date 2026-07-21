#!/usr/bin/env bash
# 2x2 ablation (topology_feature_ssl README T2) — the PRIMARY, seed-robust
# evidence. Cells = fraction of the real/real benchmark retained under each
# corruption of {real,random} features x {real,rewired} edges:
#
#   real feat  · real edge   -> the normal eval sweep (run_eval_sweep.sh); ref=1.00
#   random feat· real edge   -> --ablate-features noise      (_ablN)
#   real feat  · rewired edge-> --ablate-edges rewire        (_ablE)
#   random feat· rewired edge-> both                         (_ablNE)
#
# "Learns both" ⇒ the rep drops materially under BOTH halves (not features-only
# like NM, which survives rewired-edge and collapses only under random-feat).
# This script runs the THREE corrupted conditions on the feature tasks (reg + pl)
# plus static-LP; the intact reference comes from run_eval_sweep.sh. Frozen
# encoders only — no training. Offline/free apart from GPU forward passes.
#
# Usage (Tucker, prodigy env, tmux):
#   MODEL_LIST=scripts/experiments/topology_feature_ssl/model_list.txt \
#     bash scripts/experiments/setup/topology_feature_ssl/run_2x2_ablation.sh --gpus 0,1,2,3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ML="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
[[ -f "${ML}" ]] || { echo "model list not found: ${ML} (run make_model_list.sh)" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

RUNNER=scripts/eval/eval_ckpts_all_graph_tasks_tucker.py
# Set STRUCTURAL=directed3 when the model list is E1-E4 (E1 input_dim 771).
STRUCTURAL_ARGS=()
[[ -n "${STRUCTURAL:-}" ]] && STRUCTURAL_ARGS=(--structural-features "${STRUCTURAL}")
GNN_TYPE_ARGS=()
[[ -n "${GNN_TYPE:-}" ]] && GNN_TYPE_ARGS=(--gnn-type "${GNN_TYPE}")  # E2: sage_multi
NO_BN_ARGS=()
[[ -n "${NO_BN_ENCODER:-}" ]] && NO_BN_ARGS=(--no-bn-encoder)          # E2b: drop-BN retry
COMMON=(--model-list "${ML}" --python python3
        --data-root /dataMeR1/phil/data
        --datasets midterm,ukr_rus_twitter,covid19_twitter,twibot20,election2020
        --continue-on-error "${STRUCTURAL_ARGS[@]}" "${GNN_TYPE_ARGS[@]}" "${NO_BN_ARGS[@]}")

run_condition() {  # $1..: extra ablation flags (label echoed)
  local label="$1"; shift
  echo "=== 2x2 condition: ${label} ($*) ==="
  # feature tasks (the discriminating half of the 2x2)
  python3 "${RUNNER}" "${COMMON[@]}" "$@" --tasks reg --shots 10 --reg-transform log1p \
    --reg-targets followers_count,statuses_count,account_age_days "${GPU_ARGS[@]}"
  python3 "${RUNNER}" "${COMMON[@]}" "$@" --tasks pl  --shots 10 "${GPU_ARGS[@]}"
  # static-LP: expected to collapse under rewired-edge for every arm (LP IS
  # topology) — a sanity check, not a discriminator.
  python3 "${RUNNER}" "${COMMON[@]}" "$@" --tasks slp --shots 0 --slp-n-query 4 "${GPU_ARGS[@]}"
}

GPU_ARGS=("$@")  # pass --gpus ... straight through to every condition
run_condition "random-feat"          --ablate-features noise
run_condition "rewired-edge"         --ablate-edges rewire
run_condition "random-feat+rewired"  --ablate-features noise --ablate-edges rewire

# join intact (from the normal sweep) with the ablated runs -> retained fraction
python3 "${SCRIPT_DIR}/parse_2x2.py" \
  --log-root /dataMeR1/phil/gfm/prodigy/log --model-list "${ML}" \
  --out scripts/experiments/analysis/topology_feature_ssl/data/ablation_2x2.csv

echo "TOPOLOGY_FEATURE_SSL_2X2_DONE"
