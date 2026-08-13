#!/usr/bin/env bash
# Capability probes (topology_feature_ssl README T3) — PRIMARY evidence.
# Linear-probe each frozen arm encoder on the planted single-rule synthetic graphs
# and report the probe AUC per rule. The probe score = classification linear-probe
# AUC on the frozen center-node representation.
#
# Prereq: build the synthetic graphs once (writes covid dict-format .pt files that
# the covid loader + eval driver consume as probe_* datasets):
#   python scripts/experiments/setup/topology_feature_ssl/make_probe_graphs.py \
#     --out-dir /dataMeR1/phil/data/synthetic_probes/graphs
#
# Usage (Tucker, prodigy env, tmux):
#   MODEL_LIST=scripts/experiments/topology_feature_ssl/model_list.txt \
#     bash scripts/experiments/setup/topology_feature_ssl/run_capability_probes.sh --gpus 0,1,2,3
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ML="${MODEL_LIST:-${SCRIPT_DIR}/model_list.txt}"
[[ -f "${ML}" ]] || { echo "model list not found: ${ML} (run make_model_list.sh)" >&2; exit 2; }

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"

PROBE_GRAPHS=/dataMeR1/phil/data/synthetic_probes/graphs
if ! ls "${PROBE_GRAPHS}"/count_threshold.pt >/dev/null 2>&1; then
  echo "building synthetic probe graphs -> ${PROBE_GRAPHS}"
  python3 "${SCRIPT_DIR}/make_probe_graphs.py" --out-dir "${PROBE_GRAPHS}"
fi

RUNNER=scripts/eval/eval_ckpts_all_graph_tasks_tucker.py
# E1-E4 encoders need structural inputs at probe time too (E1 input_dim 771):
# set STRUCTURAL=directed3 and use an E1-E4-only model list.
STRUCTURAL_ARGS=()
[[ -n "${STRUCTURAL:-}" ]] && STRUCTURAL_ARGS=(--structural-features "${STRUCTURAL}")
GNN_TYPE_ARGS=()
[[ -n "${GNN_TYPE:-}" ]] && GNN_TYPE_ARGS=(--gnn-type "${GNN_TYPE}")  # E2: sage_multi
NO_BN_ARGS=()
[[ -n "${NO_BN_ENCODER:-}" ]] && NO_BN_ARGS=(--no-bn-encoder)          # E2b: drop-BN retry
# Linear-probe classification (frozen rep -> linear head) on each single-rule graph.
python3 "${RUNNER}" \
  --model-list "${ML}" --python python3 \
  --data-root /dataMeR1/phil/data \
  --datasets probe_count_threshold,probe_in_degree,probe_out_degree,probe_existence,probe_conjunction \
  --tasks pl --pl-linear-probe --shots 50 \
  --continue-on-error "${STRUCTURAL_ARGS[@]}" "${GNN_TYPE_ARGS[@]}" "${NO_BN_ARGS[@]}" "$@"

python3 "${SCRIPT_DIR}/parse_capability_probes.py" \
  --log-root /dataMeR1/phil/gfm/prodigy/log --model-list "${ML}" \
  --out scripts/experiments/analysis/objectives/topology_vs_features/topology_feature_ssl/data/capability_probes.csv

echo "TOPOLOGY_FEATURE_SSL_CAPABILITY_PROBES_DONE"
