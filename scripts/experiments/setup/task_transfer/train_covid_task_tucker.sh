#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-nm}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/dataMeR1/phil/gfm/prodigy}"
COVID_ROOT="${COVID_ROOT:-/dataMeR1/phil/data/covid19_twitter/graphs}"
GRAPH_FILENAME="${GRAPH_FILENAME:-retweet_graph_parquet.pt}"
DEVICE="${DEVICE:-0}"

case "${TASK}" in
  nm|neighbor_matching) CONFIG_PATH="${SCRIPT_DIR}/covid_nm_smoke.yaml" ;;
  cl|contrastive) CONFIG_PATH="${SCRIPT_DIR}/covid_cl_smoke.yaml" ;;
  fp|masked_feature_prediction) CONFIG_PATH="${SCRIPT_DIR}/covid_fp_smoke.yaml" ;;
  *)
    echo "Unknown task '${TASK}'. Use nm, cl, or fp." >&2
    exit 2
    ;;
esac

cmd=(
  python3 experiments/run_single_experiment.py
  --config "${CONFIG_PATH}"
  --root "${COVID_ROOT}"
  --graph_filename "${GRAPH_FILENAME}"
  --device "${DEVICE}"
  "$@"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate prodigy
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

cd "${REPO_ROOT}"
"${cmd[@]}"
