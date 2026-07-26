#!/usr/bin/env bash
# Train/adapt a model on the TwiBot-20 retweet graph (bot-vs-human classification).
#
# Defaults run the smoke config. Override CONFIG_PATH for a full run, or pass
# extra CLI args (they override the YAML). Preview with DRY_RUN=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/dataMeR1/phil/gfm/prodigy}"
TWIBOT_ROOT="${TWIBOT_ROOT:-/dataMeR1/phil/data/twibot20/graphs}"
GRAPH_FILENAME="${GRAPH_FILENAME:-retweet_graph.pt}"
DEVICE="${DEVICE:-0}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/twibot20_cls_smoke.yaml}"

cmd=(
  python3 experiments/run_single_experiment.py
  --config "${CONFIG_PATH}"
  --root "${TWIBOT_ROOT}"
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
