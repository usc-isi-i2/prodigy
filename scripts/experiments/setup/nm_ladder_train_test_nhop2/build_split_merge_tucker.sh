#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/merge_all8_static_split.yaml"
OUT="/dataMeR1/phil/data/merged/graphs/ukr_rus_covid_midterm_all8_static_split_retweet_graph.pt"

if [[ -e "${OUT}" && "${FORCE:-0}" != "1" ]]; then
  echo "refusing to overwrite existing artifact: ${OUT}" >&2
  echo "inspect it first; set FORCE=1 only for an intentional rebuild" >&2
  exit 2
fi

cmd=(python3 scripts/graph_construction/merge_disjoint_graph_pt.py "${CONFIG}")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'DRY:'; printf ' %q' "${cmd[@]}"; printf '\n'
  exit 0
fi

export PATH="/home/mhchu/miniconda3/bin:${PATH}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bio-embeddings-v001
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
cd "${REPO_ROOT}"
"${cmd[@]}"
