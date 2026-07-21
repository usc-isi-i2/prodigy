#!/usr/bin/env bash
# Build the 4 intermediate merged graphs for the ladder fill-in (rungs 4-7).
# Rung 8 (all8) already exists and is NOT rebuilt. Idempotent: skips a merge whose
# output .pt already exists (pass FORCE=1 to rebuild).
#
#   ./build_merges_tucker.sh              # build 4src..7src if missing
#   FORCE=1 ./build_merges_tucker.sh      # rebuild even if present
#   DRY_RUN=1 ./build_merges_tucker.sh    # print commands only
#
# Each merge is disjoint block-concat (drop_edge_features=true) -> structure + node
# feats only, carrying graph_id for within-source NM sampling. Inputs must be 768-dim.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
MERGE_PY="${REPO_ROOT}/scripts/graph_construction/merge_disjoint_graph_pt.py"
OUT_DIR="/dataMeR1/phil/data/merged/graphs"

CONFIGS=(merge_4src.yaml merge_5src.yaml merge_6src.yaml merge_7src.yaml)

out_of() {  # extract the `out:` path from a merge yaml (no yaml dep needed)
  grep -E '^out:' "${SCRIPT_DIR}/$1" | head -n1 | sed -E 's/^out:[[:space:]]*//'
}

if [[ "${DRY_RUN:-0}" != "1" ]]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate prodigy
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
cd "${REPO_ROOT}"

for cfg in "${CONFIGS[@]}"; do
  out="$(out_of "$cfg")"
  if [[ -f "$out" && "${FORCE:-0}" != "1" ]]; then
    echo "[skip] ${cfg}: output exists -> ${out} (FORCE=1 to rebuild)" >&2
    continue
  fi
  echo "[merge] ${cfg} -> ${out}" >&2
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY: python3 ${MERGE_PY} ${SCRIPT_DIR}/${cfg}"
    continue
  fi
  python3 "${MERGE_PY}" "${SCRIPT_DIR}/${cfg}"
done
echo "[merge] done; artifacts in ${OUT_DIR}" >&2
