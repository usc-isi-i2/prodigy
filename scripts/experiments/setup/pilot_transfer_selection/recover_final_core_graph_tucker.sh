#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
PYTHON="${PYTHON:-/home/mhchu/miniconda3/envs/prodigy/bin/python}"
MERGED="${DATA_ROOT}/merged/graphs/ukr_rus_covid_midterm_all9_facebook_graph.pt"
SPLIT="${DATA_ROOT}/merged/graphs/ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt"
CONFIG="${REPO_ROOT}/scripts/graph_construction/merge_ukr_rus_covid_midterm_all9_facebook.yaml"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$MERGED")"

[[ ! -e "$MERGED" ]] || { echo "refusing to overwrite $MERGED" >&2; exit 2; }
[[ ! -e "$SPLIT" ]] || { echo "refusing to overwrite $SPLIT" >&2; exit 2; }

"$PYTHON" scripts/graph_construction/merge_disjoint_graph_pt.py "$CONFIG"
"$PYTHON" scripts/experiments/setup/final_core/build_split_artifact_tucker.py \
  --input "$MERGED" \
  --output "$SPLIT"

"$PYTHON" - "$SPLIT" <<'PY'
from pathlib import Path
import sys
import torch

path = Path(sys.argv[1])
raw = torch.load(path, map_location="cpu")
expected_sources = [
    "ukr_rus", "covid", "midterm", "covid_political", "election2020",
    "ukr_rus_suspended", "twibot20", "cp_hk", "facebook_page_reference",
]
assert list(raw["source_graph_names"]) == expected_sources
assert tuple(raw["x"].shape) == (34_601_450, 768)
assert tuple(raw["edge_index"].shape) == (2, 191_690_740)
assert tuple(raw["edge_index_views"]["static_train"].shape) == (2, 134_183_020)
assert tuple(raw["target_edge_index_views"]["static_validation"].shape) == (2, 28_753_761)
assert tuple(raw["target_edge_index_views"]["static_test"].shape) == (2, 28_753_959)
assert raw["final_core_split_protocol"]["kind"] == "undirected_pair_70_15_15"
assert raw["final_core_split_protocol"]["seed"] == 0
print(f"verified reconstructed final-core graph: {path}")
PY
