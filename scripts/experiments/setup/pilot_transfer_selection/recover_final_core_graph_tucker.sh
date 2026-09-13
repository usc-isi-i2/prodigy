#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
DATA_ROOT="/dataMeR1/phil/data"
PYTHON="${PYTHON:-/home/mhchu/miniconda3/envs/prodigy/bin/python}"
PREFLIGHT_ONLY="${PREFLIGHT_ONLY:-0}"
MERGED="${DATA_ROOT}/merged/graphs/ukr_rus_covid_midterm_all9_facebook_graph.pt"
SPLIT="${DATA_ROOT}/merged/graphs/ukr_rus_covid_midterm_all9_facebook_final_core_split_seed0.pt"
CONFIG="${REPO_ROOT}/scripts/graph_construction/merge_ukr_rus_covid_midterm_all9_facebook.yaml"

cd "$REPO_ROOT"

[[ ! -e "$MERGED" ]] || { echo "refusing to overwrite $MERGED" >&2; exit 2; }
[[ ! -e "$SPLIT" ]] || { echo "refusing to overwrite $SPLIT" >&2; exit 2; }

while read -r expected relative; do
  source_path="${DATA_ROOT}/${relative}"
  [[ -f "$source_path" ]] || { echo "missing immutable source $source_path" >&2; exit 2; }
  observed="$(stat -c %s "$source_path")"
  [[ "$observed" == "$expected" ]] || {
    echo "source size drift: $source_path expected=$expected observed=$observed" >&2
    exit 2
  }
done <<'EOF'
37006016290 ukr_rus_twitter/graphs/retweet_graph_parquet.pt
78267044770 covid19_twitter/graphs/retweet_graph_parquet.pt
1123399802 midterm/graphs/retweet_graph_parquet.pt
247456395 covid_political/graphs/retweet_graph.pt
311565265 election2020/graphs/retweet_graph.pt
182861567 ukr_rus_suspended/graphs/retweet_graph.pt
591854240 twibot20/graphs/retweet_graph.pt
1107820534 cp_hk_twitter/graphs/retweet_graph.pt
394622154 facebook_page_reference/graphs/page_reference_structural.pt
EOF

available_kib="$(df --output=avail -k "$DATA_ROOT" | tail -n 1 | tr -d ' ')"
available_mem_kib="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
(( available_kib >= 300 * 1024 * 1024 )) || {
  echo "need at least 300 GiB free under $DATA_ROOT" >&2; exit 2;
}
(( available_mem_kib >= 768 * 1024 * 1024 )) || {
  echo "need at least 768 GiB available RAM" >&2; exit 2;
}
echo "preflight passed: immutable source sizes, free disk, and available RAM"
[[ "$PREFLIGHT_ONLY" == 1 ]] && exit 0

mkdir -p "$(dirname "$MERGED")"

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
