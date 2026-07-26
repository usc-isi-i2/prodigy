#!/usr/bin/env bash
# Enrich existing retweet-graph artifacts with benchmark targets (node regression
# panel + static-LP edge views) WITHOUT a full rebuild. Idempotent: re-running
# overwrites the added fields in place.
#
# Run in the graph-construction conda env (duckdb + pyarrow + torch), e.g.
# bio-embeddings-v001 on Tucker. Reads raw parquet; the user launches this
# (it writes artifacts).
#
#   DATA_ROOT=/dataMeR1/phil/data bash scripts/graph_construction/enrich_all_graphs.sh
#
# Per-dataset feasibility (see graph_construction/README.md):
#   midterm/ukr_rus/covid19/twibot20 -> node regression + static LP
#   cp_hk_twitter                    -> static LP only (no profile metrics)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${DATA_ROOT:-/dataMeR1/phil/data}"
PY="${PY:-python3}"
ENRICH="${REPO_ROOT}/scripts/graph_construction/enrich_graph_targets.py"

run() { echo "+ $*"; "$@"; }

# midterm (flat profile columns)
run "${PY}" "${ENRICH}" --dataset midterm \
  --graph-path "${DATA_ROOT}/midterm/graphs/retweet_graph_parquet.pt" \
  --parquet-glob "${DATA_ROOT}/midterm/parquet/*/*.parquet"

# ukr_rus_twitter (flat profile columns)
run "${PY}" "${ENRICH}" --dataset ukr_rus_twitter \
  --graph-path "${DATA_ROOT}/ukr_rus_twitter/graphs/retweet_graph_parquet.pt" \
  --parquet-glob "${DATA_ROOT}/ukr_rus_twitter/parquet/*/*.parquet"

# covid19_twitter (nested user / retweeted_status.user structs)
run "${PY}" "${ENRICH}" --dataset covid19_twitter \
  --graph-path "${DATA_ROOT}/covid19_twitter/graphs/retweet_graph_parquet.pt" \
  --parquet-glob "${DATA_ROOT}/covid19_twitter/parquet/*.parquet"

# twibot20 (node.json public_metrics)
run "${PY}" "${ENRICH}" --dataset twibot20 \
  --graph-path "${DATA_ROOT}/twibot20/graphs/retweet_graph.pt" \
  --node-json "${DATA_ROOT}/twibot20/raw/Twibot-20/node.json"

# cp_hk_twitter (static LP only; no profile metrics)
run "${PY}" "${ENRICH}" --dataset cp_hk_twitter \
  --graph-path "${DATA_ROOT}/cp_hk_twitter/graphs/retweet_graph.pt"

echo "Done. Each artifact now carries node_targets (where available) + static_background/static_holdout views."
