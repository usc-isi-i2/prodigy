#!/usr/bin/env bash
set -euo pipefail

TUCKER_HOST="${TUCKER_HOST:-mhchu@10.137.32.100}"
CARC_HOST="${CARC_HOST:-eibl@10.72.0.13}"
SRC_ROOT="${SRC_ROOT:-/dataMeR1/phil/data}"
DST_ROOT="${DST_ROOT:-/scratch1/eibl/data/gte}"
LOG_DIR="${LOG_DIR:-/dataMeR1/phil/gfm/prodigy/log}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/transfer_gte_graphs_tucker_to_carc_20260707.log}"
MANIFEST="${MANIFEST:-/tmp/gte_graph_manifest.txt}"
DRY_RUN="${DRY_RUN:-0}"

ssh -A \
  -o BatchMode=yes \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=10 \
  -o ConnectTimeout=8 \
  "${TUCKER_HOST}" \
  "CARC_HOST='${CARC_HOST}' SRC_ROOT='${SRC_ROOT}' DST_ROOT='${DST_ROOT}' LOG_DIR='${LOG_DIR}' LOG_FILE='${LOG_FILE}' MANIFEST='${MANIFEST}' DRY_RUN='${DRY_RUN}' bash -s" <<'REMOTE'
set -euo pipefail

mkdir -p "${LOG_DIR}"
cat > "${MANIFEST}" <<'LIST'
covid19_twitter/graphs/retweet_graph_parquet.pt
covid19_twitter/graphs/retweet_graph_parquet.meta.json
ukr_rus_twitter/graphs/retweet_graph_parquet.pt
ukr_rus_twitter/graphs/retweet_graph_parquet.meta.json
midterm/graphs/retweet_graph_parquet.pt
midterm/graphs/retweet_graph_parquet.meta.json
cp_hk_twitter/graphs/retweet_graph.pt
cp_hk_twitter/graphs/retweet_graph.meta.json
twibot20/graphs/retweet_graph.pt
twibot20/graphs/retweet_graph.meta.json
election2020/graphs/retweet_graph.pt
election2020/graphs/retweet_graph.meta.json
covid_political/graphs/retweet_graph.pt
covid_political/graphs/retweet_graph.meta.json
ukr_rus_suspended/graphs/retweet_graph.pt
ukr_rus_suspended/graphs/retweet_graph.meta.json
merged/graphs/ukr_rus_covid_retweet_graph.pt
merged/graphs/ukr_rus_covid_retweet_graph.meta.json
merged/graphs/covid_midterm_retweet_graph.pt
merged/graphs/covid_midterm_retweet_graph.meta.json
merged/graphs/ukr_rus_covid_midterm_retweet_graph.pt
merged/graphs/ukr_rus_covid_midterm_retweet_graph.meta.json
LIST

args=(
  -rltvh
  --partial
  --append-verify
  --info=progress2,stats2
  --log-file="${LOG_FILE}"
  --files-from="${MANIFEST}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  args=(-n "${args[@]}")
fi

rsync "${args[@]}" "${SRC_ROOT}/" "${CARC_HOST}:${DST_ROOT}/"
echo "transfer_log=${LOG_FILE}"
REMOTE
