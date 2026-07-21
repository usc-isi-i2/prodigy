#!/usr/bin/env bash
# Experiment (a): build model_list_source.txt with the final checkpoint of the
# TwiBot-20 NM run (prefix nm_twibot20), to evaluate it on every other graph.
#
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list_source.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${SCRIPT_DIR}/model_list_source.txt"
PREFIX="${PREFIX:-nm_twibot20}"

d="$(ls -dt "${STATE_DIR}/${PREFIX}_"*/ 2>/dev/null | head -n1 || true)"
[[ -z "$d" ]] && { echo "ERROR: no run for ${PREFIX} under ${STATE_DIR}" >&2; exit 1; }
c="$(ls "${d}checkpoint/"state_dict_*.ckpt 2>/dev/null \
     | sed -E 's#.*/state_dict_([0-9]+)\.ckpt$#\1 &#' | sort -n -k1,1 | tail -n1 | cut -d' ' -f2-)"
[[ -z "$c" ]] && { echo "ERROR: no checkpoints in ${d}checkpoint/" >&2; exit 1; }

echo "${PREFIX} ${c}" > "${OUT}"
echo "wrote ${OUT}:" >&2; cat "${OUT}" >&2
