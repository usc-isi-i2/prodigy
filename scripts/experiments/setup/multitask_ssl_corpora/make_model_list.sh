#!/usr/bin/env bash
# Write model_list.txt for the multitask_ssl_corpora eval sweep: one line per run,
# "<corpus>_<ARM> <30k ckpt>", keyed by corpus_arm (cov_NM ... all8_MIX) so the
# benchmark CSVs carry model=cov_NM/all8_MIX/... and per-corpus MIX-vs-control
# deltas are direct table subtractions.
#
# Selects checkpoint/state_dict_30000.ckpt EXPLICITLY: the trainer never saves the
# final (40k) step (off-by-one), so 30k is the terminal ckpt — and the original
# multitask_ssl_rotation arms were all evaluated at 30k, so matched-at-30k is the
# parity point. Errors out if a 30k ckpt is missing rather than silently taking a
# different step.
#
# The run-dir glob is anchored to msc_<corpus>_<ARM>_<timestamp> (timestamp must
# start with a digit) so smoke/stale dirs like msc_cov_NM_smoke_* cannot match;
# newest matching dir by mtime wins.
#
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy-msc/state ./make_model_list.sh
#   RUNS="cov_NM all8_MIX" STATE_DIR=... ./make_model_list.sh   # restrict
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"   # setup/<name> is 4 levels below repo root
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${SCRIPT_DIR}/model_list.txt"
RUNS="${RUNS:-cov_NM cov_CL cov_FP cov_MIX all8_NM all8_CL all8_FP all8_MIX}"

ckpt_30k() {  # RUN(corpus_ARM) -> newest run dir's state_dict_30000.ckpt
  local d
  d="$(ls -dt "${STATE_DIR}/msc_$1_"[0-9]*/ 2>/dev/null | head -n1 || true)"
  [[ -z "${d}" ]] && return 1
  local c="${d}checkpoint/state_dict_30000.ckpt"
  [[ -f "${c}" ]] || return 1
  echo "${c}"
}

: > "${OUT}"
missing=0
for run in ${RUNS}; do
  if c="$(ckpt_30k "${run}")"; then
    echo "${run} ${c}" >> "${OUT}"
  else
    echo "ERROR: no 30k checkpoint for msc_${run}_* under ${STATE_DIR}" >&2
    missing=1
  fi
done
[[ "${missing}" -eq 0 ]] || exit 1
echo "wrote ${OUT}:" >&2; cat "${OUT}" >&2
