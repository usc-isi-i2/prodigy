#!/usr/bin/env bash
# Write model_list.txt pointing at the final checkpoint of the within-source run.
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${SCRIPT_DIR}/model_list.txt"

prefix="nm_xsrc_within_source"
run_dir="$(ls -dt "${STATE_DIR}/${prefix}_"*/ 2>/dev/null | head -n1 || true)"
[[ -z "${run_dir}" ]] && { echo "no run dir for ${prefix} under ${STATE_DIR}" >&2; exit 1; }
ckpt="$(ls "${run_dir}checkpoint/"state_dict_*.ckpt 2>/dev/null | sort -t_ -k3 -n | tail -n1 || true)"
[[ -z "${ckpt}" ]] && { echo "no checkpoint in ${run_dir}checkpoint/" >&2; exit 1; }
echo "${prefix} ${ckpt}" > "${OUT}"
echo "wrote ${OUT}:" >&2; cat "${OUT}" >&2
