#!/usr/bin/env bash
# Write model_list.txt = final checkpoint of each of the 5 sweep runs.
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${SCRIPT_DIR}/model_list.txt"

run_dir_of() { ls -dt "${STATE_DIR}/$1_"*/ 2>/dev/null | head -n1 || true; }
final_ckpt() { local d; d="$(run_dir_of "$1")"; [[ -z "$d" ]] && return 1
  ls "${d}checkpoint/"state_dict_*.ckpt 2>/dev/null \
    | sed -E 's#.*/state_dict_([0-9]+)\.ckpt$#\1 &#' | sort -n -k1,1 | tail -n1 | cut -d' ' -f2-; }

: > "${OUT}"
for prefix in nm_pxsrc_p000 nm_pxsrc_p010 nm_pxsrc_p025 nm_pxsrc_p050 nm_pxsrc_p100; do
  ckpt="$(final_ckpt "$prefix" || true)"
  if [[ -n "$ckpt" ]]; then
    echo "${prefix} ${ckpt}" >> "${OUT}"
  else
    echo "WARN: no checkpoint yet for ${prefix}" >&2
  fi
done
echo "wrote ${OUT}:" >&2; cat "${OUT}" >&2
