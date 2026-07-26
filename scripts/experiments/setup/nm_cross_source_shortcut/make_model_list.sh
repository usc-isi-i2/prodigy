#!/usr/bin/env bash
# Write model_list.txt for the within-source run, at TWO checkpoints for a fair
# comparison against the single-source / proportional-merged runs:
#   nm_xsrc_within_source_match : same step count as the single-source runs (matched compute)
#   nm_xsrc_within_source_full  : final checkpoint                          (per-domain exposure)
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${SCRIPT_DIR}/model_list.txt"

run_dir_of() { ls -dt "${STATE_DIR}/$1_"*/ 2>/dev/null | head -n1 || true; }
final_ckpt() { local d; d="$(run_dir_of "$1")"; [[ -z "$d" ]] && return 1
  ls "${d}checkpoint/"state_dict_*.ckpt 2>/dev/null \
    | sed -E 's#.*/state_dict_([0-9]+)\.ckpt$#\1 &#' | sort -n -k1,1 | tail -n1 | cut -d' ' -f2-; }
step_of() { basename "$1" | sed -E 's/state_dict_([0-9]+)\.ckpt/\1/'; }
ckpt_at() { local d; d="$(run_dir_of "$1")"; [[ -z "$d" ]] && return 1; local p="${d}checkpoint/state_dict_$2.ckpt"; [[ -f "$p" ]] && echo "$p"; }

# Matched-compute step = the single-source runs' final step (from the matrix experiment).
match_step="$(step_of "$(final_ckpt nm_matrix_covid)")"
echo "matched-compute step = ${match_step}" >&2

full="$(final_ckpt nm_xsrc_within_source)" || { echo "no run dir for nm_xsrc_within_source" >&2; exit 1; }
: > "${OUT}"
matched="$(ckpt_at nm_xsrc_within_source "$match_step" || true)"
[[ -n "$matched" ]] && echo "nm_xsrc_within_source_match $matched" >> "${OUT}" || echo "WARN: no within ckpt at ${match_step}" >&2
echo "nm_xsrc_within_source_full $full" >> "${OUT}"
echo "wrote ${OUT}:" >&2; cat "${OUT}" >&2
