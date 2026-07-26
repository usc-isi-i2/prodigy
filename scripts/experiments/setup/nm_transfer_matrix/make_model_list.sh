#!/usr/bin/env bash
# Build model_list.txt for the ukr/cov transfer matrix.
#
# Single-source runs train 60k steps (final ckpt ~50k); merged trains 120k (final
# ~110k). For a FAIR comparison the merged model is evaluated at TWO checkpoints:
#   nm_matrix_merged_match : same step count as the single-source runs -> matched total compute
#   nm_matrix_merged_full  : merged's final checkpoint                 -> matched per-domain exposure
#
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
OUT="${SCRIPT_DIR}/model_list.txt"

run_dir_of() { ls -dt "${STATE_DIR}/$1_"*/ 2>/dev/null | head -n1 || true; }
final_ckpt() {  # prefix -> highest-step ckpt path (sort by numeric step in basename)
  local d; d="$(run_dir_of "$1")"; [[ -z "$d" ]] && return 1
  ls "${d}checkpoint/"state_dict_*.ckpt 2>/dev/null \
    | sed -E 's#.*/state_dict_([0-9]+)\.ckpt$#\1 &#' | sort -n -k1,1 | tail -n1 | cut -d' ' -f2-
}
step_of() { basename "$1" | sed -E 's/state_dict_([0-9]+)\.ckpt/\1/'; }
ckpt_at() { local d; d="$(run_dir_of "$1")"; [[ -z "$d" ]] && return 1; local p="${d}checkpoint/state_dict_$2.ckpt"; [[ -f "$p" ]] && echo "$p"; }

: > "${OUT}"

# Single-source: final checkpoint.
for prefix in nm_matrix_ukr nm_matrix_covid; do
  c="$(final_ckpt "$prefix")" || { echo "WARN: no run for $prefix" >&2; continue; }
  echo "$prefix $c" >> "${OUT}"
done

# Matched-compute step = single-source final step (use covid's).
match_step="$(step_of "$(final_ckpt nm_matrix_covid)")"
echo "matched-compute step = ${match_step}" >&2

# Merged: *_match (at match_step) and *_full (final).
full="$(final_ckpt nm_matrix_merged)" || { echo "WARN: no run for nm_matrix_merged" >&2; }
if [[ -n "${full:-}" ]]; then
  matched="$(ckpt_at nm_matrix_merged "$match_step" || true)"
  [[ -n "$matched" ]] && echo "nm_matrix_merged_match $matched" >> "${OUT}" || echo "WARN: no merged ckpt at ${match_step}" >&2
  echo "nm_matrix_merged_full $full" >> "${OUT}"
fi

echo "wrote ${OUT}:" >&2; cat "${OUT}" >&2
