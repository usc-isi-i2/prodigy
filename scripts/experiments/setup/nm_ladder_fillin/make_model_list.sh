#!/usr/bin/env bash
# Write model_list.txt pointing each fill-in rung (4src..7src) at its MATCHED-40k
# checkpoint (state_dict_40000.ckpt) — the same budget the merged NM ladder and the
# single-source matrix were evaluated at, so results drop straight into the table.
#
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy/state ./make_model_list.sh
#   STEP=40000 ./make_model_list.sh        # override the eval step
#
# If a run has no state_dict_<STEP>.ckpt it is reported and skipped (WARN).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
STEP="${STEP:-40000}"
OUT="${SCRIPT_DIR}/model_list.txt"

PREFIXES=(
  nm_ladder_4src
  nm_ladder_5src
  nm_ladder_6src
  nm_ladder_7src
)

run_dir_of() { ls -dt "${STATE_DIR}/$1_"*/ 2>/dev/null | head -n1 || true; }

: > "${OUT}"
missing=0
for prefix in "${PREFIXES[@]}"; do
  d="$(run_dir_of "$prefix")"
  if [[ -z "$d" ]]; then
    echo "WARN: no run dir for ${prefix} under ${STATE_DIR}" >&2
    missing=1; continue
  fi
  ckpt="${d}checkpoint/state_dict_${STEP}.ckpt"
  if [[ -f "$ckpt" ]]; then
    echo "${prefix} ${ckpt}" >> "${OUT}"
  else
    echo "WARN: ${prefix}: no state_dict_${STEP}.ckpt in ${d}checkpoint/ (did it reach ${STEP} steps?)" >&2
    missing=1
  fi
done

echo "wrote ${OUT}:" >&2
cat "${OUT}" >&2
[[ "${missing}" == "1" ]] && echo "NOTE: some models missing at step ${STEP} — see WARN lines above." >&2
exit 0
