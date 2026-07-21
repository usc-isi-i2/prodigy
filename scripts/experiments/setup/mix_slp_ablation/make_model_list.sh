#!/usr/bin/env bash
# Write model_list.txt for the mix_slp_ablation 2x2: "<ARM> <ckpt>" rows for the
# multitask_ssl_rotation 30k checkpoints (MIX treatment + NM control).
#
# Unlike multitask_ssl_rotation/make_model_list.sh (newest run dir by mtime,
# highest-step ckpt), this PINS the checkpoint step to CKPT_STEP (default 30000)
# so the ablation evaluates exactly the checkpoints behind the published
# FINDINGS numbers, and it errors instead of silently picking another step.
#
#   STATE_DIR=/dataMeR1/phil/gfm/prodigy-mtr/state \
#     bash scripts/experiments/setup/mix_slp_ablation/make_model_list.sh
#   ARMS="NM MIX" CKPT_STEP=30000 STATE_DIR=... bash .../make_model_list.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${STATE_DIR:?set STATE_DIR to the dir holding mtr_<ARM>_<ts>/ runs}"
OUT="${SCRIPT_DIR}/model_list.txt"
ARMS="${ARMS:-NM MIX}"
CKPT_STEP="${CKPT_STEP:-30000}"

: > "${OUT}"
for arm in ${ARMS}; do
  # newest run dir for the arm (there is one real run per arm; smoke runs were
  # deleted). Fail loudly if 0 or the pinned-step ckpt is missing.
  d="$(ls -dt "${STATE_DIR}/mtr_${arm}_"*/ 2>/dev/null | head -n1 || true)"
  [[ -n "${d}" ]] || { echo "ERROR: no run dir ${STATE_DIR}/mtr_${arm}_*" >&2; exit 1; }
  n="$(ls -d "${STATE_DIR}/mtr_${arm}_"*/ 2>/dev/null | wc -l)"
  [[ "${n}" -gt 1 ]] && echo "WARN: ${n} run dirs for mtr_${arm}_*, using newest: ${d}" >&2
  c="${d}checkpoint/state_dict_${CKPT_STEP}.ckpt"
  [[ -f "${c}" ]] || { echo "ERROR: missing ${c}" >&2; exit 1; }
  echo "${arm} ${c}" >> "${OUT}"
done
echo "wrote ${OUT}:" >&2; cat "${OUT}" >&2
