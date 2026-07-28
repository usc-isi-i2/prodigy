#!/usr/bin/env bash
# Wait until all 5 sweep runs have a final checkpoint (step >= FINAL_STEP), then
# auto-run: make_model_list -> eval (NM 3-shot/30-way on ukr+covid) -> build_sweep table.
# Meant to run detached in tmux so results are ready without babysitting.
#   DEVICE=2 tmux new-session -d -s pxsrc_watch \
#     'export PATH="/home/mhchu/miniconda3/bin:$PATH"; bash .../watch_and_eval.sh'
# Env: STATE_DIR (default <repo>/state), DEVICE (eval GPU, default 2),
#      FINAL_STEP (default 110000), POLL_SEC (default 300).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
DEVICE="${DEVICE:-2}"
FINAL_STEP="${FINAL_STEP:-110000}"
POLL_SEC="${POLL_SEC:-300}"
WLOG="${SCRIPT_DIR}/run_logs/watcher_$(date +%Y%m%d_%H%M%S).log"
PREFIXES=(nm_pxsrc_p000 nm_pxsrc_p010 nm_pxsrc_p025 nm_pxsrc_p050 nm_pxsrc_p100)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "${WLOG}"; }

max_step() {  # max checkpoint step for a run prefix, or -1
  local d; d="$(ls -dt "${STATE_DIR}/$1_"*/ 2>/dev/null | head -1)"; [ -z "$d" ] && { echo -1; return; }
  ls "${d}checkpoint/"state_dict_*.ckpt 2>/dev/null \
    | sed -E 's#.*/state_dict_([0-9]+)\.ckpt$#\1#' | sort -n | tail -1 | grep -E '^[0-9]+$' || echo -1
}

all_done() {
  for p in "${PREFIXES[@]}"; do
    [ "$(max_step "$p")" -ge "${FINAL_STEP}" ] 2>/dev/null || return 1
  done
  return 0
}

log "watcher up. waiting for final ckpt (step>=${FINAL_STEP}) in: ${PREFIXES[*]}"
while ! all_done; do
  log "not ready: $(for p in "${PREFIXES[@]}"; do echo -n "${p##nm_pxsrc_}=$(max_step "$p") "; done)"
  sleep "${POLL_SEC}"
done
log "all final ckpts present. building model_list + eval on GPU ${DEVICE}..."

bash "${SCRIPT_DIR}/make_model_list.sh" 2>&1 | tee -a "${WLOG}"
bash "${SCRIPT_DIR}/eval_tucker.sh" --device "${DEVICE}" 2>&1 | tee -a "${WLOG}"

PY=/home/mhchu/miniconda3/envs/prodigy/bin/python3
"${PY}" "${SCRIPT_DIR}/build_sweep.py" --log-root "${REPO_ROOT}/log" \
  --shots 3 --n-way 30 --metric all --out-csv "${SCRIPT_DIR}/sweep.csv" \
  2>&1 | tee "${SCRIPT_DIR}/RESULTS.txt" | tee -a "${WLOG}"
log "DONE. table -> RESULTS.txt, long-form -> sweep.csv"
