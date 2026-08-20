#!/usr/bin/env bash
# Stream checkpoint evaluation on GPU 1 while GPU 0 trains the ladder.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
STATE_DIR="${STATE_DIR:-${REPO_ROOT}/state}"
GPU="${GPU:-1}"
POLL_SECONDS="${POLL_SECONDS:-30}"
LOG_DIR="${SCRIPT_DIR}/run_logs"
ALL_DATASETS="ukr_rus_twitter,covid19_twitter,midterm,covid_political,election2020,ukr_rus_suspended,twibot20,cp_hk_twitter"
mkdir -p "${LOG_DIR}"

[[ "${GPU}" == "1" ]] || {
  echo "this streaming protocol reserves GPU 1 for evaluation" >&2
  exit 2
}
python3 "${SCRIPT_DIR}/make_configs.py" --check >/dev/null

checkpoint_for() {
  local prefix="$1" step="$2" run_dir checkpoint
  while true; do
    run_dir="$(find "${STATE_DIR}" -maxdepth 1 -type d -name "${prefix}_*" -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | head -1 | cut -d' ' -f2-)"
    checkpoint="${run_dir:+${run_dir}/checkpoint/state_dict_${step}.ckpt}"
    if [[ -n "${checkpoint}" && -s "${checkpoint}" ]]; then
      local first_size second_size
      first_size="$(stat -c %s "${checkpoint}")"
      sleep 5
      second_size="$(stat -c %s "${checkpoint}")"
      if [[ "${first_size}" == "${second_size}" && "${second_size}" -gt 0 ]]; then
        printf '%s\n' "${checkpoint}"
        return 0
      fi
    fi
    sleep "${POLL_SECONDS}"
  done
}

tail -n +2 "${SCRIPT_DIR}/manifest.tsv" | while IFS=$'\t' read -r rung prefix newcomer _graph_id _filename; do
  for step in 10000 20000 30000 40000; do
    echo "[watch] waiting for ${prefix} step ${step}" >&2
    checkpoint="$(checkpoint_for "${prefix}" "${step}")"
    if [[ "${step}" == "40000" ]]; then
      datasets="${ALL_DATASETS}"
      scope="all8"
    else
      datasets="ukr_rus_twitter,${newcomer},cp_hk_twitter"
      datasets="$(awk -v RS=, '!seen[$0]++ { out = out (out ? "," : "") $0 } END { print out }' <<<"${datasets}")"
      scope="sentinel"
    fi
    label="${prefix}_step${step}"
    log="${LOG_DIR}/eval_${label}_${scope}_gpu${GPU}.log"
    echo "[gpu ${GPU}] evaluating ${label} on ${datasets} -> ${log}" >&2
    if "${SCRIPT_DIR}/eval_checkpoint_tucker.sh" \
        "${label}" "${checkpoint}" "${datasets}" "${GPU}" >"${log}" 2>&1; then
      echo "[gpu ${GPU}] OK ${label} ${scope}" >&2
    else
      echo "[gpu ${GPU}] FAIL ${label} ${scope} (continuing watcher)" >&2
    fi
  done
done
