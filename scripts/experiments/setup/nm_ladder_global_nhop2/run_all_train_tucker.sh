#!/usr/bin/env bash
# Train the seven nontrivial naive-global ladder rungs sequentially on GPU 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/run_logs"
mkdir -p "${LOG_DIR}"

GPU="${GPU:-0}"
[[ "${GPU}" == "0" ]] || {
  echo "this streaming protocol reserves GPU 0 for training and GPU 1 for evaluation" >&2
  exit 2
}
python3 "${SCRIPT_DIR}/make_configs.py" --check >/dev/null

ALL_CONFIGS=()
while IFS= read -r config; do
  [[ -n "${config}" ]] && ALL_CONFIGS+=("${config}")
done < <(python3 "${SCRIPT_DIR}/make_configs.py" --list-configs)
is_skipped() { [[ " ${SKIP:-} " == *" $1 "* ]]; }
stamp="$(date +%Y%m%d_%H%M%S)"

for config in "${ALL_CONFIGS[@]}"; do
  if is_skipped "${config}"; then
    echo "skipping ${config}" >&2
    continue
  fi
  name="$(basename "${config%.yaml}")"
  log="${LOG_DIR}/${name}_gpu${GPU}_${stamp}.log"
  echo "[gpu ${GPU}] launching ${name} -> ${log}" >&2
  if DRY_RUN="${DRY_RUN:-0}" "${SCRIPT_DIR}/train_nm_tucker.sh" \
      "${config}" --device "${GPU}" "$@" >"${log}" 2>&1; then
    echo "[gpu ${GPU}] OK ${name}" >&2
  else
    echo "[gpu ${GPU}] FAIL ${name} (see ${log})" >&2
    exit 1
  fi
done
echo "all requested global-merge rungs completed" >&2
