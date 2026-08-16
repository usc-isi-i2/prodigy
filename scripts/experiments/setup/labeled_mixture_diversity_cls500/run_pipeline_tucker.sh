#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPUS="${GPUS:-0 1}" bash "${SCRIPT_DIR}/run_train_tucker.sh"
GPUS="${GPUS:-0 1}" bash "${SCRIPT_DIR}/run_eval_tucker.sh"
